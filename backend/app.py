"""Phase 5 — FastAPI backend.

Run from the project root:
    uvicorn backend.app:app --reload --port 8000
Then open http://localhost:8000/docs for interactive Swagger docs.

Endpoints:
    GET  /health  liveness + readiness: is the index loaded, which model, how many chunks
    POST /chat    {message, history, patient} -> {answer, sources[], retrieved, ...}
    POST /ingest  rebuild the vector index from data/raw (optionally protected by ADMIN_TOKEN)

Design notes (the WHY):
- Pydantic request/response models are the API contract: invalid input (weight -5 kg,
  unknown age group, empty message) is rejected with a 422 before any LLM tokens are
  spent, and /docs is generated from the same models.
- /chat is a plain `def`, not `async def`: the agent does blocking work (Groq HTTP
  calls, CPU-bound embeddings). FastAPI runs sync endpoints in a thread pool, so one
  slow request doesn't freeze the server's event loop.
- The API is stateless: the client sends the conversation history with each request.
  No server-side sessions means nothing to store, expire or leak between users, and
  any number of server instances can serve any request.
- The embedding model and Chroma index are warmed up at startup, so the first user
  doesn't pay a ~10 s model-loading delay.
- Errors map to honest HTTP status codes: 422 bad input, 503 LLM/rate limit or index
  rebuilding (retry later), 500 misconfiguration.
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.agent import AgentError, run_agent
from backend.config import settings
from backend.prompts import DISCLAIMER
from backend.tools import known_drugs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
for noisy in ("httpx", "sentence_transformers", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("medidose.api")

# Held while /ingest rebuilds the index. Chats arriving meanwhile get a 503 instead of
# reading a half-built collection.
_ingest_lock = threading.Lock()


# ---------- Schemas ----------

class Patient(BaseModel):
    age_group: Optional[Literal["neonate", "infant", "child", "adolescent", "adult", "elderly"]] = None
    weight_kg: Optional[float] = Field(default=None, gt=0, le=350)
    renal_impairment: bool = False
    hepatic_impairment: bool = False


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)
    patient: Optional[Patient] = None

    model_config = {"json_schema_extra": {"examples": [{
        "message": "What is the dose of gabapentin?",
        "history": [],
        "patient": {"age_group": "adult", "weight_kg": 70,
                    "renal_impairment": True, "hepatic_impairment": False},
    }]}}


class Source(BaseModel):
    id: int                     # matches the [n] markers in the answer
    drug_name: str
    section: str
    source_url: str
    score: float                # cosine relevance, 0-1
    text: str                   # the retrieved passage, so students can check the claim
    cited: bool                 # whether the answer actually cites [id]


class ToolCall(BaseModel):
    query: Optional[str] = None
    drug_name: Optional[str] = None
    section: Optional[str] = None
    n_results: int = 0


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieved: bool             # did the agent decide to retrieve? (the "agentic" decision)
    tool_calls: list[ToolCall]  # what it searched for, for transparency / the demo
    unverified_numbers: list[str]
    disclaimer: str = DISCLAIMER


class IngestRequest(BaseModel):
    download: bool = Field(default=False, description="Re-download labels from openFDA first.")


# ---------- App ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from backend.vectorstore import get_embeddings, get_vectorstore
        get_embeddings().embed_query("warm-up")
        log.info("Index ready: %d chunks in '%s'", get_vectorstore()._collection.count(),
                 settings.chroma_collection)
    except Exception as exc:  # start anyway; /health will report the problem
        log.error("Startup warm-up failed: %s", exc)
    yield


app = FastAPI(
    title="MediDose AI",
    description="Agentic RAG over FDA drug labels. Educational use only; not medical advice.",
    version="1.0.0",
    lifespan=lifespan,
)
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 that reports WHERE the input is wrong without echoing the input back.

    WHY: FastAPI's default handler includes the offending input; when that input held
    invalid Unicode (found while testing), rendering the error itself crashed with a 500.
    Not echoing user input is also safer (no reflected content, smaller responses).
    """
    errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)


@app.get("/health")
def health() -> dict:
    """Cheap readiness check: no LLM call, so it costs no tokens."""
    try:
        from backend.vectorstore import get_vectorstore
        chunks = get_vectorstore()._collection.count()
    except Exception as exc:
        log.error("Health check failed: %s", exc)
        chunks = 0
    llm_configured = bool(settings.groq_api_key) and not settings.groq_api_key.startswith("your_")
    return {
        "status": "ok" if chunks and llm_configured else "degraded",
        "index_chunks": chunks,
        "drugs": len(known_drugs()),
        "llm_configured": llm_configured,
        "model": settings.groq_model,
        "embedding_model": settings.embedding_model,
        "ingesting": _ingest_lock.locked(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if _ingest_lock.locked():
        raise HTTPException(503, "The knowledge base is being rebuilt. Try again in a minute.")
    try:
        result = run_agent(
            req.message,
            history=[t.model_dump() for t in req.history],
            patient=req.patient.model_dump() if req.patient else None,
        )
    except AgentError as exc:   # Groq rate limit / outage
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:  # e.g. GROQ_API_KEY missing
        log.error("Chat failed: %s", exc)
        raise HTTPException(500, str(exc)) from exc
    return ChatResponse(**result.to_dict())


@app.post("/ingest")
def ingest(req: IngestRequest | None = None,
           x_admin_token: Optional[str] = Header(default=None)) -> dict:
    """Rebuild the vector index (blocking, ~1-2 minutes on CPU)."""
    if settings.admin_token and x_admin_token != settings.admin_token:
        raise HTTPException(401, "Missing or invalid X-Admin-Token header.")
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "An ingestion is already running.")
    try:
        if req and req.download:
            from data.download_openfda import main as download
            download()
        from backend.ingest import build_index
        stats = build_index(reset=True, verbose=False)
        log.info("Ingestion finished: %d chunks", stats["chunks"])
        return {"status": "ok", "stats": stats}
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        _ingest_lock.release()
