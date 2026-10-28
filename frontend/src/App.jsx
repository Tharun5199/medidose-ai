import { useEffect, useRef, useState } from "react";
import { getHealth, sendChat } from "./api.js";
import Message from "./components/Message.jsx";
import PatientPanel, { EMPTY_PATIENT, toApiPatient } from "./components/PatientPanel.jsx";

const EXAMPLES = [
  { q: "What are the common side effects of atorvastatin?" },
  { q: "Can ibuprofen be taken with lisinopril?" },
  { q: "What is the paracetamol dose for this child?",
    patient: { enabled: true, age_group: "child", weight_kg: "20", renal_impairment: false, hepatic_impairment: false } },
  { q: "What is the dose of gabapentin?",
    patient: { enabled: true, age_group: "adult", weight_kg: "", renal_impairment: true, hepatic_impairment: false } },
];

const HISTORY_TURNS = 6; // the backend only uses the last 6 anyway; don't send more

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [patient, setPatient] = useState(EMPTY_PATIENT);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const endRef = useRef(null);
  const nextId = useRef(1);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "down" }));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function ask(text, patientOverride) {
    const message = text.trim();
    if (!message || loading) return;
    const p = patientOverride ?? patient;
    if (patientOverride) setPatient(patientOverride);
    const apiPatient = toApiPatient(p);

    // History = text of earlier turns only (errors excluded). The API is stateless.
    const history = messages
      .filter((m) => !m.error)
      .slice(-HISTORY_TURNS)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, { id: nextId.current++, role: "user", content: message, patient: apiPatient }]);
    setInput("");
    setLoading(true);
    try {
      const res = await sendChat({ message, history, patient: apiPatient });
      setMessages((prev) => [...prev, { id: nextId.current++, role: "assistant", content: res.answer, ...res }]);
    } catch (err) {
      setMessages((prev) => [...prev, { id: nextId.current++, role: "assistant", error: err.message }]);
    } finally {
      setLoading(false);
    }
  }

  const onKeyDown = (e) => {
    // Enter sends; Shift+Enter makes a new line.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(input);
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>℞</span>
          <div>
            <h1>MediDose AI</h1>
            <p>Answers grounded in FDA drug labels, with sources</p>
          </div>
        </div>
        <div className={`status ${health?.status === "ok" ? "ok" : "bad"}`} title={health?.model || ""}>
          {health == null
            ? "Connecting…"
            : health.status === "ok"
              ? `${health.drugs} drugs · ${health.index_chunks.toLocaleString()} passages`
              : "Backend unavailable"}
        </div>
      </header>

      <div className="banner" role="note">
        <strong>Educational use only.</strong> Not medical advice. Answers come only from the indexed FDA labels;
        always verify against the official prescribing information and consult a qualified clinician.
      </div>

      <div className="layout">
        <PatientPanel patient={patient} onChange={setPatient} />

        <main className="chat">
          <div className="messages" aria-live="polite">
            {messages.length === 0 && (
              <div className="empty">
                <h2>Ask about a drug</h2>
                <p>Indications, dosage, side effects, interactions, contraindications, or use in special populations.</p>
                <div className="examples">
                  {EXAMPLES.map((ex) => (
                    <button key={ex.q} onClick={() => ask(ex.q, ex.patient)}>
                      {ex.q}
                      {ex.patient && <span className="ex-tag">with patient</span>}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m) => <Message key={m.id} msg={m} />)}
            {loading && (
              <div className="msg assistant thinking">
                <span className="dots" aria-hidden><i /><i /><i /></span> Deciding whether to search the drug labels…
              </div>
            )}
            <div ref={endRef} />
          </div>

          <form className="composer" onSubmit={(e) => { e.preventDefault(); ask(input); }}>
            <textarea
              rows={2}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="e.g. What is the maximum daily dose of acetaminophen?"
              maxLength={2000}
              aria-label="Your question"
            />
            <button type="submit" disabled={loading || !input.trim()}>Ask</button>
          </form>
        </main>
      </div>
    </div>
  );
}
