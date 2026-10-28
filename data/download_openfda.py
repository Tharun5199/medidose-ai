"""Phase 1 — Download drug labels from the public openFDA API.

Run from the project root:
    python -m data.download_openfda            # all 30 drugs in data/drug_list.json
    python -m data.download_openfda ibuprofen  # just one (handy for debugging)

What it produces, per drug, in data/raw/:
    <drug>.json  - the raw openFDA label record + our provenance metadata
    <drug>.md    - a normalised, human-readable version with one "## <section>"
                   heading per clinical section. Phase 2 chunks THIS file.

Design notes (the WHY):
- Source: openFDA drug labels are the FDA-approved "package inserts". They are
  public domain, authoritative, and already split into named fields
  (dosage_and_administration, adverse_reactions, ...). That structure is what
  lets us attach a reliable `section` metadata tag to every chunk later.
- One label per drug: openFDA returns dozens of labels per drug (every
  manufacturer, every combo product). We score candidates and keep the single
  best one (single-ingredient, prescription, most complete, newest) so the
  knowledge base has no near-duplicate chunks that would crowd the top-k results.
- Canonical sections: Rx labels and OTC "Drug Facts" labels use different field
  names (e.g. `adverse_reactions` vs nothing, `dosage_and_administration` vs
  `directions`). We map both onto a small fixed vocabulary
  (dosage, side_effects, interactions, ...). The agent's retrieval tool can then
  filter by section without knowing which label format a drug came from.
- Provenance: we store the DailyMed URL (via the label's set_id) so every
  answer can cite a source a student can open and verify.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
from datetime import datetime, timezone

import requests

from backend.config import settings

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"

# openFDA field name -> our canonical section name.
# Order matters: it is the order sections appear in the .md file.
SECTION_MAP: dict[str, list[str]] = {
    "boxed_warning": ["boxed_warning"],
    "indications": ["indications_and_usage", "purpose", "uses"],
    "dosage": ["dosage_and_administration", "dosage_forms_and_strengths", "directions"],
    "contraindications": ["contraindications", "do_not_use"],
    "warnings": ["warnings_and_cautions", "warnings", "precautions",
                 "ask_doctor", "when_using", "stop_use"],
    "side_effects": ["adverse_reactions"],
    "interactions": ["drug_interactions", "ask_doctor_or_pharmacist"],
    # Renal/hepatic/paediatric/geriatric guidance lives here — the key section
    # for the personalised-dosage feature.
    "special_populations": ["use_in_specific_populations", "pregnancy", "nursing_mothers",
                            "pediatric_use", "geriatric_use", "pregnancy_or_breast_feeding"],
    "overdose": ["overdosage"],
    "pharmacology": ["mechanism_of_action", "clinical_pharmacology", "pharmacokinetics"],
}

# Sections that make a label useful for students; used to score candidates.
KEY_FIELDS = ["indications_and_usage", "dosage_and_administration", "contraindications",
              "adverse_reactions", "drug_interactions", "use_in_specific_populations"]

REQUEST_DELAY_S = 0.3  # stay well under openFDA's 240 requests/minute limit


def query_labels(drug: str, product_type: str, limit: int = 25) -> list[dict]:
    """Return up to `limit` openFDA label records whose generic name matches `drug`."""
    search = f'openfda.generic_name:"{drug}" AND openfda.product_type:"{product_type}"'
    params = {"search": search, "limit": str(limit)}
    if settings.openfda_api_key:
        params["api_key"] = settings.openfda_api_key
    # openFDA expects '+' / spaces in the search expression, so build the query
    # string ourselves rather than letting requests escape it differently.
    url = OPENFDA_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:  # openFDA's way of saying "no matches"
        return []
    resp.raise_for_status()
    return resp.json().get("results", [])


def score_label(label: dict, drug: str) -> tuple:
    """Higher tuple = better candidate. Python compares tuples element by element."""
    generic_names = [g.lower() for g in label.get("openfda", {}).get("generic_name", [])]
    # Single-ingredient: e.g. "metformin hydrochloride" yes, "glipizide and metformin" no,
    # "olmesartan medoxomil-hydrochlorothiazide" no (combos use 'and', ',', '-' or '/').
    single = any(drug in g and not any(sep in g for sep in (" and ", ",", "-", "/"))
                 for g in generic_names)
    completeness = sum(1 for f in KEY_FIELDS if label.get(f))
    total_text = sum(len(" ".join(label.get(f, []))) for fields in SECTION_MAP.values() for f in fields)
    effective = label.get("effective_time", "0")
    return (single, completeness, effective, total_text)


def pick_best_label(drug: str) -> dict | None:
    """Prefer a prescription label; fall back to OTC (e.g. aspirin is mostly OTC)."""
    for product_type in ("HUMAN PRESCRIPTION DRUG", "HUMAN OTC DRUG"):
        candidates = query_labels(drug, product_type)
        time.sleep(REQUEST_DELAY_S)
        if not candidates:
            continue
        best = max(candidates, key=lambda lbl: score_label(lbl, drug))
        if score_label(best, drug)[0]:  # only accept a single-ingredient label
            return best
    return None


def clean(text_parts: list[str]) -> str:
    """openFDA fields are lists of strings; join them and normalise whitespace."""
    joined = "\n\n".join(part.strip() for part in text_parts if part and part.strip())
    return "\n".join(line.strip() for line in joined.splitlines())


def normalise(label: dict, drug: dict) -> dict:
    """Turn a raw openFDA record into our canonical {section: text} document."""
    sections: dict[str, str] = {}
    for canonical, fields in SECTION_MAP.items():
        texts: list[str] = []
        for f in fields:
            if not label.get(f):
                continue
            text = clean(label[f])
            # Labels nest fields: `pharmacokinetics` and `mechanism_of_action` are
            # usually repeated verbatim inside `clinical_pharmacology`. Keeping both
            # would index the same passage twice and waste top-k slots on duplicates.
            if any(text[:300] in existing for existing in texts):
                continue
            texts.append(text)
        if texts:
            sections[canonical] = "\n\n".join(texts)

    openfda = label.get("openfda", {})
    set_id = label.get("set_id", "")
    return {
        "drug_name": drug["name"],
        "aliases": drug.get("aliases", []),
        "brand_names": openfda.get("brand_name", [])[:5],
        "generic_name": (openfda.get("generic_name") or [drug["name"]])[0],
        "product_type": (openfda.get("product_type") or [""])[0],
        "manufacturer": (openfda.get("manufacturer_name") or [""])[0],
        "set_id": set_id,
        "effective_time": label.get("effective_time", ""),
        "source_url": DAILYMED_URL.format(set_id=set_id) if set_id else OPENFDA_URL,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sections": sections,
    }


def to_markdown(doc: dict) -> str:
    """Markdown with a metadata header and one '## section' per canonical section.

    The '## ' headings are deliberate: Phase 2's chunker splits on them first,
    so a chunk never mixes e.g. dosage text with side-effect text.
    """
    header = [
        f"# {doc['drug_name'].title()}",
        f"drug_name: {doc['drug_name']}",
        f"aliases: {', '.join(doc['aliases'])}",
        f"generic_name: {doc['generic_name']}",
        f"product_type: {doc['product_type']}",
        f"source_url: {doc['source_url']}",
        "",
    ]
    body = [f"## {name}\n\n{text}\n" for name, text in doc["sections"].items()]
    return "\n".join(header) + "\n" + "\n".join(body)


def main(only: list[str] | None = None) -> None:
    drugs = json.loads(settings.drug_list_path.read_text(encoding="utf-8"))
    if only:
        drugs = [d for d in drugs if d["name"] in only]
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    print(f"Downloading {len(drugs)} drug labels from openFDA...\n")
    for drug in drugs:
        name = drug["name"]
        try:
            label = pick_best_label(name)
        except requests.RequestException as exc:
            print(f"  [ERROR] {name}: {exc}")
            failed.append(name)
            continue
        if label is None:
            print(f"  [MISS ] {name}: no single-ingredient label found")
            failed.append(name)
            continue

        doc = normalise(label, drug)
        (settings.raw_dir / f"{name}.json").write_text(
            json.dumps({"metadata": {k: v for k, v in doc.items() if k != "sections"},
                        "raw_label": label}, indent=2), encoding="utf-8")
        (settings.raw_dir / f"{name}.md").write_text(to_markdown(doc), encoding="utf-8")
        ok += 1
        chars = sum(len(t) for t in doc["sections"].values())
        print(f"  [OK   ] {name:<20} {doc['product_type'][6:]:<18} "
              f"{len(doc['sections']):>2} sections  {chars:>7,} chars")

    print(f"\nDone: {ok} saved to {settings.raw_dir}, {len(failed)} failed {failed or ''}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
