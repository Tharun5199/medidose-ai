// The "Sources" section under each answer.
// WHY show the passage text, not just a link: a student can verify each [n] claim
// against the exact words the model was given, which is the point of a grounded system.
// Cited sources come first; retrieved-but-uncited ones are tucked away to reduce noise.
function SourceItem({ s, msgId, highlighted }) {
  return (
    <li id={`src-${msgId}-${s.id}`} className={`source ${highlighted ? "highlight" : ""}`}>
      <details>
        <summary>
          <span className="src-id">[{s.id}]</span>
          <span className="src-drug">{s.drug_name}</span>
          <span className="src-section">{s.section.replace("_", " ")}</span>
          <span className="src-score" title="Cosine relevance of this passage to the search">
            relevance {s.score.toFixed(2)}
          </span>
        </summary>
        <blockquote>{s.text.replace(/^\[[^\]]*\]\n/, "")}</blockquote>
        <a href={s.source_url} target="_blank" rel="noreferrer">
          Open FDA label on DailyMed ↗
        </a>
      </details>
    </li>
  );
}

export default function Sources({ sources, msgId, highlightId }) {
  if (!sources?.length) return null;
  const cited = sources.filter((s) => s.cited);
  const other = sources.filter((s) => !s.cited);

  return (
    <section className="sources" aria-label="Sources">
      <h3>Sources</h3>
      {cited.length > 0 ? (
        <ul>{cited.map((s) => <SourceItem key={s.id} s={s} msgId={msgId} highlighted={s.id === highlightId} />)}</ul>
      ) : (
        <p className="hint">The answer did not cite any specific passage.</p>
      )}
      {other.length > 0 && (
        <details className="more">
          <summary>Also retrieved but not cited ({other.length})</summary>
          <ul>{other.map((s) => <SourceItem key={s.id} s={s} msgId={msgId} highlighted={s.id === highlightId} />)}</ul>
        </details>
      )}
    </section>
  );
}
