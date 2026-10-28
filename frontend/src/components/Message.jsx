import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Sources from "./Sources.jsx";
import { patientSummary } from "./PatientPanel.jsx";

// Turn "[2]" citation markers into links to that message's source entry.
// WHY: clicking a citation jumps to and opens the exact passage behind the claim.
const linkCitations = (text, msgId) =>
  text.replace(/\[(\d+)\](?!\()/g, (_, n) => `[[${n}]](#src-${msgId}-${n})`);

// One line explaining what the agent decided: the visible "agentic" decision.
function AgentDecision({ retrieved, toolCalls }) {
  if (!retrieved) {
    return <div className="decision none">Answered without retrieval (no drug facts needed)</div>;
  }
  return (
    <div className="decision">
      Searched the drug labels {toolCalls.length}×
      {toolCalls.length > 1 && " (refined the query)"}:{" "}
      {toolCalls.map((t, i) => (
        <code key={i}>
          {t.drug_name || "all drugs"}/{t.section || "any section"}
        </code>
      ))}
    </div>
  );
}

export default function Message({ msg }) {
  const [highlight, setHighlight] = useState(null);

  if (msg.role === "user") {
    return (
      <div className="msg user">
        <p>{msg.content}</p>
        {msg.patient && <div className="patient-tag">Patient: {patientSummary(msg.patient)}</div>}
      </div>
    );
  }

  if (msg.error) {
    return <div className="msg error" role="alert">⚠ {msg.error}</div>;
  }

  const onCitationClick = (e, href) => {
    e.preventDefault();
    const el = document.getElementById(href.slice(1));
    if (!el) return;
    setHighlight(Number(href.split("-").pop()));
    // Uncited sources live inside a collapsed <details>; open the parents, then the item.
    let parent = el.parentElement;
    while (parent) {
      if (parent.tagName === "DETAILS") parent.open = true;
      parent = parent.parentElement;
    }
    el.querySelector("details").open = true;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <div className="msg assistant">
      <AgentDecision retrieved={msg.retrieved} toolCalls={msg.tool_calls || []} />
      <div className="answer">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ href, children }) =>
              href?.startsWith("#src-") ? (
                <a href={href} className="cite" onClick={(e) => onCitationClick(e, href)}>{children}</a>
              ) : (
                <a href={href} target="_blank" rel="noreferrer">{children}</a>
              ),
          }}
        >
          {linkCitations(msg.content, msg.id)}
        </ReactMarkdown>
      </div>

      {msg.unverified_numbers?.length > 0 && (
        <div className="warn" role="note">
          ⚠ Some figures could not be matched to the retrieved sources:{" "}
          <strong>{msg.unverified_numbers.join(", ")}</strong>. Verify them in the label before relying on them.
        </div>
      )}

      <Sources sources={msg.sources} msgId={msg.id} highlightId={highlight} />
      <p className="disclaimer-line">{msg.disclaimer}</p>
    </div>
  );
}
