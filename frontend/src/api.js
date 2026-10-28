// All backend calls live here, so components never deal with URLs or HTTP details.
// Default "/api" goes through the Vite dev proxy (see vite.config.js).
const API_URL = import.meta.env.VITE_API_URL || "/api";

async function request(path, options) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    // FastAPI errors: {detail: "text"} or {detail: [{loc, msg}]} for validation (422).
    const detail = Array.isArray(body.detail)
      ? body.detail.map((e) => `${e.loc.slice(1).join(".")}: ${e.msg}`).join("; ")
      : body.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return body;
}

export const getHealth = () => request("/health");

export const sendChat = ({ message, history, patient }) =>
  request("/chat", { method: "POST", body: JSON.stringify({ message, history, patient }) });
