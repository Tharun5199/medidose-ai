// Patient parameters for personalised dosage guidance.
// WHY an explicit on/off switch: parameters silently left over from an earlier question
// would personalise answers the student didn't intend. When "off", nothing is sent.
const AGE_GROUPS = ["neonate", "infant", "child", "adolescent", "adult", "elderly"];

export const EMPTY_PATIENT = {
  enabled: false,
  age_group: "",
  weight_kg: "",
  renal_impairment: false,
  hepatic_impairment: false,
};

// Convert form state into the API's Patient schema (or null when disabled/empty).
export function toApiPatient(p) {
  if (!p.enabled) return null;
  const out = {
    age_group: p.age_group || null,
    weight_kg: p.weight_kg === "" ? null : Number(p.weight_kg),
    renal_impairment: p.renal_impairment,
    hepatic_impairment: p.hepatic_impairment,
  };
  const anySet = out.age_group || out.weight_kg || out.renal_impairment || out.hepatic_impairment;
  return anySet ? out : null;
}

export function patientSummary(p) {
  if (!p) return "";
  return [
    p.age_group,
    p.weight_kg && `${p.weight_kg} kg`,
    p.renal_impairment && "renal impairment",
    p.hepatic_impairment && "hepatic impairment",
  ]
    .filter(Boolean)
    .join(" · ");
}

export default function PatientPanel({ patient, onChange }) {
  const set = (field) => (e) =>
    onChange({ ...patient, [field]: e.target.type === "checkbox" ? e.target.checked : e.target.value });
  const weightInvalid = patient.weight_kg !== "" && !(Number(patient.weight_kg) > 0 && Number(patient.weight_kg) <= 350);

  return (
    <aside className="panel" aria-labelledby="patient-title">
      <div className="panel-head">
        <h2 id="patient-title">Patient parameters</h2>
        <label className="switch">
          <input type="checkbox" checked={patient.enabled} onChange={set("enabled")} />
          <span>{patient.enabled ? "On" : "Off"}</span>
        </label>
      </div>
      <p className="hint">
        When on, these are sent with each question so the agent searches for dosing guidance for this case.
      </p>

      <fieldset disabled={!patient.enabled}>
        <label>
          Age group
          <select value={patient.age_group} onChange={set("age_group")}>
            <option value="">Not specified</option>
            {AGE_GROUPS.map((g) => (
              <option key={g} value={g}>{g[0].toUpperCase() + g.slice(1)}</option>
            ))}
          </select>
        </label>

        <label>
          Weight (kg)
          <input type="number" min="0.5" max="350" step="0.1" placeholder="e.g. 20"
                 value={patient.weight_kg} onChange={set("weight_kg")} aria-invalid={weightInvalid} />
          {weightInvalid && <span className="field-error">Enter a weight between 0 and 350 kg</span>}
        </label>

        <label className="check">
          <input type="checkbox" checked={patient.renal_impairment} onChange={set("renal_impairment")} />
          Renal (kidney) impairment
        </label>
        <label className="check">
          <input type="checkbox" checked={patient.hepatic_impairment} onChange={set("hepatic_impairment")} />
          Hepatic (liver) impairment
        </label>

        <button type="button" className="link" onClick={() => onChange({ ...EMPTY_PATIENT, enabled: true })}>
          Clear
        </button>
      </fieldset>
    </aside>
  );
}
