import html
import importlib
import io
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cgi


HOST = "127.0.0.1"
PORT = 8501
PATIENTS_FILE = "patients.txt"
DEFAULT_SHOTS = 2048
OUTPUT_FILE = "doctor_unhealthy_patients.txt"
CURRENT_UPLOAD = {
    "filename": None,
    "file_bytes": None,
}


@dataclass(frozen=True)
class Patient:
    name: str
    dna: str


def normalize_codon(codon):
    normalized = codon.strip().upper()
    if len(normalized) != 3:
        raise ValueError("Target codon must be exactly 3 DNA bases.")
    invalid_bases = sorted(set(normalized) - set("ATCG"))
    if invalid_bases:
        raise ValueError(f"Target codon has invalid DNA bases: {', '.join(invalid_bases)}")
    return normalized


def load_patients(path):
    patients = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            if "→" in line:
                name, dna = line.split("→", 1)
            elif "->" in line:
                name, dna = line.split("->", 1)
            else:
                raise ValueError(f"Line {line_number} is missing a patient/DNA separator.")
            dna = dna.strip().upper()
            if len(dna) % 3 != 0:
                raise ValueError(f"{name.strip()} has DNA length {len(dna)}, which is not divisible by 3.")
            if set(dna) - set("ATCG"):
                raise ValueError(f"{name.strip()} has invalid DNA bases.")
            patients.append(Patient(name=name.strip(), dna=dna))
    if not patients:
        raise ValueError(f"No patients found in {path}.")
    return patients


def patient_has_codon(patient, target_codon, codon_index=None):
    if codon_index is not None:
        start = codon_index * 3
        return patient.dna[start : start + 3] == target_codon
    return any(patient.dna[index : index + 3] == target_codon for index in range(0, len(patient.dna), 3))


def find_matching_patient_indices(patients, target_codon, codon_index=None):
    return [
        index
        for index, patient in enumerate(patients)
        if patient_has_codon(patient, target_codon, codon_index)
    ]


def save_patient_names(patient_names, path):
    with open(path, "w", encoding="utf-8") as file:
        for name in patient_names:
            file.write(f"{name}\n")


def run_grover_from_backend(patients, target_codon, codon_index, shots):
    with redirect_stdout(io.StringIO()):
        backend = importlib.import_module("grover_patient_search")
    return backend.run_grover_search(
        patients,
        target_codon,
        codon_index=codon_index,
        shots=shots,
    )


def analyze_patients(file_bytes, filename, target_codon, codon_index, run_quantum, shots):
    target_codon = normalize_codon(target_codon)

    if file_bytes is None:
        raise ValueError("No file uploaded")

    with tempfile.NamedTemporaryFile("wb", delete=True) as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()
        patients = load_patients(temp_file.name)

    start = time.perf_counter()
    marked_indices = find_matching_patient_indices(patients, target_codon, codon_index)
    classical_time = time.perf_counter() - start
    classical_steps = len(patients)
    unhealthy_patients = [patients[index].name for index in marked_indices]
    save_patient_names(unhealthy_patients, OUTPUT_FILE)

    quantum_result = None
    quantum_error = None
    if run_quantum:
        try:
            quantum_result = run_grover_from_backend(patients, target_codon, codon_index, shots)
        except Exception as error:
            quantum_error = str(error)

    return {
        "filename": filename,
        "patients": patients,
        "target_codon": target_codon,
        "codon_index": codon_index,
        "unhealthy_patients": unhealthy_patients,
        "marked_indices": marked_indices,
        "classical_time": classical_time,
        "classical_steps": classical_steps,
        "quantum_result": quantum_result,
        "quantum_error": quantum_error,
    }


def default_form_values():
    return {
        "target_codon": "AAC",
        "codon_index": "11",
        "shots": str(DEFAULT_SHOTS),
        "run_quantum": True,
    }


def render_page(result=None, error=None, form_values=None, allow_current_file=False):
    form_values = form_values or default_form_values()
    target_value = html.escape(form_values.get("target_codon", "AAC"))
    codon_value = html.escape(form_values.get("codon_index", "11"))
    shots_value = html.escape(form_values.get("shots", str(DEFAULT_SHOTS)))
    quantum_checked = " checked" if form_values.get("run_quantum", True) else ""
    result_html = """
        <section class="empty-state">
          <div class="dna-visual" aria-hidden="true">
            <span></span><span></span><span></span><span></span><span></span>
          </div>
          <div>
            <h2>Ready for case review</h2>
            <p>No mutation scan has been run in this session.</p>
          </div>
        </section>
        """
    current_file_html = ""
    if allow_current_file and CURRENT_UPLOAD["filename"]:
        current_file_html = f"""
        <div class="current-file">
          <span>Current File</span>
          <strong>{html.escape(CURRENT_UPLOAD["filename"])}</strong>
        </div>
        <input type="hidden" name="has_current_file" value="1">
        """

    if error:
        result_html = f"""
        <section class="alert">
          <strong>Scan failed</strong>
          <p>{html.escape(error)}</p>
        </section>
        """
    elif result:
        patient_count = len(result["patients"])
        match_count = len(result["unhealthy_patients"])
        codon_index = result["codon_index"] if result["codon_index"] is not None else "Any"
        grover_iterations = 0
        if result["quantum_result"]:
            grover_iterations = result["quantum_result"]["iterations"]
        max_steps = max(result["classical_steps"], grover_iterations, 1)
        classical_bar_width = (result["classical_steps"] / max_steps) * 100
        grover_bar_width = (grover_iterations / max_steps) * 100
        patient_rows = "\n".join(
            f"""
            <tr>
              <td><span class="patient-badge">{html.escape(name)}</span></td>
              <td><span class="index-pill">{index}</span></td>
            </tr>
            """
            for name, index in zip(result["unhealthy_patients"], result["marked_indices"])
        )
        if not patient_rows:
            patient_rows = '<tr><td colspan="2" class="empty">No matching patients found.</td></tr>'

        quantum = result["quantum_result"]
        quantum_error = result["quantum_error"]
        if quantum:
            quantum_html = f"""
            <div class="metric">
              <span>Grover Iterations</span>
              <strong>{quantum["iterations"]}</strong>
            </div>
            """
        elif quantum_error:
            quantum_html = f"""
            <section class="notice">
              <strong>Quantum simulation unavailable</strong>
              <p>{html.escape(quantum_error)}</p>
            </section>
            """
        else:
            quantum_html = """
            <section class="notice">
              <strong>Quantum simulation skipped</strong>
              <p>Classical mutation screening completed.</p>
            </section>
            """

        result_html = f"""
        <section class="summary">
          <div class="metric metric-blue">
            <span>Patients</span>
            <strong>{patient_count}</strong>
          </div>
          <div class="metric metric-red">
            <span>Mutation Matches</span>
            <strong>{match_count}</strong>
          </div>
          <div class="metric metric-slate">
            <span>Classical Steps</span>
            <strong>{result["classical_steps"]}</strong>
          </div>
          {quantum_html}
        </section>

        <section class="step-chart panel">
          <div class="section-heading compact">
            <h2>Search Steps</h2>
            <span>Classical vs Grover</span>
          </div>
          <div class="bar-row">
            <div class="bar-label">Classical</div>
            <div class="bar-track">
              <div class="bar classical" style="width: {classical_bar_width:.2f}%"></div>
            </div>
            <div class="bar-value">{result["classical_steps"]}</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Grover</div>
            <div class="bar-track">
              <div class="bar quantum" style="width: {grover_bar_width:.2f}%"></div>
            </div>
            <div class="bar-value">{grover_iterations}</div>
          </div>
        </section>

        <section class="scan-details">
          <div>
            <span class="label">File</span>
            <strong>{html.escape(result["filename"])}</strong>
          </div>
          <div>
            <span class="label">Target Codon</span>
            <strong>{html.escape(result["target_codon"])}</strong>
          </div>
          <div>
            <span class="label">Codon Index</span>
            <strong>{codon_index}</strong>
          </div>
        </section>

        <section class="results panel">
          <div class="section-heading">
            <h2>Patient IDs</h2>
            <span>{html.escape(OUTPUT_FILE)}</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Patient</th>
                <th>Index</th>
              </tr>
            </thead>
            <tbody>{patient_rows}</tbody>
          </table>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mutation Search Console</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #647083;
      --line: #d9e3eb;
      --panel: #ffffff;
      --band: #eef6f7;
      --accent: #0f8b7e;
      --accent-dark: #0f6d65;
      --accent-soft: #e6f6f3;
      --blue: #2563eb;
      --red: #dc2626;
      --green: #16a34a;
      --warn: #8a4b00;
      --danger: #9f1239;
      --shadow: 0 22px 70px rgba(23, 32, 51, 0.09);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 139, 126, 0.13), transparent 34rem),
        linear-gradient(180deg, #f5fafb 0%, #eef4f7 100%);
      letter-spacing: 0;
      min-height: 100vh;
    }}
    header {{
      padding: 30px clamp(18px, 4vw, 56px) 18px;
    }}
    .topline {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      max-width: 1240px;
      margin: 0 auto;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
      color: inherit;
      text-decoration: none;
    }}
    .brand:hover h1 {{ color: var(--accent-dark); }}
    .brand-mark {{
      width: 52px;
      height: 52px;
      border-radius: 16px;
      background: linear-gradient(145deg, #0f8b7e, #1d4ed8);
      box-shadow: 0 14px 30px rgba(15, 139, 126, 0.25);
      position: relative;
      flex: 0 0 auto;
      overflow: hidden;
    }}
    .brand-mark::before, .brand-mark::after {{
      content: "";
      position: absolute;
      width: 9px;
      height: 40px;
      top: 6px;
      border-radius: 999px;
      border: 2px solid rgba(255, 255, 255, 0.9);
    }}
    .brand-mark::before {{
      left: 14px;
      transform: rotate(23deg);
    }}
    .brand-mark::after {{
      right: 14px;
      transform: rotate(-23deg);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.85rem, 4vw, 3.25rem);
      line-height: 1;
      font-weight: 760;
    }}
    .eyebrow {{
      margin: 0 0 7px;
      color: var(--accent-dark);
      font-size: 0.78rem;
      font-weight: 850;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .status {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid #b9dcd6;
      background: rgba(255, 255, 255, 0.8);
      color: var(--accent-dark);
      border-radius: 999px;
      padding: 10px 14px;
      font-weight: 700;
      white-space: nowrap;
      box-shadow: 0 10px 24px rgba(23, 32, 51, 0.07);
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 18px clamp(18px, 4vw, 56px) 52px;
    }}
    .workspace {{
      display: grid;
      grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
      gap: 24px;
      align-items: start;
    }}
    form, .panel, .alert, .notice, .empty-state {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    form {{
      padding: 0;
      overflow: hidden;
      position: sticky;
      top: 18px;
    }}
    .form-header {{
      padding: 20px 20px 16px;
      background: linear-gradient(135deg, #ffffff 0%, #eef8f6 100%);
      border-bottom: 1px solid var(--line);
    }}
    .form-header h2 {{
      margin: 0;
      font-size: 1.18rem;
    }}
    .form-header p {{
      margin: 6px 0 0;
      color: var(--muted);
      line-height: 1.4;
    }}
    fieldset {{
      border: 0;
      padding: 20px;
      margin: 0;
      display: grid;
      gap: 15px;
    }}
    label {{
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 0.9rem;
      font-weight: 700;
    }}
    input[type="text"], input[type="number"], input[type="file"] {{
      width: 100%;
      min-height: 46px;
      border: 1px solid #c8d3dd;
      border-radius: 6px;
      background: #fbfdfe;
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
    }}
    input:focus {{
      outline: 3px solid rgba(15, 139, 126, 0.16);
      border-color: var(--accent);
    }}
    input[type="file"] {{
      border-style: dashed;
      background: #f6fbfb;
      cursor: pointer;
    }}
    .check-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 36px;
      color: var(--ink);
      font-weight: 700;
    }}
    .check-row input {{ width: 18px; height: 18px; accent-color: var(--accent); }}
    .current-file {{
      border: 1px solid #b9ded8;
      border-radius: 6px;
      background: var(--accent-soft);
      color: var(--muted);
      padding: 11px 12px;
      font-size: 0.9rem;
      overflow-wrap: anywhere;
      display: grid;
      gap: 3px;
    }}
    .current-file span {{
      color: var(--accent-dark);
      font-size: 0.72rem;
      font-weight: 850;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .current-file strong {{ color: var(--ink); }}
    .upload-field {{
      display: grid;
      gap: 8px;
    }}
    .upload-zone {{
      min-height: 132px;
      border: 1.5px dashed #9fc9c4;
      border-radius: 8px;
      background:
        linear-gradient(135deg, rgba(15, 139, 126, 0.08), rgba(37, 99, 235, 0.06)),
        #fbfefe;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 18px;
      cursor: pointer;
      transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
    }}
    .upload-zone:hover {{
      border-color: var(--accent);
      background:
        linear-gradient(135deg, rgba(15, 139, 126, 0.12), rgba(37, 99, 235, 0.08)),
        #ffffff;
      transform: translateY(-1px);
    }}
    .upload-zone.has-file {{
      border-style: solid;
      border-color: #8fd1c7;
      background:
        linear-gradient(135deg, rgba(15, 139, 126, 0.14), rgba(34, 197, 94, 0.08)),
        #ffffff;
    }}
    .upload-icon {{
      width: 42px;
      height: 42px;
      border-radius: 14px;
      display: inline-grid;
      place-items: center;
      margin: 0 auto 10px;
      color: white;
      background: linear-gradient(135deg, var(--accent), #2563eb);
      box-shadow: 0 12px 24px rgba(15, 139, 126, 0.23);
    }}
    .upload-title {{
      display: block;
      color: var(--ink);
      font-size: 0.98rem;
      font-weight: 850;
    }}
    .upload-copy {{
      display: block;
      color: var(--muted);
      margin-top: 4px;
      font-size: 0.84rem;
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    .upload-zone input[type="file"] {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}
    button {{
      min-height: 48px;
      border: 0;
      border-radius: 6px;
      background: linear-gradient(135deg, var(--accent), #2563eb);
      color: white;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 15px 26px rgba(15, 139, 126, 0.24);
    }}
    button:hover {{ filter: brightness(0.96); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .metric {{
      min-height: 112px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      display: grid;
      align-content: space-between;
      gap: 12px;
      box-shadow: 0 14px 32px rgba(23, 32, 51, 0.06);
      position: relative;
      overflow: hidden;
    }}
    .metric::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 5px;
      background: var(--accent);
    }}
    .metric span, .label {{
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric strong {{
      font-size: clamp(1.35rem, 3vw, 2rem);
      line-height: 1;
    }}
    .metric-blue::before {{ background: var(--blue); }}
    .metric-red::before {{ background: var(--red); }}
    .metric-slate::before {{ background: #475569; }}
    .metric-red {{
      border-color: #fecaca;
      background: linear-gradient(180deg, #ffffff 0%, #fff7f7 100%);
    }}
    .scan-details {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-bottom: 18px;
    }}
    .scan-details > div {{
      background: rgba(238, 246, 247, 0.86);
      border: 1px solid #cfe1e7;
      border-radius: 8px;
      padding: 14px;
      display: grid;
      gap: 5px;
      min-width: 0;
    }}
    .scan-details strong {{
      overflow-wrap: anywhere;
    }}
    .results {{ overflow: hidden; }}
    .section-heading {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }}
    .section-heading.compact {{
      padding: 14px 16px;
    }}
    h2 {{ margin: 0; font-size: 1.1rem; }}
    .section-heading span {{ color: var(--muted); font-size: 0.88rem; font-weight: 700; }}
    .step-chart {{
      padding-bottom: 16px;
      margin-bottom: 18px;
      overflow: hidden;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 96px minmax(120px, 1fr) 78px;
      gap: 14px;
      align-items: center;
      padding: 16px 18px 0;
    }}
    .bar-label {{
      color: var(--muted);
      font-weight: 800;
    }}
    .bar-track {{
      height: 34px;
      border: 1px solid #d7e1e8;
      border-radius: 6px;
      background: linear-gradient(180deg, #f4f7f9 0%, #eef3f6 100%);
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      min-width: 3px;
      border-radius: 0 6px 6px 0;
    }}
    .bar.classical {{
      background: linear-gradient(90deg, #ef4444, #dc2626);
    }}
    .bar.quantum {{
      background: linear-gradient(90deg, #22c55e, #16a34a);
    }}
    .bar-value {{
      font-weight: 850;
      text-align: right;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
    }}
    th, td {{
      text-align: left;
      padding: 13px 18px;
      border-bottom: 1px solid #edf1f4;
    }}
    th {{
      background: #f6f8fa;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    tbody tr:nth-child(even) {{ background: #fbfdfe; }}
    .patient-badge {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border-radius: 999px;
      padding: 4px 10px;
      background: #edf7ff;
      color: #1d4ed8;
      font-weight: 800;
    }}
    .index-pill {{
      display: inline-flex;
      min-width: 44px;
      justify-content: center;
      border-radius: 999px;
      padding: 4px 10px;
      background: #f1f5f9;
      color: #334155;
      font-weight: 800;
    }}
    .empty {{ color: var(--muted); text-align: center; }}
    .alert, .notice {{
      padding: 16px 18px;
      margin-bottom: 18px;
    }}
    .alert {{
      border-color: #fecdd3;
      background: #fff1f2;
      color: var(--danger);
    }}
    .notice {{
      border-color: #fed7aa;
      background: #fff7ed;
      color: var(--warn);
    }}
    .alert p, .notice p {{ margin: 6px 0 0; }}
    .empty-state {{
      min-height: 430px;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 36px;
      background:
        radial-gradient(circle at center, rgba(15, 139, 126, 0.08), transparent 18rem),
        #ffffff;
    }}
    .empty-state h2 {{
      margin: 18px 0 8px;
      font-size: 1.45rem;
    }}
    .empty-state p {{
      margin: 0;
      color: var(--muted);
    }}
    .dna-visual {{
      width: min(260px, 76vw);
      height: 150px;
      display: grid;
      align-items: center;
      gap: 11px;
    }}
    .dna-visual span {{
      display: block;
      height: 9px;
      border-radius: 999px;
      background: linear-gradient(90deg, #2563eb 0 16%, transparent 16% 27%, #0f8b7e 27% 73%, transparent 73% 84%, #dc2626 84% 100%);
      box-shadow: 0 11px 24px rgba(23, 32, 51, 0.08);
    }}
    .dna-visual span:nth-child(2), .dna-visual span:nth-child(4) {{
      transform: scaleX(0.82);
      opacity: 0.75;
    }}
    .dna-visual span:nth-child(3) {{
      transform: scaleX(0.66);
      opacity: 0.58;
    }}
    @media (max-width: 850px) {{
      .topline {{ align-items: flex-start; flex-direction: column; }}
      .workspace, .summary, .scan-details {{ grid-template-columns: 1fr; }}
      form {{ position: static; }}
      .bar-row {{ grid-template-columns: 76px minmax(80px, 1fr) 56px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topline">
      <a class="brand" href="/reset" aria-label="Return to home screen">
        <div class="brand-mark" aria-hidden="true"></div>
        <div>
          <p class="eyebrow">Quantum Genomics</p>
          <h1>Mutation Search Console</h1>
        </div>
      </a>
      <span class="status">Clinical Review</span>
    </div>
  </header>
  <main>
    <div class="workspace">
      <form method="post" enctype="multipart/form-data">
        <div class="form-header">
          <h2>Case Intake</h2>
          <p>Patient DNA screening</p>
        </div>
        <fieldset>
          <label class="upload-field">
            Patient DNA File
            <span class="upload-zone" id="uploadZone">
              <span>
                <span class="upload-icon" aria-hidden="true">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                    <path d="M12 16V5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
                    <path d="M7.5 9.5L12 5L16.5 9.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M5 16.5V18C5 18.5523 5.44772 19 6 19H18C18.5523 19 19 18.5523 19 18V16.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
                  </svg>
                </span>
                <span class="upload-title" id="uploadTitle">Choose patient DNA file</span>
                <span class="upload-copy" id="uploadCopy">Any file type accepted</span>
              </span>
              <input type="file" name="patients_file" id="patientsFile">
            </span>
          </label>
          {current_file_html}
          <label>
            Target Codon
            <input type="text" name="target_codon" value="{target_value}" maxlength="3" pattern="[AaTtCcGg]{{3}}" required>
          </label>
          <label>
            Codon Index
            <input type="number" name="codon_index" value="{codon_value}" min="0">
          </label>
          <label>
            Quantum Shots
            <input type="number" name="shots" value="{shots_value}" min="128" step="128">
          </label>
          <label class="check-row">
            <input type="checkbox" name="run_quantum"{quantum_checked}>
            Run Grover simulation
          </label>
          <button type="submit">Run Scan</button>
        </fieldset>
      </form>
      <div>
        {result_html}
      </div>
    </div>
  </main>
  <script>
    const patientsFile = document.getElementById("patientsFile");
    const uploadZone = document.getElementById("uploadZone");
    const uploadTitle = document.getElementById("uploadTitle");
    const uploadCopy = document.getElementById("uploadCopy");

    patientsFile?.addEventListener("change", () => {{
      const file = patientsFile.files && patientsFile.files[0];
      if (!file) {{
        uploadZone.classList.remove("has-file");
        uploadTitle.textContent = "Choose patient DNA file";
        uploadCopy.textContent = "Any file type accepted";
        return;
      }}
      uploadZone.classList.add("has-file");
      uploadTitle.textContent = "File selected";
      uploadCopy.textContent = file.name;
    }});
  </script>
</body>
</html>"""


class MutationSearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] == "/reset":
            CURRENT_UPLOAD["filename"] = None
            CURRENT_UPLOAD["file_bytes"] = None
        self.send_html(render_page())

    def do_POST(self):
        form_values = default_form_values()
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )

            uploaded = form["patients_file"] if "patients_file" in form else None
            can_reuse_current_file = form.getfirst("has_current_file") == "1"
            file_bytes = CURRENT_UPLOAD["file_bytes"] if can_reuse_current_file else None
            filename = CURRENT_UPLOAD["filename"] if can_reuse_current_file else None
            if uploaded is not None and uploaded.filename:
                filename = uploaded.filename
                file_bytes = uploaded.file.read()
                CURRENT_UPLOAD["filename"] = filename
                CURRENT_UPLOAD["file_bytes"] = file_bytes
                can_reuse_current_file = True

            target_codon = form.getfirst("target_codon", "AAC")
            codon_value = form.getfirst("codon_index", "").strip()
            codon_index = int(codon_value) if codon_value else None
            shots_value = form.getfirst("shots", str(DEFAULT_SHOTS))
            shots = int(shots_value)
            run_quantum = form.getfirst("run_quantum") == "on"
            form_values = {
                "target_codon": target_codon,
                "codon_index": codon_value,
                "shots": shots_value,
                "run_quantum": run_quantum,
            }

            started = time.perf_counter()
            result = analyze_patients(
                file_bytes=file_bytes,
                filename=filename,
                target_codon=target_codon,
                codon_index=codon_index,
                run_quantum=run_quantum,
                shots=shots,
            )
            result["request_time"] = time.perf_counter() - started
            self.send_html(
                render_page(
                    result=result,
                    form_values=form_values,
                    allow_current_file=can_reuse_current_file,
                )
            )
        except Exception as error:
            self.send_html(
                render_page(
                    error=str(error),
                    form_values=form_values,
                    allow_current_file=form.getfirst("has_current_file") == "1"
                    if "form" in locals()
                    else False,
                ),
                status=400,
            )

    def send_html(self, body, status=200):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), MutationSearchHandler)
    print(f"Mutation Search Console running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
