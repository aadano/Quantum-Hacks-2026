from dataclasses import dataclass

PATIENTS_FILE = "patients.txt"
DEFAULT_OUTPUT_FILE = "unhealthy_patients.txt"


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


def load_patients(path=PATIENTS_FILE):
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
            name = name.strip()
            dna = dna.strip().upper()
            if len(dna) % 3 != 0:
                raise ValueError(f"{name} has DNA length {len(dna)}, which is not divisible by 3.")
            if set(dna) - set("ATCG"):
                raise ValueError(f"{name} has invalid DNA bases.")
            patients.append(Patient(name=name, dna=dna))
    if not patients:
        raise ValueError(f"No patients found in {path}.")
    return patients


def patient_has_codon(patient, target_codon, codon_index=None):
    if codon_index is not None:
        start = codon_index * 3
        return patient.dna[start : start + 3] == target_codon
    return any(patient.dna[i : i + 3] == target_codon for i in range(0, len(patient.dna), 3))


def find_matching_patient_indices(patients, target_codon, codon_index=None):
    return [
        index
        for index, patient in enumerate(patients)
        if patient_has_codon(patient, target_codon, codon_index)
    ]


def save_patient_names(patient_names, path=DEFAULT_OUTPUT_FILE):
    with open(path, "w", encoding="utf-8") as file:
        for name in patient_names:
            file.write(f"{name}\n")
