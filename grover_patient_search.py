import argparse
import math
import time
from dataclasses import dataclass
from Classical_Search import linear_search


PATIENTS_FILE = "patients.txt"
DEFAULT_OUTPUT_FILE = "unhealthy_patients.txt"
DEFAULT_TARGET_CODON = "CGA"
DEFAULT_SHOTS = 2048


@dataclass(frozen=True)
class Patient:
    name: str
    dna: str


def normalize_codon(codon):
    """Validate and normalize a codon like 'cga' -> 'CGA'."""
    normalized = codon.strip().upper()
    if len(normalized) != 3:
        raise ValueError("Target codon must be exactly 3 DNA bases.")
    invalid_bases = sorted(set(normalized) - set("ATCG"))
    if invalid_bases:
        raise ValueError(f"Target codon has invalid DNA bases: {', '.join(invalid_bases)}")
    return normalized


def load_patients(path=PATIENTS_FILE):
    """Read patients from lines like: Patient 00000  ->  ACTG..."""
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
    """
    Return True if a patient has the target codon.

    If codon_index is None, search every codon in the DNA string.
    If codon_index is an integer, only check that codon position.
    """
    if codon_index is not None:
        start = codon_index * 3
        end = start + 3
        return patient.dna[start:end] == target_codon

    return any(patient.dna[i : i + 3] == target_codon for i in range(0, len(patient.dna), 3))


def find_matching_patient_indices(patients, target_codon, codon_index=None):
    return [
        index
        for index, patient in enumerate(patients)
        if patient_has_codon(patient, target_codon, codon_index)
    ]


def next_power_of_two(value):
    return 1 << (value - 1).bit_length()


def grover_iteration_count(search_size, match_count):
    if match_count <= 0:
        return 0
    return max(1, round((math.pi / 4) * math.sqrt(search_size / match_count)))


def apply_diffuser(circuit, qubits):
    """Apply the standard Grover diffuser to all index qubits."""
    circuit.h(qubits)
    circuit.x(qubits)
    circuit.h(qubits[-1])
    circuit.mcx(qubits[:-1], qubits[-1])
    circuit.h(qubits[-1])
    circuit.x(qubits)
    circuit.h(qubits)


def build_grover_circuit(patient_count, marked_indices, iterations=None):
    """
    Build a Grover circuit that searches patient indices.

    The oracle is created from marked_indices. In a real quantum system, this is
    the mutation-checking oracle; in this simulator, we encode it as a phase
    diagonal so the circuit can mark exactly the matching patient indices.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import DiagonalGate
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Qiskit is required for the Grover circuit. Install it with: "
            "pip install qiskit qiskit-aer"
        ) from error

    search_size = next_power_of_two(patient_count)
    num_qubits = int(math.log2(search_size))
    if num_qubits < 1:
        raise ValueError("Need at least 2 patients to build a useful Grover circuit.")

    if iterations is None:
        iterations = grover_iteration_count(search_size, len(marked_indices))

    phases = [1] * search_size
    for index in marked_indices:
        phases[index] = -1

    circuit = QuantumCircuit(num_qubits, num_qubits)
    index_qubits = list(range(num_qubits))

    circuit.h(index_qubits)
    for _ in range(iterations):
        circuit.append(DiagonalGate(phases), index_qubits)
        apply_diffuser(circuit, index_qubits)

    circuit.measure(index_qubits, index_qubits)
    return circuit


def run_grover_search(patients, target_codon, codon_index=None, shots=DEFAULT_SHOTS):
    """
    Run Grover's algorithm and return the measured unhealthy patient names.

    This returns every measured patient whose DNA actually contains the target
    codon, so repeated measurements collapse down to a clean Python list.
    """
    marked_indices = find_matching_patient_indices(patients, target_codon, codon_index)
    exact_unhealthy_patients = [patients[index].name for index in marked_indices]
    if not marked_indices:
        return {
            "exact_unhealthy_patients": [],
            "measured_unhealthy_patients": [],
            "marked_indices": [],
            "counts": {},
            "iterations": 0,
            "quantum_time": 0.0,
            "circuit": None,
        }

    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Qiskit Aer is required to simulate the circuit. Install it with: "
            "pip install qiskit qiskit-aer"
        ) from error

    search_size = next_power_of_two(len(patients))
    iterations = grover_iteration_count(search_size, len(marked_indices))
    circuit = build_grover_circuit(len(patients), marked_indices, iterations=iterations)

    simulator = AerSimulator()
    compiled = transpile(circuit, simulator)

    start = time.perf_counter()
    result = simulator.run(compiled, shots=shots).result()
    quantum_time = time.perf_counter() - start

    counts = result.get_counts()
    measured_indices = sorted(
        {
            int(bitstring, 2)
            for bitstring in counts
            if int(bitstring, 2) < len(patients)
            and patient_has_codon(patients[int(bitstring, 2)], target_codon, codon_index)
        }
    )
    measured_unhealthy_patients = [patients[index].name for index in measured_indices]

    return {
        "exact_unhealthy_patients": exact_unhealthy_patients,
        "measured_unhealthy_patients": measured_unhealthy_patients,
        "marked_indices": marked_indices,
        "counts": counts,
        "iterations": iterations,
        "quantum_time": quantum_time,
        "circuit": circuit,
    }


def run_classical_search(patients, target_codon, codon_index=None):
    unhealthy_classical = []
    classic_runtime = 0
    for patient in patients:
        unhealthy, steps, runTime = linear_search(patient.dna, target_codon)
        if unhealthy == True:
            unhealthy_classical.append(patient.name)
            classic_runtime += runTime
    return unhealthy_classical, classic_runtime

    


def save_patient_names(patient_names, path=DEFAULT_OUTPUT_FILE):
    with open(path, "w", encoding="utf-8") as file:
        for name in patient_names:
            file.write(f"{name}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use Grover's algorithm to search patients for a target DNA codon."
    )
    parser.add_argument("--patients-file", default=PATIENTS_FILE)
    parser.add_argument("--target-codon", default=DEFAULT_TARGET_CODON)
    parser.add_argument(
        "--codon-index",
        type=int,
        default=None,
        help="Optional zero-based codon position to check. Omit this to search all codons.",
    )
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--draw-circuit",
        action="store_true",
        help="Save the Grover circuit diagram to grover_patient_circuit.png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_codon = normalize_codon(args.target_codon)
    patients = load_patients(args.patients_file)

    unhealthy_classical, classical_time = run_classical_search(
        patients, target_codon, args.codon_index
    )
    grover_result = run_grover_search(
        patients, target_codon, codon_index=args.codon_index, shots=args.shots
    )

    unhealthy_patients = grover_result["exact_unhealthy_patients"]
    measured_unhealthy_patients = grover_result["measured_unhealthy_patients"]
    save_patient_names(unhealthy_patients, args.output_file)

    print("=== Patient Mutation Search ===")
    print(f"Patients loaded: {len(patients)}")
    print(f"Target codon: {target_codon}")
    print(f"Codon index: {args.codon_index if args.codon_index is not None else 'any'}")
    print()
    print("=== Classical Linear Search ===")
    print(f"Matches found: {len(unhealthy_classical)}")
    print(f"Comparisons: {len(patients)} patients")
    print(f"Runtime: {classical_time:.8f}s")
    print()
    print("=== Grover Search Simulation ===")
    print(f"Oracle marked states: {len(grover_result['marked_indices'])}")
    print(f"Grover iterations: {grover_result['iterations']}")
    print(f"Measurements: {args.shots} shots")
    print(f"Matched patients observed in measurements: {len(measured_unhealthy_patients)}")
    print(f"Runtime: {grover_result['quantum_time']:.8f}s")
    print(f"Unhealthy patients saved to: {args.output_file}")
    print(f"unhealthy_patients = {unhealthy_patients}")

    if args.draw_circuit and grover_result["circuit"] is not None:
        grover_result["circuit"].draw(output="mpl", filename="grover_patient_circuit.png")
        print("Circuit diagram saved to: grover_patient_circuit.png")

    if set(measured_unhealthy_patients) != set(unhealthy_classical):
        print()
        print("Note: Grover measurements did not observe every marked patient.")
        print("Increase --shots if the target codon appears in many patients.")


if __name__ == "__main__":
    main()
