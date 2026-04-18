from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import math

# -----------------------------
# Constants
# -----------------------------
NUM_QUBITS   = 10
NUM_PATIENTS = 2 ** NUM_QUBITS   # 1024 patients
TARGET_PATIENT = 37              # the one patient with a mutation

# -----------------------------
# Phase 2: Patient Database
# -----------------------------

REFERENCE = "ATGGCTTATGATTGGCAGAAACGTACTGGTCCTTCTTTCGAACATAATGTTTTAATTTGCATGGCTTATGATTGGCAGAAACGTACTGGT"

# Codon 4 (positions 12-14): TGG (Tryptophan) -> TGA (Stop codon)
# Nonsense mutation — protein truncated prematurely
MUTATED = REFERENCE[:12] + "TGA" + REFERENCE[15:]


def generate_patients():
    patients = []
    for i in range(NUM_PATIENTS):
        if i == TARGET_PATIENT:
            patients.append(MUTATED)
        else:
            patients.append(REFERENCE)
    return patients


def verify_database(patients):
    mutated = [i for i, p in enumerate(patients) if p != REFERENCE]
    print("=== Database Verification ===")
    print(f"Total patients       : {len(patients)}")
    print(f"Mutated patient(s)   : {mutated}")
    print(f"All same length      : {len(set(len(p) for p in patients)) == 1}")
    for i in mutated:
        print(f"Patient {i}: codon 4 = '{patients[i][12:15]}' (ref = '{REFERENCE[12:15]}')")
    print()
# -----------------------------
# Phase 3: Classical Benchmark
# -----------------------------

def classical_search(patients):
    steps = 0
    for i, patient in enumerate(patients):
        steps += 1
        if patient != REFERENCE:
            return i, steps
    return None, steps


# -----------------------------
# Phase 4: Oracle
# -----------------------------

def apply_oracle(qc, target_index, n_qubits):
    target_bits = format(target_index, f'0{n_qubits}b')

    for i, bit in enumerate(reversed(target_bits)):
        if bit == '0':
            qc.x(i)

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    for i, bit in enumerate(reversed(target_bits)):
        if bit == '0':
            qc.x(i)


# -----------------------------
# Phase 5: Diffuser
# -----------------------------

def apply_diffuser(qc, n_qubits):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


# -----------------------------
# Phase 6: Grover Circuit
# -----------------------------

def build_grover_circuit(target_index, n_qubits):
    num_iterations = round(math.pi / 4 * math.sqrt(2 ** n_qubits))
    qc = QuantumCircuit(n_qubits, n_qubits)

    qc.h(range(n_qubits))
    qc.barrier()

    for _ in range(num_iterations):
        apply_oracle(qc, target_index, n_qubits)
        qc.barrier()
        apply_diffuser(qc, n_qubits)
        qc.barrier()

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, num_iterations


# -----------------------------
# Phase 7: Run and Compare
# -----------------------------

def main():
    import time

    patients = generate_patients()
    verify_database(patients)

    # Classical
    print("=== Classical Search ===")
    t0 = time.time()
    found_classical, classical_steps = classical_search(patients)
    classical_time = time.time() - t0
    print(f"Found patient    : {found_classical}")
    print(f"Steps taken      : {classical_steps} / {NUM_PATIENTS}")
    print(f"Time             : {classical_time:.6f}s")
    print()

    # Quantum
    print("=== Quantum Search ===")
    qc, num_iterations = build_grover_circuit(TARGET_PATIENT, NUM_QUBITS)
    t0 = time.time()
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=1024).result()
    quantum_time = time.time() - t0
    counts = result.get_counts()

    top_state = max(counts, key=counts.get)
    found_quantum = int(top_state, 2)
    print(f"Found patient    : {found_quantum}")
    print(f"Iterations used  : {num_iterations} / {NUM_PATIENTS}")
    print(f"Simulation time  : {quantum_time:.6f}s")
    print()

    # Comparison
    print("=== Speedup Comparison ===")
    print(f"Classical steps  : {classical_steps}")
    print(f"Quantum steps    : {num_iterations}")
    print(f"Algorithmic speedup : {classical_steps / num_iterations:.1f}x fewer steps")
    print("Note: simulation runs on a classical CPU so wall time is not a fair comparison.")
    print("On real quantum hardware, the step reduction is the actual speedup.")
    print()

    # Phase 8: Visualise
    fig = plot_histogram(counts, title=f"Grover Search — Mutated Patient {TARGET_PATIENT}")
    fig.savefig("grover_patient_histogram.png")
    print("Histogram saved to grover_patient_histogram.png")
    fig.show()


if __name__ == "__main__":
    main()