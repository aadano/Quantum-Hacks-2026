# pip install qiskit qiskit-aer matplotlib numpy pylatexenc

import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

from grover_dna_search import (
    generate_patients,
    build_grover_circuit,
    classical_search,
    TARGET_PATIENT,
    NUM_QUBITS,
    NUM_PATIENTS,
)

if __name__ == "__main__":
    patients = generate_patients()

    # Classical search
    t0 = time.time()
    found_classical, classical_steps = classical_search(patients)
    classical_time = time.time() - t0
    print(f"Classical: found patient {found_classical} in {classical_steps} steps, {classical_time:.8f}s")

    # Quantum search
    qc, num_iterations = build_grover_circuit(TARGET_PATIENT, NUM_QUBITS)
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    t0 = time.time()
    result = simulator.run(compiled, shots=1024).result()
    quantum_time = time.time() - t0
    counts = result.get_counts()
    top_state = max(counts, key=counts.get)
    found_quantum = int(top_state, 2)
    print(f"Quantum:   found patient {found_quantum} in {num_iterations} iterations, {quantum_time:.8f}s")
    print(f"Speedup: {classical_steps / num_iterations:.1f}x fewer steps\n")

    # Visual 1: Grover measurement histogram
    fig = plot_histogram(counts, title=f"Grover Search — Mutated Patient {TARGET_PATIENT}")
    fig.savefig("grover_patient_histogram.png")
    print("Saved: grover_patient_histogram.png")

    # Visual 2: steps comparison bar chart
    plt.figure()
    plt.bar(["Classical", "Quantum"], [classical_steps, num_iterations], color=["steelblue", "darkorange"])
    plt.ylabel("Steps / Iterations")
    plt.title(f"Search Steps: {NUM_PATIENTS} Patients")
    plt.savefig("steps_comparison.png")
    print("Saved: steps_comparison.png")

    # Visual 3: complexity scaling chart
    N = np.arange(1, 2000)
    plt.figure()
    plt.plot(N, N, label="Classical O(N)", color="steelblue")
    plt.plot(N, np.sqrt(N), label="Quantum O(√N)", color="darkorange")
    plt.xlabel("Database Size (N)")
    plt.ylabel("Steps")
    plt.title("Classical vs Quantum Search Complexity")
    plt.legend()
    plt.savefig("complexity_comparison.png")
    print("Saved: complexity_comparison.png")

    plt.show()
