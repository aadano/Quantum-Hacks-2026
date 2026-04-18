import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

from grover_patient_search import (
    load_patients,
    run_classical_search,
    run_grover_search,
    DEFAULT_TARGET_CODON,
    PATIENTS_FILE
)

if __name__ == "__main__":
    patients = load_patients(PATIENTS_FILE)
    target_codon = DEFAULT_TARGET_CODON
    num_patients = len(patients)

    # Classical search
    t0 = time.time()
    unhealthy_classical, classical_time = run_classical_search(patients, target_codon, codon_index=0)
    classical_steps = num_patients
    print(f"Classical: found {len(unhealthy_classical)} patients in {classical_time:.8f}s")

    # Quantum search
    grover_result = run_grover_search(patients, target_codon, codon_index=0)
    counts = grover_result["counts"]
    num_iterations = grover_result["iterations"]
    quantum_time = grover_result["quantum_time"]
    qc = grover_result["circuit"]

    print(f"Quantum:   found {len(grover_result['measured_unhealthy_patients'])} patients in {num_iterations} iterations, {quantum_time:.8f}s")
    if num_iterations > 0:
        print(f"Speedup: {classical_steps / num_iterations:.1f}x fewer steps\n")

    # Circuit diagram
    if qc is not None:
        fig = qc.draw(output="mpl", fold=20, style={"backgroundcolor": "#FFFFFF"})
        fig.savefig("circuit.png", dpi=150, bbox_inches="tight")
        fig.savefig("circuit.svg", bbox_inches="tight")
        print("Saved: circuit.png")

    # Grover measurement histogram
    if counts:
        readable_counts = {f"Patient {int(k, 2)}": v for k, v in counts.items()}
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(readable_counts.keys(), readable_counts.values(), color="darkorange")
        ax.set_xlabel("Patient")
        ax.set_ylabel("Measurement Count")
        ax.set_title(f"Grover Search: DNA Target '{target_codon}'")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig("grover_dna_histogram.png")
        print("Saved: grover_dna_histogram.png")

    # Steps comparison bar chart
    if num_iterations > 0:
        plt.figure()
        plt.bar(["Classical", "Quantum"], [classical_steps, num_iterations], color=["steelblue", "darkorange"])
        plt.ylabel("Steps / Iterations")
        plt.title(f"Search Steps: {num_patients} Patients")
        plt.savefig("steps_comparison.png")
        print("Saved: steps_comparison.png")

    # Complexity scaling chart
    N_vals = np.arange(1, 2000)
    plt.figure()
    plt.plot(N_vals, N_vals, label="Classical O(N)", color="steelblue")
    plt.plot(N_vals, np.sqrt(N_vals), label="Quantum O(√N)", color="darkorange")
    plt.xlabel("Database Size (N)")
    plt.ylabel("Steps")
    plt.title("Classical vs Quantum Search Complexity")
    plt.legend()
    plt.savefig("complexity_comparison.png")
    print("Saved: complexity_comparison.png")

    plt.show()
