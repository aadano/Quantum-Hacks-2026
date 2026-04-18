import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

from grover_patient_search import run_grover_search, build_grover_circuit
from Classical_search import linear_search



if __name__ == "__main__":
    # Classical search
    position, steps, classical_time = linear_search("ATCGGCTA", "TA")
    print(f"Classical: found 'TA' at position {position} in {steps} steps, {classical_time:.8f}s")

    # Quantum search
    counts, quantum_time, top_state, found_sequence = run_grover_search()
    print(f"Quantum:   found '{found_sequence}' (|{top_state}>) in 1 step, {quantum_time:.8f}s")

    # Circuit diagram
    qc = build_grover_circuit()
    qc.draw(output="mpl", filename="circuit.png")

    # Grover histogram
    fig = plot_histogram(counts, title="Grover Search: DNA Target 'TA'")
    fig.savefig("grover_dna_histogram.png")

    # Steps comparison
    plt.figure()
    plt.bar(["Classical", "Quantum"], [steps, 1])
    plt.ylabel("Steps to find target")
    plt.title("Search Steps Comparison")
    plt.savefig("steps_comparison.png")

    # Complexity scaling
    N = np.arange(1, 1001)
    plt.figure()
    plt.plot(N, N, label="Classical O(N)")
    plt.plot(N, np.sqrt(N), label="Quantum O(√N)")
    plt.xlabel("Database Size (N)")
    plt.ylabel("Steps")
    plt.title("Classical vs Quantum Complexity")
    plt.legend()
    plt.savefig("complexity_comparison.png")

    plt.show()
