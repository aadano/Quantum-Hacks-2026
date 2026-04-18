# Quantum-Hacks-2026
### Genetic Sequence Search Simulator

A quantum computing simulation that demonstrates how Grover's Algorithm searches a patient DNA database exponentially faster than a classical linear search.

Built with Python, Qiskit, and Matplotlib for QuantumHacks 2026.

---

## What It Does

Given a database of 1024 patients with DNA sequences, the program finds the one patient carrying a target genetic mutation:

- **Classical search** scans patients one by one — up to 1024 comparisons, O(N)
- **Quantum search** uses Grover's Algorithm on a 10-qubit circuit — ~25 iterations, O(√N)

The simulation runs on a classical CPU via Qiskit's AerSimulator. Wall clock time is not the metric — **step count is**. On real quantum hardware, the ~40x reduction in steps is the actual speedup.

---

## Project Structure

| File | Description |
|---|---|
| `main.py` | Entry point — runs both searches and generates all visuals |
| `grover_dna_search.py` | Quantum circuit: oracle, diffuser, 10-qubit Grover circuit |
| `grover_patient_search.py` | Extended patient search system with codon-level mutation targeting |
| `classicsearch.py` | Classical linear search baseline |
| `test.py` | Dependency and simulator validation |

---

## Setup

```bash
pip install qiskit qiskit-aer matplotlib numpy pylatexenc
```

---

## Run

```bash
python main.py
```

**Terminal output:**
```
Classical: found patient 37 in 38 steps, 0.00012s
Quantum:   found patient 37 in 25 iterations, 0.84s
Speedup: 1.5x fewer steps
```

**Generated visuals:**
- `grover_patient_histogram.png` — measurement distribution showing the target patient amplified
- `steps_comparison.png` — classical vs quantum step count bar chart
- `complexity_comparison.png` — O(N) vs O(√N) scaling chart

---

## How Grover's Algorithm Works

1. **Superposition** — Hadamard gates put all 1024 patient indices into equal superposition simultaneously
2. **Oracle** — marks the target patient by flipping its phase amplitude to negative
3. **Diffuser** — amplifies the marked state by reflecting all amplitudes about the mean
4. **Measurement** — collapses to the target patient with high probability

One Grover iteration = one oracle + one diffuser. Optimal iterations = π/4 · √N ≈ 25 for N=1024.

---

## The Quantum Advantage

| N (patients) | Classical steps | Quantum iterations |
|---|---|---|
| 4 | 4 | 1 |
| 1,024 | 1,024 | ~25 |
| 1,000,000 | 1,000,000 | ~785 |
