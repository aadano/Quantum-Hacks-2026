# test.py
import qiskit
import qiskit_aer
import streamlit
import matplotlib
import numpy
import pylatexenc
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

print("All dependencies installed!")

# Build a tiny test circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Run it
simulator = AerSimulator()
compiled = transpile(qc, simulator)
result = simulator.run(compiled, shots=1000).result()
counts = result.get_counts()

print(counts)  # Should print something like {'00': 503, '11': 497}