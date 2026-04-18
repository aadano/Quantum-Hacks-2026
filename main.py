# run these dependencies in terminal to install necessary libraries
#pip install qiskit qiskit-aer qiskit-ibm-runtime streamlit matplotlib numpy pylatexenc


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import GroverOperator
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector

