from qiskit import QuantumCircuit
from qiskit_aer import Aer

# Build quantum circuit
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

# Get the simulator
simulator = Aer.get_backend('qasm_simulator')

# Run the simulation
job = simulator.run(qc, shots=1000)
result = job.result()
counts = result.get_counts()

print("Quantum coin flip results:", counts)