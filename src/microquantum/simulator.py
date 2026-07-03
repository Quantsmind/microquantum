import numpy as np

class QuantumCircuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0

    def _apply_gate(self, gate: np.ndarray, target: int):
        dim_before = 2**target
        dim_after = 2**(self.num_qubits - target - 1)
        self.state = self.state.reshape((dim_before, 2, dim_after))
        self.state = np.einsum('ij,ajb->aib', gate, self.state)
        self.state = self.state.flatten()

    def h(self, target: int):
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._apply_gate(H, target)

    def measure(self, shots: int = 1000) -> dict:
        probabilities = np.abs(self.state) ** 2
        probabilities /= np.sum(probabilities)
        outcomes = np.random.choice(2**self.num_qubits, size=shots, p=probabilities)
        
        counts = {}
        for outcome in outcomes:
            binary_str = format(outcome, f'0{self.num_qubits}b')
            counts[binary_str] = counts.get(binary_str, 0) + 1
        return dict(sorted(counts.items()))
