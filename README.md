# QRTlib

This library implements **Quantum Real Transforms (QRTs)**, including:

- **Quantum Hartley Transform (QHT)**
  - *Approach 1:* Recursive QHT proposed in [1].
  - *Approach 2:* Linear Combination of Unitaries (LCU) implementation (most efficient).

- **Quantum Sine and Cosine Transforms**
  - *Approach:* Optimized implementation of [2].

---

## Installation

To install dependencies:
```bash
pip install -r requirements.txt

```

## Usage example
`QHTGate` is one of the core component of this library. It constructs a quantum circuit implementing
the Quantum Hartley Transform (QHT) using either:
- **Recursive** method (`type="REC"`)
- **Linear Combination of Unitaries (LCU)** method (`type="LCU"`)

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from QHTGate import QHTGate
import numpy as np

# helper function to extract amplitudes of data qubits
def extract_data_register_amplitudes(sv, data_indices):
    total_qubits = int(np.log2(len(sv)))
    """
    Extract amplitudes of data qubits ONLY when all other qubits (ancilla) are in |0⟩.

    Args:
        sv: full statevector (np.ndarray or Statevector.data)
        data_indices: list[int], indices of data qubits (physical positions in circuit)

    Returns:
        np.ndarray of size 2^len(data_indices) with complex amplitudes
    """
    n = len(data_indices)
    out = np.zeros(2**n, dtype=complex)

    all_indices = set(range(total_qubits))
    ancilla_indices = list(all_indices - set(data_indices))

    for i, amp in enumerate(sv):
        bin_str = f"{i:0{total_qubits}b}"[::-1]  
        
        if any(bin_str[j] != "0" for j in ancilla_indices):
            continue

        q_bits = "".join([bin_str[j] for j in data_indices])
        q_idx = int(q_bits[::-1], 2)
        out[q_idx] = amp
    return out


# example for quantum hartley transform of "0000" using LCU method
gate = QHTGate(4, type="LCU")
qc = QuantumCircuit(8)
qc.append(gate, list(range(8)))

full_bits = "0" * 4 + "0000"

index = int(full_bits, 2)
sv = Statevector.from_int(index, dims=2**qc.num_qubits)
evolved = sv.evolve(qc)

quantum_output = extract_data_register_amplitudes(
    evolved.data, data_indices=list(range(4))
)

print(quantum_output)
```


## References

[1] Doliskani, Jake and Mirzaei, Morteza and Mousavi, Ali. "Public-key quantum money and fast real transforms". (2025)
[2] Klappenecker, Andreas and Rotteler, Martin. "Discrete cosine transforms on quantum computers". (2001)
