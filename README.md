# QRTlib

This library implements **Quantum Real Transforms (QRTs)**, including:

- **Quantum Hartley Transform (QHT)**
  - *Approach 1:* Recursive QHT proposed in [1].
  - *Approach 2:* Using Linear Combination of Unitaries (LCU) proposed in [2].

- **Quantum Sine and Cosine Transforms**
- *Approach:* Optimized implementation of [3].
- *Types:*
  - Quantum Sine and Cosine Transform type I
  - Quantum Sine and Cosine Transform type II
  - Quantum Sine and Cosine Transform type III
  - Quantum Sine and Cosine Transform type IV
    

---

## Installation

To install dependencies:
```bash
pip install -r requirements.txt

```

## Usage example
The following snippet adds Quantum Hartley Transform (QHT) to your circuit using LCU method:

- *Notes:* For recursive method one can use (`type="REC"`).


```python
from qiskit import QuantumCircuit
from QHTGate import QHTGate



# Example for quantum hartley transform of using LCU method for 4 qubits
gate = QHTGate(4, type="LCU")
qc = QuantumCircuit(8)
qc.append(gate, list(range(8)))

```


## References

[1] Doliskani, Jake and Mirzaei, Morteza and Mousavi, Ali. "Public-key quantum money and fast real transforms". (2025)

[2] Ahmadkhaniha, Armin and Doliskani, Jake and Chen, Lu and Sun, Zhifu. "QRTlib: A Library for Fast Quantum Real Transforms". (2025)

[3] Klappenecker, Andreas and Rotteler, Martin. "Discrete cosine transforms on quantum computers". (2001)
