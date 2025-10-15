from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector
from transform_utils import (
    ctrl_twos_complement,
)


def qst_type_I(
    circuit: QuantumCircuit, anc_qubits: list[int], target_qubits: list[int]
) -> QuantumCircuit:
    """
    Type-I Quantum Sine Transform (QST-I) using a QFT-based A-gate structure.

    Mathematical definition:
        S_N^{I} = sqrt(2 / N) [ sin(m n π / N) ],   for m, n = 1, 2, …, N − 1

    Description:
        This implementation constructs QST-I by using an auxiliary unitary A_N
        that produces the antisymmetric superposition required for the sine basis,
        avoiding the need for large multi-controlled gates in earlier algorithms.

    Gate structure:
        - A_N  = (controlled two’s complement) ∘ H(ctrl)
        - Apply QFT_{2N} on all qubits.
        - A_N^{-1} = H(ctrl) ∘ (controlled two’s complement)

    Notes:
        - ctrl = target_qubits[-1] is the control qubit.
        - anc_qubits provide the carry chain for the controlled two’s complement.
    """

    ctrl = target_qubits[-1]  # control qubit for the transformation

    circuit.x(ctrl)  # X on control
    # --- A_N block -------------------------------------------------------------
    circuit.h(ctrl)  # H on control
    ctrl_twos_complement(
        circuit, anc_qubits, target_qubits
    )  # controlled two’s complement
    # --------------------------------------------------------------------------

    # Apply QFT_{2N} across the entire register
    circuit.append(QFTGate(len(target_qubits)), target_qubits)

    # --- A_N^{-1} block --------------------------------------------------------
    ctrl_twos_complement(
        circuit, anc_qubits, target_qubits
    )  # inverse two’s complement (self-inverse)
    circuit.h(ctrl)  # H on control
    # --------------------------------------------------------------------------

    # Clear the global phase introduced by the QFT
    # Negate the phase on the |1> state of the control qubit
    circuit.sdg(ctrl)  # S† to remove global i phase
    circuit.x(ctrl)  # X on control

    return circuit


# if __name__ == "__main__":

#     import numpy as np

#     def t1_sin_mat(N):

#         dst_mat = np.zeros((N, N), dtype=np.float64)

#         for i in range(N):
#             for j in range(N):
#                 dst_mat[i, j] = np.sin(np.pi * i * j / N)

#         return dst_mat

#     def cst(input, N):
#         x = np.zeros(N)
#         x[input] = 1
#         sMAt = t1_sin_mat(N)
#         result = np.matmul(sMAt, x)
#         normalized_result = result / np.linalg.norm(result)

#         return normalized_result

#     def t1_cosin_mat(N):

#         dst_mat = np.zeros((N, N), dtype=np.float64)

#         for i in range(N):
#             for j in range(N):
#                 dst_mat[i, j] = np.cos(np.pi * (i) * (j + 1 / 2) / N)
#                 if i == N - 1:
#                     dst_mat[i, j] *= 1 / np.sqrt(2)

#         return dst_mat

#     def cct(input, N):
#         x = np.zeros(N)
#         x[input] = 1
#         sMAt = t1_cosin_mat(N)
#         result = np.matmul(sMAt, x)
#         normalized_result = result / np.linalg.norm(result)

#         return normalized_result

#     from qiskit.quantum_info import Statevector

#     def statevector_to_dirac(sv: Statevector, threshold: float = 1e-10) -> str:
#         """Convert a Qiskit Statevector into a Dirac-notation string."""
#         amps = sv.data
#         n_qubits = int(np.log2(len(amps)))
#         terms = []
#         for idx, amp in enumerate(amps):
#             if abs(amp) > threshold:
#                 # Format amplitude neatly
#                 re, im = amp.real, amp.imag
#                 if abs(im) < threshold:
#                     amp_str = f"{re:.4g}"
#                 elif abs(re) < threshold:
#                     amp_str = f"{im:.4g}j"
#                 else:
#                     amp_str = f"{re:.4g}{im:+.4g}j"
#                 # Binary label, MSB on the left
#                 bitstr = format(idx, f"0{n_qubits}b")
#                 terms.append(f"{amp_str}|{bitstr}⟩")
#         return " + ".join(terms) if terms else "0"

#     result = cst(5, 2**3)

#     print("CST result:", np.round(result, 4))

#     qc1a = QuantumCircuit(5)
#     # qc1a.x(1)
#     # qc1a.x(0)

#     target_qubits = list(range(0, 4))
#     anc_qubits = list(range(4, 5))

#     qst_type_I(qc1a, anc_qubits, target_qubits)

#     print(qc1a.draw())

#     init = Statevector.from_label("00101")
#     output = init.evolve(qc1a)

#     # print("======================Quantum Circuit Output===================")
#     # print(output.data)
#     # print(output.probabilities([3]))
#     # print("sum of output:", np.sum(output.data))

#     state_result = statevector_to_dirac(output)
#     print("Statevector in Dirac notation:", state_result)
