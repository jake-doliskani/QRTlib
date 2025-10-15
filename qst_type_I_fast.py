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
