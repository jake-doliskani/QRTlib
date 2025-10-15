from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, Diagonal
from qiskit.quantum_info import Statevector
from transform_utils import D_One_gate, ctrl_ones_complement
import numpy as np


def qst_type_IV(circuit: QuantumCircuit, target_qubits: list[int]):
    """
    Type-IV Quantum Sine Transform (QST-IV).

    Description:
        Implements the Type-IV sine branch using the shared core
        M · U_N^T · QFT_{2N} · U_N, with the control qubit set to |1⟩.
        Surrounding single-qubit gates prepare/select the sine branch and
        correct the final phase as required by the Type-IV basis.

    Steps:
        1) X(ctrl): select sine branch (ctrl = |1⟩).
        2) _qst_qct_transform_type_IV(...): apply shared Type-IV core.
        3) S(ctrl): phase adjustment for the sine branch.
        4) X(ctrl): return control to |0⟩.

    Notes:
        - ctrl = target_qubits[-1] is the control qubit.
    """
    ctrl = target_qubits[-1]  # control qubit

    # 1) Prepare control in |1⟩ to select the sine branch
    circuit.x(ctrl)

    # 2) Apply the shared Type-IV transform core
    _qst_qct_transform_type_IV(circuit, target_qubits)

    # 3) Phase tweak for the sine branch (S, not S†)
    circuit.s(ctrl)

    # 4) Restore control to |0⟩
    circuit.x(ctrl)


def qct_type_IV(circuit: QuantumCircuit, target_qubits: list[int]):
    """
    Type-IV Quantum Cosine Transform (QCT-IV).

    Description:
        Implements the Type-IV cosine branch using the shared core
        M · U_N^T · QFT_{2N} · U_N, with the control qubit left in |0⟩.
        No extra phase correction is required for the cosine branch here.

    Steps:
        1) Ensure ctrl = |0⟩ (cosine branch).
        2) _qst_qct_transform_type_IV(...): apply shared Type-IV core.

    Notes:
        - ctrl = target_qubits[-1] is the control qubit.
    """
    # 1) Apply the shared Type-IV transform core for the cosine branch
    _qst_qct_transform_type_IV(circuit, target_qubits)


def _qst_qct_transform_type_IV(circuit: QuantumCircuit, target_qubits: list[int]):
    """
    Shared core for Type-IV cosine/sine transforms:
        M · U_N^T · QFT_{2N} · U_N

    Structure implemented here:
        - U_N  = π₁ · D₂ · (H S* on ctrl)
                where  D₂ = (C ⊗ I_N) · (Δ₁ ⊕ Δ₂), with the correction:
                       when is_L_adjoint=True in D_One_gate, we set Δ₂ = Δ₁*.
        - QFT_{2N}
        - U_N^T (transpose of U_N)
        - M = diag(ω_{4N}, ω_{4N}) ⊗ I_N (global phase on ctrl)

    Gate mapping to code:
        • (H S*) on ctrl           →  sdg(ctrl); h(ctrl)
        • D₂                        →  D_One_gate(..., is_L_adjoint=True)   # sets Δ₂ = Δ₁*
        • π₁ (bitwise complement)   →  ctrl_ones_complement(...)
        • QFT_{2N}                  →  QFTGate(len(target_qubits)) on full register
        • U_N^T                     →  (same blocks again, arranged as transpose)
        • M                         →  create_m_gate(2 ** (len(target_qubits) - 1)) on ctrl
    """
    ctrl = target_qubits[-1]

    # U_N: (H S*) on ctrl
    circuit.sdg(ctrl)
    circuit.h(ctrl)

    # U_N: D₂ with Δ₂ = Δ₁* (via is_L_adjoint=True)
    D_One_gate(circuit, target_qubits, is_L_adjoint=True)

    # U_N: π₁
    ctrl_ones_complement(circuit, target_qubits)

    # QFT_{2N}
    circuit.append(QFTGate(len(target_qubits)), target_qubits)

    # U_N^T: π₁
    ctrl_ones_complement(circuit, target_qubits)

    # U_N^T: D₂^T (same call pattern; Δ₂ = Δ₁* is still in effect)
    D_One_gate(circuit, target_qubits, is_L_adjoint=True)

    # U_N^T: (H S*)^T = H then S† on ctrl
    circuit.h(ctrl)
    circuit.sdg(ctrl)

    # M on ctrl
    m_gate = create_m_gate(2 ** (len(target_qubits) - 1))
    circuit.append(m_gate, [ctrl])


def create_m_gate(N):
    # M = diag(ω_{4N}, ω_{4N}) on ctrl (global phase adjust)
    phase = np.pi / (4 * N)

    qc = QuantumCircuit(1, name=f"M(N={N})")
    qc.global_phase = phase

    return qc.to_instruction()
