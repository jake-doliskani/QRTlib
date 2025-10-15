from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, MCXGate
from qiskit.quantum_info import Statevector
from transform_utils import (
    compute_or_forward,
    compute_or_backward,
    ctrl_dec_by_1,
    V_N_gate,
    D_One_gate,
    ctrl_twos_complement,
)


def qst_type_II(
    circuit: QuantumCircuit, anc_qubits: list[int], target_qubits: list[int]
):
    """
    Type-II Quantum Sine Transform (QST-II).

    Mathematical definition:
        S_N^{II} = sqrt(2 / N) [ k_m * sin(m (n − 1/2) π / N) ],   for m, n = 1, 2, …, N
        where k_m = 1 / sqrt(2) for m = N, and k_m = 1 otherwise.

    Description:
        Implements the Type-II sine transform branch from the shared core
        unitary U_N^* · QFT_{2N} · V_N. The sine branch corresponds to the
        control qubit being initialized to |1⟩. After applying the shared
        transformation, an additional Z gate is used to remove the global
        phase introduced by the Fourier transform.

    Implementation steps:
        1) X(ctrl): prepare control in |1⟩ to select the sine branch.
        2) _qst_qct_transform_type_II(...): apply the shared Type-II transform core.
        3) Z(ctrl): remove global phase.
        4) X(ctrl): return control to |0⟩.

    Notes:
        - ctrl = target_qubits[-1] is the control qubit.
        - anc_qubits provide carry and workspace for internal arithmetic.
    """
    ctrl = target_qubits[-1]  # control qubit

    # 1) Prepare control in |1⟩ to activate sine branch
    circuit.x(ctrl)

    # 2) Apply the shared Type-II transform core
    _qst_qct_transform_type_II(circuit, anc_qubits, target_qubits)

    # 3) Remove global phase introduced by QFT
    circuit.z(ctrl)

    # 4) Return control to |0⟩
    circuit.x(ctrl)


def qct_type_II(
    circuit: QuantumCircuit, anc_qubits: list[int], target_qubits: list[int]
):
    """
    Type-II Quantum Cosine Transform (QCT-II).

    Mathematical definition:
        C_N^{II} = sqrt(2 / N) [ k_m * cos(m (n + 1/2) π / N) ],   for m, n = 0, 1, …, N − 1
        where k_m = 1 / sqrt(2) for m = 0, and k_m = 1 otherwise.

    Description:
        Implements the Type-II cosine transform branch from the shared core
        unitary U_N^* · QFT_{2N} · V_N. The cosine branch corresponds to
        the control qubit being initialized to |0⟩. Unlike the sine branch,
        no additional phase correction is needed.

    Implementation steps:
        1) Ensure control qubit is in |0⟩ (cosine branch).
        2) _qst_qct_transform_type_II(...): apply the shared Type-II transform core.

    Notes:
        - ctrl = target_qubits[-1] is the control qubit.
        - anc_qubits provide carry and workspace for internal arithmetic.
    """
    # 1) Apply the shared Type-II transform core for the cosine branch
    _qst_qct_transform_type_II(circuit, anc_qubits, target_qubits)


def _qst_qct_transform_type_II(
    circuit: QuantumCircuit, anc_qubits: list[int], target_qubits: list[int]
):
    """
    Shared core circuit for the Type-II Quantum Cosine and Sine Transforms.

    Description:
        This subroutine implements the joint transform block that forms the
        basis of both QCT-II and QST-II. It corresponds to the unitary:
            U_N^* · QFT_{2N} · V_N

        In the overall architecture:
          - QCT-II uses this block with the control qubit in |0⟩ (cosine branch).
          - QST-II uses this block with the control qubit in |1⟩ (sine branch).

        The decomposition follows the formulation by Klappenecker and Roetteler (2001),
        where U_N and V_N are auxiliary unitaries that prepare and unprepare
        the data/control entanglement necessary for the Fourier-based cosine/sine transforms.

    Implementation structure:
        1) V_N_gate: prepares the input register and control qubit correlations
           for the Type-II transformation.
        2) QFT_{2N}: performs the quantum Fourier transform over the joint system.
        3) U_N_dagger_gate: reverses the V_N preparation and separates the cosine/sine outputs.

    Notes:
        - target_qubits[-1] is the control qubit (ctrl).
        - anc_qubits are used inside U_N_dagger_gate for controlled two’s-complement
          and carry-chain operations.
        - This block is shared by both QCT-II and QST-II and does not include
          any phase or branch-specific corrections.
    """

    # 1) Apply V_N gate (prepares the entangled input state)
    V_N_gate(circuit, target_qubits)

    # 2) Apply QFT_{2N} to the entire register
    circuit.append(QFTGate(len(target_qubits)), target_qubits)

    # 3) Apply U_N† (unprepare step that extracts cosine/sine subspaces)
    U_N_dagger_gate(circuit, anc_qubits, target_qubits)


def U_N_dagger_gate(
    circuit: QuantumCircuit, anc_qubits: list[int], target_qubits: list[int]
):
    """
    Apply U_N dagger gate to the circuit.

    U_N = pi_2^-1 (DD) pi^-1 D_one
    """

    # apply s gate first

    # for data_idx in target_qubits[-2::-1]:
    #     circuit.cx(target_qubits[-1], data_idx)

    D_One_gate(circuit, target_qubits)  # apply D_One gate

    # controlled addition by 1 only need n-2 ancilla qubits
    ctrl_twos_complement(
        circuit, anc_qubits[1:], target_qubits
    )  # apply pi inverse gate

    # # check if data qubits are all 0 or not
    operation_log = compute_or_forward(circuit, target_qubits, anc_qubits, False)

    # apply DD gate
    G_gate(circuit, target_qubits, anc_qubits)
    compute_or_backward(circuit, operation_log)

    ctrl_dec_by_1(circuit, anc_qubits[1:], target_qubits)


def G_gate(circuit: QuantumCircuit, target_qubits: list[int], anc_qubits: list[int]):
    """
    Apply DD gate to the circuit.

    DD: use sdg and a control J gate

    first check if data qubits are all 0 or not
    """

    # B^t = S H
    circuit.h(target_qubits[-1])  # apply control H gate
    circuit.s(target_qubits[-1])  # apply control

    # J = Sdag H Sdag
    # apply when x is zero
    circuit.x(anc_qubits[0])  # apply X gate to the control qubit
    circuit.csdg(anc_qubits[0], target_qubits[-1])  # apply control S-dagger gate
    circuit.ch(anc_qubits[0], target_qubits[-1])  # apply control H gate
    circuit.csdg(anc_qubits[0], target_qubits[-1])  # apply control S-dagger gate
    circuit.x(anc_qubits[0])
