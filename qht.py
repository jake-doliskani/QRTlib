import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, MCXGate
from transform_utils import ctrl_twos_complement


# ========= qht using Linear Combination of Unitaries ==========


def _build_unitary_w(data: list[int]):
    """
    Construct the unitary W used in the Linear Combination of Unitaries (LCU)
    framework for the Quantum Hartley Transform.

    This unitary implements a controlled selection between two operations:
    - A global phase shift corresponding to U₀ = e^(-iπ/4) I
    - A controlled two's-complement transformation corresponding to U₁ = e^(iπ/4) T

    The selector qubit determines which unitary is applied. Ancilla qubits are
    used internally to realize the two's-complement operation on the data qubits.

    Returns:
        Gate: A Qiskit `Gate` object representing the W unitary.
    """
    n = len(data)

    qc = QuantumCircuit(2 * n - 1, name="w")
    control_qubit = 0
    data_qubits = list(range(1, n + 1))
    anc_qubits = list(range(n + 1, 2 * n - 1))
    
    qc.h(control_qubit)
    ctrl_twos_complement(qc, anc_qubits[0 : n - 2], data_qubits + [control_qubit])
    qc.rz(np.pi / 2, control_qubit)
    qc.h(control_qubit)

    return qc.to_gate(label="W")


def _build_unitary_reflection(qubits: list[int]):
    """
    Construct the reflection operator R over the all-zeros state |0...0⟩.

    This gate implements the transformation:
        R = I - 2|0...0⟩⟨0...0|

    Effectively, this applies a phase flip (-1) to the |0...0⟩ state,
    leaving all other basis states unchanged. Such reflections are
    commonly used in amplitude amplification and block-encoding constructions.

    Implementation details:
    - Surround the operation with X gates to map |0...0⟩ to |1...1⟩.
    - Apply a multi-controlled Z (phase flip) on the last qubit.
    - Map back with X gates.
    - Special case: for n=1 qubit, this reduces to a single Z gate.

    Args:
        qubits (list[int]): List of qubits to which the reflection is applied.

    Returns:
        Gate: A Qiskit `Gate` object representing the reflection operator R.
    """
    n = len(qubits)
    qc = QuantumCircuit(n, name="R")

    qc.x(range(n))

    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
        qc.h(n - 1)

    qc.x(range(n))

    return qc.to_gate(label="R")


def _build_unitary_s(
    circuit: QuantumCircuit,
    helper: int,
    selector: int,
    data: list[int],
    ancillas: list[int],
):
    """
    Construct and append the S′ operator for customized Oblivious Amplitude
    Amplification in the LCU implementation of the Quantum Hartley Transform.

    Background:
        In the standard Oblivious Amplitude Amplification scheme,
        the amplification operator is:
            S = -W R W† R
        where:
            - W is the unitary constructed from the LCU block-encoding,
            - R is the reflection about |0⟩ in the selector register.

        To achieve exact amplitude amplification with only one round,
        we extend the register with a second ancilla qubit (the "helper"),
        prepared in a custom state P|0⟩. This defines an extended operator:
            W′ = P ⊗ W
        and the new amplification operator becomes:
            S′ = -W′ R′ W′† R′
        where R′ is the reflection about |00⟩ in the (helper, selector) subspace.

    Implementation details:
        1. Reflect about the joint state of (selector, helper).
        2. Apply a Hadamard to the helper (part of preparing P).
        3. Conjugate the reflection with W†.
        4. Reflect again on (selector, helper).
        5. Apply another Hadamard on the helper.
        6. Conjugate with W.

    Args:
        circuit (QuantumCircuit): Circuit to which the S′ operator is appended.
        helper (int): Second ancilla qubit (prepares amplitude distribution P).
        selector (int): Original LCU selector qubit.
        data (list[int]): Data qubits on which the Hartley transform acts.
        ancillas (list[int]): Ancilla workspace for two's complement inside W.

    Returns:
        None: The operator is appended directly to the provided `circuit`.
    """
    circuit.append(_build_unitary_reflection([selector, helper]), [selector, helper])

    circuit.h(helper)
    circuit.append(
        _build_unitary_w(data).inverse(),
        [selector] + data + ancillas,
    )

    circuit.append(_build_unitary_reflection([selector, helper]), [selector, helper])

    circuit.h(helper)
    circuit.append(_build_unitary_w(data), [selector] + data + ancillas)


def _build_qht_lcu(
    circuit: QuantumCircuit,
    helper: int,
    selector: int,
    data: list[int],
    ancillas: list[int],
):
    """
    Construct the Quantum Hartley Transform (QHT) using the
    Linear Combination of Unitaries (LCU) framework with
    Oblivious Amplitude Amplification.

    Overview:
        The Hartley transform is implemented as a block-encoding
        via LCU:
            ν = (1/√2)(U₀ + U₁)
        where U₀ = e^{-iπ/4} I and U₁ = e^{iπ/4} T
        (T = two's-complement operator).

        The block-encoding unitary W is built as:
            W = (H ⊗ I) U (H ⊗ I)

        Exact amplitude amplification is achieved by introducing
        an additional helper qubit, defining the operator:
            S′ = -W′ R′ W′† R′
        with W′ = P ⊗ W and R′ a reflection about |00⟩ in
        (helper, selector).

        After amplification, a Quantum Fourier Transform (QFT)
        is applied on the data register. A global phase of π is
        added to correctly align the amplitudes with the
        classical Hartley transform.

    Special case:
        - For a single data qubit, the QHT reduces to a Hadamard gate.

    Args:
        circuit (QuantumCircuit): Circuit to which the QHT is appended.
        helper (int): Extra ancilla qubit used for Oblivious Amplitude Amplification preparation.
        selector (int): Selector qubit for the LCU unitary.
        data (list[int]): Data qubits on which the Hartley transform is applied.
        ancillas (list[int]): Ancilla qubits used for the two's-complement inside W.

    Returns:
        QuantumCircuit: The input circuit with the QHT appended.
    """
    n = len(data)
    if n == 1:
        circuit.h(data[0])
        return
    # Next three operators on the selector implement -1 in the S′
    circuit.x(selector)
    circuit.z(selector)
    circuit.x(selector)
    
    
    circuit.h(helper)
    circuit.append(_build_unitary_w(data), [selector] + data + ancillas)
    _build_unitary_s(circuit, helper, selector, data, ancillas)

    qft_circuit = QFTGate(num_qubits=n)
    circuit.append(qft_circuit, data)
    

    return circuit


# ==================== qht recursive ====================


def _build_unitary_UR(circuit, ctrl, target_y, target_b, N):
    """
    Apply the single-qubit rotation unitary U_R from the Quantum Hartley Transform (QHT).

    This unitary implements the map:
        |c⟩|y⟩|b⟩  ↦  (R(y,b)|c⟩)|y⟩|b⟩ ,
    where the rotation matrix is defined as
        R(y, b) = [[ cos(2π b y / N),  sin(2π b y / N)],
                [-sin(2π b y / N), cos(2π b y / N)] ].

    In the circuit, this is realized by applying controlled rotations on the control
    qubit `ctrl`, conditioned jointly on the bits of the y-register (`target_y`) and
    the auxiliary qubit `target_b`. Each qubit in the y-register contributes an angle
    of -4π·2^j / N to the rotation, where j is the qubit index, reproducing the
    frequency-dependent phase structure required by the Hartley transform recursion.

    Args:
        circuit (QuantumCircuit): Quantum circuit where U_R is appended.
        ctrl (Qubit): The control qubit |c⟩ that undergoes the R(y,b) rotation.
        target_y (list[Qubit]): Register encoding the integer y; each qubit contributes
            exponentially (2^j) to the rotation angle.
        target_b (list[Qubit]): Register encoding the bit b that modulates the sign of
            the rotation. Only the first qubit is used here.
        N (int): Transform size, setting the normalization of the rotation angles.

    Returns:
        None: The operation is appended in place onto the given circuit.
    """
    for j, y_qubit in enumerate(target_y):
        theta = -4 * np.pi * (2**j) / N
        circuit.mcry(theta, [y_qubit, target_b[0]], ctrl, [], mode=None)


def _build_qht_rec(circuit, ctrl_anc, data, anc):
    """
    Recursive implementation of the Quantum Hartley Transform (QHT).

    This function follows Algorithm 1 from the QHT recursive construction, reducing an
    n-qubit transform into a sequence of (n-1)-qubit subproblems. At each
    recursion level, the algorithm introduces an ancilla qubit, applies
    structured arithmetic operations, and inserts controlled rotations to
    encode the Hartley. The recursion terminates with a Hadamard gate
    when only a single data qubit remains.

    The steps mirror the pseudocode of Algorithm 1:

        1. Base case: for a single data qubit, apply H.
        2. Recursively compute QHT_{N/2} on the subregister.
        3. Apply H on the ancilla `ctrl`.
        4. Apply the controlled two's-complement (subtract from N/2).
        5. Apply the rotation unitary U_R, which implements R(y,b).
        6. Uncompute the controlled two's-complement.
        7. Apply a correction for the y=0 case using X, H, and multi-controlled X.
        8. Apply H on the ancilla `ctrl`.
        9. Apply CNOT between the last data qubit and the ancilla.
        10. Apply a final H on the last data qubit.

    Args:
        circuit (QuantumCircuit): Quantum circuit where QHT operations are appended.
        ctrl (Qubit): Ancilla/control qubit used in steps 3, 6, 8, and 9.
        data (list[Qubit]): Register on which the QHT is applied. The first qubit
            acts as the "b" register, while the remainder encodes the y-register.
        anc (list[Qubit]): Ancillary workspace for controlled arithmetic in
            step 4 (two's complement).

    Returns:
        None: The recursive QHT is applied in place to the given circuit.
    """
    if len(data) == 1:
        circuit.h(data[0])
        return

    _build_qht_rec(circuit, ctrl_anc, data[1:], anc)
    target_data = data[1:][::-1]

    N = 2 ** len(data)

    # Step 3: H on ancilla
    circuit.h(ctrl_anc)

    # Step 4: Controlled twos_complement

    ctrl_twos_complement(
        circuit,
        anc[0 : len(target_data) - 2],
        target_data + [ctrl_anc],
    )

    # Step 5: Apply UR
    _build_unitary_UR(
        circuit, ctrl=ctrl_anc, target_y=target_data, target_b=[data[0]], N=N
    )

    # Step 5.1: uncompute Controlled twos_complement

    ctrl_twos_complement(
        circuit,
        anc[0 : len(target_data) - 2],
        target_data + [ctrl_anc],
    )

    # Step 5.2: deal with extra term when y=0 in half sum formula
    for i in range(0, len(target_data)):
        circuit.x(target_data[i])

    circuit.h(data[0])
    circuit.append(
        MCXGate(len([ctrl_anc] + target_data)),
        [ctrl_anc] + target_data + [data[0]],
    )
    circuit.h(data[0])
    for i in range(0, len(target_data)):
        circuit.x(target_data[i])

    # Step 6: H on ancilla
    circuit.h(ctrl_anc)

    # Step 7: CNOT(ancilla, last data)
    circuit.cx(data[0], ctrl_anc)

    # Step 8: H on last data
    circuit.h(data[0])
