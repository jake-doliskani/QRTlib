import sys
from pathlib import Path

import numpy as np
from scipy.linalg import toeplitz

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector




ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from QHTGate import QHTGate



from test.test_utils import (
    hartley_matrix,
    extract_data_register_amplitudes,
)


# ============================================================
# 1. TOEPLITZ MATRIX
# ============================================================

def make_toeplitz(N, rho):
    """
    Construct:

        T[i,j] = rho^|i-j|
    """

    first_column = (
        rho ** np.arange(N)
    )

    T = toeplitz(
        first_column
    )

    return T


# ============================================================
# 2. CLASSICAL HARTLEY PRECONDITIONER
# ============================================================

def make_hartley_preconditioner(T):
    """
    Construct the classical reference:

        P^(-1/2)
        =
        H diag(1/sqrt(p)) H.
    """

    N = T.shape[0]

    # --------------------------------------------------------
    # Normalize YOUR classical Hartley matrix.
    # --------------------------------------------------------

    H = (
        hartley_matrix(N)
        / np.sqrt(N)
    )

    # --------------------------------------------------------
    # Transform T into Hartley basis.
    # --------------------------------------------------------

    T_H = (
        H
        @ T
        @ H
    )

    # --------------------------------------------------------
    # Spectral values.
    # --------------------------------------------------------

    p = np.diag(
        T_H
    ).copy()

    if np.any(p <= 0):
        raise ValueError(
            "Non-positive Hartley spectral value."
        )

    # --------------------------------------------------------
    # P^(-1/2)
    #
    # This is our exact classical reference.
    # --------------------------------------------------------

    P_inv_sqrt = (
        H
        @ np.diag(
            1.0 / np.sqrt(p)
        )
        @ H
    )

    return H, p, P_inv_sqrt


# ============================================================
# 3. VERSION-A DIAGONAL SPECTRAL OPERATION
# ============================================================

def apply_inverse_sqrt_spectrum(
    qc,
    data_qubits,
    flag_qubit,
    p,
):
    """
    Implement the diagonal operation

        diag(1/sqrt(p_k))

    using an additional flag qubit.

    This is the application-validation implementation.

    For each Hartley basis state |k>, we perform

        |k>|0>
        ->
        |k>[
             sqrt(1-g_k^2)|0>
             +
             g_k|1>
           ]

    with

        g_k = C/sqrt(p_k).

    Therefore, conditional on flag=1,
    amplitudes are multiplied by 1/sqrt(p_k).
    """

    n_data = len(
        data_qubits
    )

    N = 2**n_data

    if len(p) != N:
        raise ValueError(
            "len(p) must equal 2**n_data."
        )

    # --------------------------------------------------------
    # Choose C so that:
    #
    # C / sqrt(p_k) <= 1
    #
    # for every k.
    # --------------------------------------------------------

    C = (
        0.99
        * np.sqrt(
            np.min(p)
        )
    )

    gains = (
        C
        / np.sqrt(p)
    )

    # ========================================================
    # LOOP THROUGH HARTLEY BASIS STATES
    # ========================================================

    for k in range(N):

        gain = np.clip(
            gains[k],
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # RY(theta)|0> gives:
        #
        # cos(theta/2)|0>
        # +
        # sin(theta/2)|1>
        #
        # We want:
        #
        # sin(theta/2) = gain
        #
        # therefore:
        #
        # theta = 2 asin(gain)
        # ----------------------------------------------------

        theta = (
            2.0
            * np.arcsin(gain)
        )

        # ----------------------------------------------------
        # Binary representation of k.
        #
        # q0 is the least-significant bit,
        # consistent with your Qiskit implementation.
        # ----------------------------------------------------

        bits = [
            (k >> q) & 1
            for q in range(n_data)
        ]

        # ----------------------------------------------------
        # Multi-controlled RY triggers when every
        # control is 1.
        #
        # If the desired bit of k is 0,
        # temporarily apply X.
        #
        # This converts the specific |k> state
        # into |11...1> for the control.
        # ----------------------------------------------------

        for qubit, bit in zip(
            data_qubits,
            bits
        ):

            if bit == 0:
                qc.x(qubit)

        # ----------------------------------------------------
        # Controlled RY on the flag.
        # ----------------------------------------------------

        controlled_ry = (
            RYGate(theta)
            .control(n_data)
        )

        qc.append(
            controlled_ry,
            data_qubits
            + [flag_qubit]
        )

        # ----------------------------------------------------
        # Undo temporary X gates.
        # ----------------------------------------------------

        for qubit, bit in zip(
            data_qubits,
            bits
        ):

            if bit == 0:
                qc.x(qubit)

    return C


# ============================================================
# 4. COMPLETE QUANTUM PRECONDITIONING PRIMITIVE
# ============================================================

def quantum_precondition_state(
    vector,
    p,
    n_data,
):
    """
    Apply:

        QHT-LCU
            ->
        diag(1/sqrt(p))
            ->
        QHT-LCU

    to a normalized input state.

    This implements:

        P^(-1/2)
        =
        H diag(1/sqrt(p)) H.

    Returns:

        quantum_data
        success_probability
        C
        circuit
    """

    N = 2**n_data

    # --------------------------------------------------------
    # Convert input to complex NumPy vector.
    # --------------------------------------------------------

    vector = np.asarray(
        vector,
        dtype=complex
    )

    if len(vector) != N:
        raise ValueError(
            "Input vector length must equal 2**n_data."
        )

    # Normalize the quantum input state.
    vector = (
        vector
        / np.linalg.norm(vector)
    )

    # ========================================================
    # YOUR QHT-LCU
    # ========================================================

    gate = QHTGate(
        n_data,
        type="LCU"
    )

    # Your QHT gate already knows its complete
    # data + work-ancilla width.
    qht_width = (
        gate.num_qubits
    )

    # Data are always the first n_data qubits.
    data_qubits = list(
        range(n_data)
    )

    # Complete QHT register.
    qht_qubits = list(
        range(qht_width)
    )

    # ========================================================
    # ADD ONE APPLICATION FLAG
    # ========================================================

    flag_qubit = (
        qht_width
    )

    total_qubits = (
        qht_width + 1
    )

    qc = QuantumCircuit(
        total_qubits
    )

    # ========================================================
    # FIRST QHT-LCU
    #
    # |psi> -> H|psi>
    # ========================================================

    qc.append(
        gate,
        qht_qubits
    )

    # ========================================================
    # SPECTRAL SCALING
    #
    # H|psi>
    #
    # ->
    #
    # D^(-1/2) H|psi>
    # ========================================================

    C = apply_inverse_sqrt_spectrum(
        qc=qc,
        data_qubits=data_qubits,
        flag_qubit=flag_qubit,
        p=p,
    )

    # ========================================================
    # SECOND QHT-LCU
    #
    # D^(-1/2)H|psi>
    #
    # ->
    #
    # H D^(-1/2) H|psi>
    #
    # =
    #
    # P^(-1/2)|psi>
    # ========================================================

    qc.append(
        gate,
        qht_qubits
    )

    # ========================================================
    # POSTSELECTION TRICK USING YOUR EXISTING UTILITY
    # ========================================================
    #
    # Successful branch currently has:
    #
    # flag = 1
    #
    # But your extract_data_register_amplitudes()
    # keeps states only when every NON-DATA qubit is 0.
    #
    # So:
    #
    # success 1 -> 0
    #
    # failure 0 -> 1
    #
    # After this X gate, your extraction function
    # automatically extracts only the successful branch.
    # ========================================================

    qc.x(
        flag_qubit
    )

    # ========================================================
    # CONSTRUCT INITIAL FULL STATE
    # ========================================================
    #
    # Data qubits occupy the lowest positions:
    #
    # q0, q1, ..., q(n_data-1)
    #
    # and every QHT work qubit + flag starts at zero.
    #
    # Therefore the first N amplitudes correspond to
    # our input vector.
    # ========================================================

    initial_full_state = np.zeros(
        2**total_qubits,
        dtype=complex
    )

    initial_full_state[:N] = (
        vector
    )

    sv = Statevector(
        initial_full_state
    )

    # ========================================================
    # IDEAL EVOLUTION
    # ========================================================

    evolved = sv.evolve(
        qc
    )

    # ========================================================
    # USE YOUR EXISTING EXTRACTION FUNCTION
    # ========================================================
    #
    # This returns data-register amplitudes only when:
    #
    # QHT ancillas = 0
    # flag          = 0
    #
    # Because of the X above, flag=0 means
    # successful spectral scaling.
    # ========================================================

    quantum_data = (
        extract_data_register_amplitudes(
            evolved.data,
            data_indices=list(
                range(n_data)
            )
        )
    )

    # ========================================================
    # SUCCESS PROBABILITY
    # ========================================================
    #
    # The extracted branch is not normalized.
    #
    # Its squared norm is the probability of
    # obtaining the successful flag result.
    # ========================================================

    success_probability = (
        np.linalg.norm(
            quantum_data
        ) ** 2
    )

    if success_probability < 1e-15:
        raise RuntimeError(
            "Successful branch has zero probability."
        )

    # ========================================================
    # NORMALIZE SUCCESSFUL DATA STATE
    # ========================================================

    quantum_data = (
        quantum_data
        / np.linalg.norm(
            quantum_data
        )
    )

    return (
        quantum_data,
        success_probability,
        C,
        qc,
    )


# ============================================================
# 5. MAIN APPLICATION VALIDATION
# ============================================================

def main():

    # ========================================================
    # START SMALL
    # ========================================================
    #
    # n_data = 3
    #
    # means:
    #
    # N = 8 dimensional vector
    #
    # Your QHT-LCU uses its own required work qubits.
    # ========================================================

    n_data = 3

    N = 2**n_data

    rho = 0.99

    print()
    print("========================================")
    print("STEP 2: QHT-LCU PRECONDITIONER")
    print("========================================")

    print(
        "n_data =",
        n_data
    )

    print(
        "N =",
        N
    )

    print(
        "rho =",
        rho
    )

    # ========================================================
    # BUILD TOEPLITZ SYSTEM
    # ========================================================

    T = make_toeplitz(
        N=N,
        rho=rho
    )

    # ========================================================
    # CLASSICAL REFERENCE
    # ========================================================

    H, p, P_inv_sqrt = (
        make_hartley_preconditioner(T)
    )

    # ========================================================
    # INPUT VECTOR
    # ========================================================

    rng = np.random.default_rng(
        7
    )

    b = rng.normal(
        size=N
    )

    b = (
        b
        / np.linalg.norm(b)
    )

    # ========================================================
    # EXACT CLASSICAL TARGET
    #
    # target =
    #
    # P^(-1/2) b
    #
    # normalized as a quantum state.
    # ========================================================

    target = (
        P_inv_sqrt
        @ b
    )

    target = (
        target
        / np.linalg.norm(
            target
        )
    )

    # ========================================================
    # QHT-LCU APPLICATION
    # ========================================================

    (
        quantum_output,
        success_probability,
        C,
        qc,
    ) = quantum_precondition_state(
        vector=b,
        p=p,
        n_data=n_data,
    )

    # ========================================================
    # FIDELITY
    # ========================================================

    overlap = np.vdot(
        target,
        quantum_output
    )

    fidelity = (
        np.abs(overlap) ** 2
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("----------------------------------------")
    print("REGISTER INFORMATION")
    print("----------------------------------------")

    print(
        "QHT-LCU gate qubits =",
        QHTGate(
            n_data,
            type="LCU"
        ).num_qubits
    )

    print(
        "complete application qubits =",
        qc.num_qubits
    )

    print()
    print("----------------------------------------")
    print("SPECTRAL OPERATION")
    print("----------------------------------------")

    print(
        "minimum p_k =",
        np.min(p)
    )

    print(
        "spectral scaling constant C =",
        C
    )

    print(
        "success probability =",
        success_probability
    )

    print()
    print("----------------------------------------")
    print("APPLICATION VALIDATION")
    print("----------------------------------------")

    print(
        "application fidelity =",
        fidelity
    )

    print(
        "application infidelity =",
        1.0 - fidelity
    )


if __name__ == "__main__":
    main()