import sys
from pathlib import Path

import numpy as np




ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from application.qht_lcu_preconditioner import (
    make_toeplitz,
    make_hartley_preconditioner,
    quantum_precondition_state,
)


# ============================================================
# STATE FIDELITY
# ============================================================

def state_fidelity(a, b):
    """
    Fidelity between two pure states:

        F = |<a|b>|^2

    Both vectors are normalized inside this function.
    """

    a = np.asarray(
        a,
        dtype=complex
    )

    b = np.asarray(
        b,
        dtype=complex
    )

    a = (
        a
        / np.linalg.norm(a)
    )

    b = (
        b
        / np.linalg.norm(b)
    )

    return (
        np.abs(
            np.vdot(a, b)
        ) ** 2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # APPLICATION PARAMETERS
    # ========================================================

    n_data = 3

    N = 2**n_data

    rho = 0.99

    print()
    print("========================================")
    print("STEP 3: FULL VERSION-A APPLICATION")
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
    # ORIGINAL SYSTEM MATRIX
    # ========================================================

    T = make_toeplitz(
        N=N,
        rho=rho
    )

    # ========================================================
    # CLASSICAL HARTLEY PRECONDITIONER INFORMATION
    # ========================================================

    (
        H,
        p,
        P_inv_sqrt,
    ) = make_hartley_preconditioner(
        T
    )

    # ========================================================
    # PRECONDITIONED MATRIX
    #
    # B = P^(-1/2) T P^(-1/2)
    # ========================================================

    B = (
        P_inv_sqrt
        @ T
        @ P_inv_sqrt
    )

    # ========================================================
    # INPUT VECTOR b
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
    # EXACT SOLUTION OF ORIGINAL SYSTEM
    #
    # T x = b
    # ========================================================

    x_exact = np.linalg.solve(
        T,
        b
    )

    # For state comparison we only care about direction,
    # so normalize the exact solution.
    x_exact_state = (
        x_exact
        / np.linalg.norm(
            x_exact
        )
    )

    # ========================================================
    # FIRST QUANTUM PRECONDITIONING
    #
    # |b>
    #
    # ->
    #
    # P^(-1/2)|b>
    #
    # using:
    #
    # QHT-LCU
    #   ->
    # D^(-1/2)
    #   ->
    # QHT-LCU
    # ========================================================

    (
        b_pre_quantum,
        success_b,
        _,
        _,
    ) = quantum_precondition_state(
        vector=b,
        p=p,
        n_data=n_data,
    )

    # ========================================================
    # VERSION-A CENTRAL LINEAR SOLVE
    # ========================================================
    #
    # We now solve:
    #
    # B y = b'
    #
    # where:
    #
    # b' is the quantum-produced
    # normalized preconditioned state.
    #
    # For Version A this solve remains classical.
    #
    # Because b' differs from the unnormalized
    # P^(-1/2)b only by a scalar, the direction
    # of the final solution state is unchanged.
    # ========================================================

    y = np.linalg.solve(
        B,
        b_pre_quantum
    )

    # Normalize y before treating it as another
    # quantum input state.
    y_state = (
        y
        / np.linalg.norm(y)
    )

    # ========================================================
    # SECOND QUANTUM PRECONDITIONING
    #
    # |y>
    #
    # ->
    #
    # P^(-1/2)|y>
    #
    # This recovers the solution-state direction.
    # ========================================================

    (
        x_quantum,
        success_y,
        _,
        _,
    ) = quantum_precondition_state(
        vector=y_state,
        p=p,
        n_data=n_data,
    )

    # ========================================================
    # FINAL FIDELITY
    # ========================================================

    final_fidelity = state_fidelity(
        x_exact_state,
        x_quantum
    )

    # ========================================================
    # ALSO CHECK THE FIRST PRECONDITIONING STEP
    # ========================================================

    b_pre_exact = (
        P_inv_sqrt
        @ b
    )

    b_pre_exact = (
        b_pre_exact
        / np.linalg.norm(
            b_pre_exact
        )
    )

    first_preconditioner_fidelity = (
        state_fidelity(
            b_pre_exact,
            b_pre_quantum
        )
    )

    # ========================================================
    # CONDITION NUMBERS
    # ========================================================

    kappa_T = np.linalg.cond(
        T
    )

    kappa_B = np.linalg.cond(
        B
    )

    improvement = (
        kappa_T
        / kappa_B
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("----------------------------------------")
    print("CONDITIONING")
    print("----------------------------------------")

    print(
        "condition(T) =",
        kappa_T
    )

    print(
        "condition(B) =",
        kappa_B
    )

    print(
        "conditioning improvement =",
        improvement
    )

    print()
    print("----------------------------------------")
    print("FIRST QHT PRECONDITIONING")
    print("----------------------------------------")

    print(
        "success probability =",
        success_b
    )

    print(
        "preconditioner fidelity =",
        first_preconditioner_fidelity
    )

    print()
    print("----------------------------------------")
    print("SECOND QHT PRECONDITIONING")
    print("----------------------------------------")

    print(
        "success probability =",
        success_y
    )

    print()
    print("----------------------------------------")
    print("FINAL SOLUTION VALIDATION")
    print("----------------------------------------")

    print(
        "final solution-state fidelity =",
        final_fidelity
    )

    print(
        "final solution-state infidelity =",
        1.0 - final_fidelity
    )


if __name__ == "__main__":
    main()