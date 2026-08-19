import sys
from pathlib import Path

import numpy as np
from scipy.linalg import toeplitz



ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from test.test_utils import hartley_matrix


# ============================================================
# 1. BUILD THE TOEPLITZ MATRIX
# ============================================================

def make_toeplitz(N, rho):
    """
    Construct the real symmetric Toeplitz matrix

        T[i,j] = rho^|i-j|

    Example:

        T =
        [ 1       rho     rho^2   ... ]
        [ rho     1       rho     ... ]
        [ rho^2   rho     1       ... ]
        [ ...                     ... ]

    For 0 < rho < 1, this matrix is positive definite.

    As rho becomes closer to 1, the matrix becomes
    increasingly ill-conditioned.
    """

    # First column:
    #
    # [1, rho, rho^2, ..., rho^(N-1)]
    first_column = rho ** np.arange(N)

    # scipy constructs the full Toeplitz matrix.
    T = toeplitz(first_column)

    return T


# ============================================================
# 2. BUILD THE HARTLEY PRECONDITIONER
# ============================================================

def make_hartley_preconditioner(T):
    """
    Construct the Hartley preconditioner.

    Starting from T:

        T_H = H T H

    Extract:

        p_k = (T_H)_kk

    Define:

        P = H diag(p) H

    and therefore

        P^(-1/2)
        =
        H diag(1/sqrt(p)) H.
    """

    N = T.shape[0]

    # --------------------------------------------------------
    # YOUR hartley_matrix(N) returns:
    #
    # cos(...) + sin(...)
    #
    # without the 1/sqrt(N) normalization.
    #
    # Your unit tests normalize transformed vectors afterward.
    #
    # Here we need the actual orthogonal Hartley matrix:
    #
    # H_normalized = H / sqrt(N)
    # --------------------------------------------------------

    H = hartley_matrix(N) / np.sqrt(N)

    # --------------------------------------------------------
    # Express T in the Hartley basis:
    #
    # T_H = H T H
    # --------------------------------------------------------

    T_H = H @ T @ H

    # --------------------------------------------------------
    # Keep the diagonal elements:
    #
    # p_k = (H T H)_kk
    # --------------------------------------------------------

    p = np.diag(T_H).copy()

    # We need positive p_k because later we calculate
    # 1/sqrt(p_k).
    if np.any(p <= 0):
        raise ValueError(
            "Hartley preconditioner contains "
            "non-positive spectral values."
        )

    # --------------------------------------------------------
    # Construct:
    #
    # P = H diag(p) H
    # --------------------------------------------------------

    P = (
        H
        @ np.diag(p)
        @ H
    )

    # --------------------------------------------------------
    # The key operation for our quantum application:
    #
    # P^(-1/2)
    # =
    # H diag(1/sqrt(p_k)) H
    #
    # Later:
    #
    # H -> YOUR QHT-LCU
    # --------------------------------------------------------

    P_inv_sqrt = (
        H
        @ np.diag(
            1.0 / np.sqrt(p)
        )
        @ H
    )

    return H, p, P, P_inv_sqrt


# ============================================================
# 3. MAIN EXPERIMENT
# ============================================================

def main():

    # ========================================================
    # PROBLEM PARAMETERS
    # ========================================================

    N = 32
    rho = 0.99

    print()
    print("========================================")
    print("STEP 1: CLASSICAL HARTLEY PRECONDITIONER")
    print("========================================")

    print("N =", N)
    print("rho =", rho)

    # ========================================================
    # ORIGINAL MATRIX
    # ========================================================

    T = make_toeplitz(
        N=N,
        rho=rho
    )

    # ========================================================
    # CONSTRUCT PRECONDITIONER
    # ========================================================

    H, p, P, P_inv_sqrt = (
        make_hartley_preconditioner(T)
    )

    # ========================================================
    # CHECK 1:
    #
    # Hartley transform should be self-inverse:
    #
    # H H = I
    # ========================================================

    H_error = np.linalg.norm(
        H @ H - np.eye(N)
    )

    print()
    print(
        "||H H - I|| =",
        H_error
    )

    # ========================================================
    # CHECK 2:
    #
    # H should diagonalize P.
    #
    # H P H = diag(p)
    # ========================================================

    P_H = H @ P @ H

    off_diagonal = (
        P_H
        - np.diag(
            np.diag(P_H)
        )
    )

    diagonalization_error = np.linalg.norm(
        off_diagonal
    )

    print(
        "Hartley diagonalization error =",
        diagonalization_error
    )

    print(
        "minimum p_k =",
        np.min(p)
    )

    # ========================================================
    # CONSTRUCT PRECONDITIONED MATRIX
    #
    # B = P^(-1/2) T P^(-1/2)
    # ========================================================

    B = (
        P_inv_sqrt
        @ T
        @ P_inv_sqrt
    )

    # ========================================================
    # CONDITION NUMBERS
    # ========================================================

    kappa_T = np.linalg.cond(T)

    kappa_B = np.linalg.cond(B)

    improvement = (
        kappa_T / kappa_B
    )

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
        "improvement factor =",
        improvement
    )

    # ========================================================
    # VERIFY THAT PRECONDITIONING DOES NOT CHANGE
    # THE UNDERLYING SOLUTION
    # ========================================================

    # Create a reproducible test vector b.
    rng = np.random.default_rng(7)

    b = rng.normal(
        size=N
    )

    # Normalize b.
    b = (
        b
        / np.linalg.norm(b)
    )

    # --------------------------------------------------------
    # ORIGINAL SYSTEM:
    #
    # T x = b
    # --------------------------------------------------------

    x_direct = np.linalg.solve(
        T,
        b
    )

    # --------------------------------------------------------
    # PRECONDITIONED RHS:
    #
    # b' = P^(-1/2) b
    # --------------------------------------------------------

    b_pre = (
        P_inv_sqrt
        @ b
    )

    # --------------------------------------------------------
    # SOLVE:
    #
    # B y = b'
    # --------------------------------------------------------

    y = np.linalg.solve(
        B,
        b_pre
    )

    # --------------------------------------------------------
    # RECOVER:
    #
    # x = P^(-1/2) y
    # --------------------------------------------------------

    x_pre = (
        P_inv_sqrt
        @ y
    )

    # --------------------------------------------------------
    # Compare with direct solution.
    # --------------------------------------------------------

    relative_error = (
        np.linalg.norm(
            x_direct - x_pre
        )
        /
        np.linalg.norm(
            x_direct
        )
    )

    print()
    print("----------------------------------------")
    print("SOLUTION CHECK")
    print("----------------------------------------")

    print(
        "relative solution error =",
        relative_error
    )


if __name__ == "__main__":
    main()