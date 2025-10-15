# test_type_I_by_list.py
import numpy as np
from qiskit.quantum_info import Statevector


# ---------- classical refs ----------
# ---------- Type I transform ----------
def t1_sin_mat(N):
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = np.sin(np.pi * i * j / N)
    return M


def t1_cos_mat(N):
    M = np.zeros((N + 1, N + 1), dtype=np.float64)

    for i in range(N + 1):
        for j in range(N + 1):
            M[i, j] = np.cos(np.pi * i * j / N)

            if i == 0 or i == N:
                M[i, j] *= 1 / np.sqrt(2)
            if j == 0 or j == N:
                M[i, j] *= 1 / np.sqrt(2)

    return M


# ---------- Type II transform ----------
def t2_sin_mat(N):

    M = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            M[i, j] = np.sin(np.pi * (i + 1) * (j + 1 / 2) / N)
            if i == N - 1:
                M[i, j] *= 1 / np.sqrt(2)

    return M


def t2_cos_mat(N):

    M = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            M[i, j] = np.cos(np.pi * (i) * (j + 1 / 2) / N)
            if i == 0:
                M[i, j] *= 1 / np.sqrt(2)

    return M


# ---------- Type IV transform ----------
def t4_sin_mat(N):

    M = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            M[i, j] = np.sin(np.pi * (i + 1 / 2) * (j + 1 / 2) / N)

    return M


def t4_cos_mat(N):

    M = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            M[i, j] = np.cos(np.pi * (i + 1 / 2) * (j + 1 / 2) / N)

    return M


# === Auto-append Type-I QCT cases ===
# Layout per case:
#   total_len = (ancillas = d) + (target = 1 control + d data) = 2*d + 1
#   bitstring (MSB..LSB) = '0'*d + '0' + data_bits_msb_to_lsb
#   target_count = d + 1


def append_auto_tests_T1_QCT(TESTS_T1_QCT):
    for d in range(2, 8):  # data-bit width: 1..6
        target_count = d + 1
        for j in range(1 << d):  # all data values
            # data bits (MSB->LSB in the label, which corresponds to qubits [d-1..0])
            data_bits = "".join(
                "1" if (j >> k) & 1 else "0" for k in range(d - 1, -1, -1)
            )
            bitstr = ("0" * d) + "0" + data_bits  # ancillas + control(0) + data
            TESTS_T1_QCT.append((bitstr, j, target_count))


# === Auto-append Type-I QST cases ===
# Layout per case:
#   anc = d - 2  (must be >= 0, so d >= 2)
#   target_count = d + 1
#   bitstring (MSB..LSB) = '0' * anc + '0' + data_bits_msb_to_lsb
#   total_len = anc + 1 + d = 2*d - 1


def append_auto_tests_T1_QST_FAST(TESTS_T1_QST):
    for d in range(1, 8):  # data-bit width: 2..6
        anc = d - 2
        target_count = d + 1
        for j in range(1, 1 << d):  # skip j == 0 (all data_bits = 0)
            # data bits MSB->LSB so they map to little-endian data qubits [d-1..0]
            data_bits = "".join(
                "1" if (j >> k) & 1 else "0" for k in range(d - 1, -1, -1)
            )
            bitstr = ("0" * anc) + "0" + data_bits  # ancillas + control(0) + data
            TESTS_T1_QST.append((bitstr, j, target_count))


# === Auto-append Type-II QCT cases ===
# Layout per case:
#   d = number of data bits
#   anc = d - 1  (>= 0 ⇒ d >= 1)
#   target_count = d + 1  (1 control + d data)
#   bitstring (MSB..LSB) = '0' * anc + '0' + data_bits_msb_to_lsb
#   total_len = anc + 1 + d = 2*d


def append_auto_tests_T2(T):
    for d in range(2, 8):  # data-bit width: 1..6
        anc = d - 1
        target_count = d + 1
        for j in range(1 << d):  # include j == 0 unless you want to skip it
            # data bits MSB->LSB so they map to little-endian data qubits [d-1..0]
            data_bits = "".join(
                "1" if (j >> k) & 1 else "0" for k in range(d - 1, -1, -1)
            )
            bitstr = ("0" * anc) + "0" + data_bits  # ancillas + control(0) + data
            T.append((bitstr, j, target_count))


# === Auto-append Type-IV QCT cases ===
# Layout per case:
#   d = number of data bits (>=1)
#   anc = 0
#   target_count = d + 1
#   bitstring (MSB..LSB) = '0' + data_bits_msb_to_lsb
#   total_len = 1 + d


def append_auto_tests_T4(T):
    for d in range(1, 7):  # data-bit width: 1..6
        target_count = d + 1
        for j in range(1 << d):  # include j=0 unless you want to skip
            data_bits = "".join(
                "1" if (j >> k) & 1 else "0" for k in range(d - 1, -1, -1)
            )
            bitstr = "0" + data_bits  # control(0) + data
            T.append((bitstr, j, target_count))


def classical_amps(M, j, N):
    x = np.zeros(M.shape[1])
    x[j] = 1
    result = np.matmul(M, x)
    normalized_result = result / np.linalg.norm(result)
    return normalized_result


# ---------- helpers ----------
def bit_at(bitstr: str, q: int) -> int:
    """Return bit value at qubit index q given an MSB..LSB bitstring."""
    return 1 if bitstr[len(bitstr) - 1 - q] == "1" else 0


def statevector_to_list(sv: Statevector, threshold: float = 1e-10):
    """
    EXACT version of your statevector_to_dirac loop, but returns a list of
    (amp: complex, bitstr: str) with bitstr in MSB..LSB. No string formatting.
    Set threshold=None to include all amplitudes.
    """
    amps = sv.data
    n_qubits = int(np.log2(len(amps)))
    terms = []
    for idx, amp in enumerate(amps):
        if threshold is None or abs(amp) > threshold:
            bitstr = format(idx, f"0{n_qubits}b")  # MSB..LSB, matches Qiskit labels
            terms.append((amp, bitstr))
    # print(terms)
    return terms, n_qubits


def quantum_amps(terms):
    """
    Return (amps_array, bitstr_array). Amps are dtype=complex, untouched.
    """

    amps = np.array([complex(a) for a, _ in terms], dtype=complex)

    return amps


def assert_close_vec(
    got,
    expected,
    ctx: str,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-9,
    threshold: float = 1e-10,
    max_show: int = 8,
):
    # Convert to flat arrays
    g = np.asarray(got, dtype=complex).ravel()
    e = np.asarray(expected, dtype=complex).ravel()

    # Filter independently — remove entries smaller than threshold
    g_filtered = g[np.abs(g) >= threshold]
    e_filtered = e[np.abs(e) >= threshold]

    # print(f"Filtered got list (kept {g_filtered.size}/{g.size}):")
    # print(g_filtered)
    # print(f"\nFiltered expected list (kept {e_filtered.size}/{e.size}):")
    # print(e_filtered)

    # Length check after filtering
    if g_filtered.size != e_filtered.size:
        raise AssertionError(
            f"\nMismatch in {ctx}: filtered lengths differ "
            f"(got={g_filtered.size}, expected={e_filtered.size})"
        )

    # Compare real parts
    try:
        np.testing.assert_allclose(
            g_filtered.real, e_filtered.real, rtol=rtol, atol=atol
        )
        # For real transforms, imag parts of `got` should be ~0
        np.testing.assert_allclose(g_filtered.imag, 0.0, rtol=rtol, atol=atol)
    except AssertionError:
        # Show top differences
        real_err = np.abs(g_filtered.real - e_filtered.real)
        order = np.argsort(-real_err)
        lines = []
        for pos in order[:max_show]:
            lines.append(
                f"[{pos:02d}] got=({g_filtered[pos].real:.6f}{g_filtered[pos].imag:+.6f}j)  "
                f"exp=({e_filtered[pos].real:.6f}{0.0:+.6f}j)  "
                f"|Δ_real|={real_err[pos]:.3e}  |imag(got)|={abs(g_filtered[pos].imag):.3e}"
            )
        raise AssertionError(f"\nMismatch in {ctx}\n" + "\n".join(lines))


def debug_filter_and_print(got, expected, threshold=1e-10):

    g = np.asarray(got, dtype=complex).ravel()
    e = np.asarray(expected, dtype=complex).ravel()

    g_filtered = g[np.abs(g) >= threshold]
    e_filtered = e[np.abs(e) >= threshold]

    print(f"Filtered got list (kept {g_filtered.size}/{g.size}):")
    print(g_filtered)

    print(f"\nFiltered expected list (kept {e_filtered.size}/{e.size}):")
    print(e_filtered)
    return g_filtered, e_filtered


# =================== Hartley ====================

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from QHTGate import QHTGate
import numpy as np


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


def hartley_matrix(N):
    """
    discrete hartley transform
    """
    H = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            H[k, n] = np.cos(2 * np.pi * k * n / N) + np.sin(2 * np.pi * k * n / N)
    return H


def classical_hartley_output(bitstring):
    """
    classical hartley transform
    """
    N = 2 ** len(bitstring)
    vec = np.zeros(N)
    index = int(bitstring, 2)
    vec[index] = 1.0
    H = hartley_matrix(N)
    out = H @ vec
    normed = out / np.linalg.norm(out)
    return normed


def compare_qht_vs_classical_LCU(n, bitstring):

    if n == 1:
        qc = QuantumCircuit(1)
        gate = QHTGate(1, type="LCU")
        qc.append(gate, [0])
    else:

        gate = QHTGate(n, type="LCU")

            
        qc = QuantumCircuit(2*n)
        qc.append(gate, list(range(2*n)))
        
        
        
    full_bits = "0" * n + bitstring[::]

    index = int(full_bits, 2)
    sv = Statevector.from_int(index, dims=2**qc.num_qubits)

    evolved = sv.evolve(qc)

    quantum_out_data = extract_data_register_amplitudes(
                                    evolved.data, data_indices=list(range(n))
                                                             )

    classical_out = classical_hartley_output(bitstring)

    return {
        "quantum": np.round(quantum_out_data.real, 4),
        "classical": np.round(classical_out, 4),
    }


def compare_qht_vs_classical_recursive(n, bitstring):

    if n == 1:
        qc = QuantumCircuit(1)
        gate = QHTGate(1, type="REC")
        qc.append(gate, [0])
    elif n == 2:
        qc = QuantumCircuit(3)
        gate = QHTGate(2, type="REC")
        qc.append(gate, [0, 1, 2])
        
    else:

        gate = QHTGate(n, type="REC")

        qc = QuantumCircuit(2 * (n-1))
        qc.append(gate, list(range(2 * (n-1))))
    

    full_bits = "0" * (n - 2)  + bitstring  
    index = int(full_bits, 2)
    sv = Statevector.from_int(index, dims=2**qc.num_qubits)

    # Evolve
    evolved = sv.evolve(qc)

    # Extract amplitudes of the data register
    quantum_out_data = extract_data_register_amplitudes(
        evolved.data, data_indices=list(range(n))
    )

    # Classical Hartley output
    classical_out = classical_hartley_output(bitstring)

    return {
        "quantum": np.round(quantum_out_data.real, 4),
        "classical": np.round(classical_out, 4),
    }
