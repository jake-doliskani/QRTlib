# test_type_I_by_list.py
import unittest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# builders (the “builder” mentioned below are these callables)
from qst_type_I_fast import qst_type_I  # sine (QST)
from qct_qst_type_I import qct_type_I  # cosine (QCT)

from qct_qst_type_II import qst_type_II, qct_type_II  # Type II transform
from qct_qst_type_IV import qst_type_IV, qct_type_IV  # Type IV transform

from .test_utils import *

# ---------- TEST LISTS ----------
# Each tuple: (bitstring, data_index, target_count)
# target_count = (data qubits) + 1 control; ancillas are any remaining higher-index qubits.

TESTS_T1_QCT = [
    # ("001", 1, 2),
]

TESTS_T1_QST_FAST = [
    # ("00101", 5, 4),  # total=5, target_count=4 -> data=[0,1,2], control=3, anc=[4]
    # ("00001", 1, 4),
]


TESTS_T2 = [
    # ("000001", 1, 4),
    # ("000111", 7, 4),
]


TESTS_T4 = [
    # ("00001", 1, 5),
    # ("00011", 3, 5),
]


append_auto_tests_T1_QCT(TESTS_T1_QCT)
append_auto_tests_T1_QST_FAST(TESTS_T1_QST_FAST)
append_auto_tests_T2(TESTS_T2)
append_auto_tests_T4(TESTS_T4)


class TestTypeIByList(unittest.TestCase):
    RTOL = 1e-9
    ATOL = 1e-9

    def _run_case(self, builder, bitstr: str, j: int, target_count: int, M, N):
        total = len(bitstr)
        # Indices (Qiskit little-endian): we allocate qubits 0..total-1
        # target qubits are 0..target_count-1 (lowest indices)
        # - data qubits: 0..target_count-2
        # - control qubit: target_count-1
        # ancillas: target_count..total-1
        data_qubits = list(range(0, target_count - 1))
        control_idx = target_count - 1
        anc_idxs = list(range(target_count, total))

        # Build the circuit and append gates with the provided builder
        qc = QuantumCircuit(total)

        # if target_count == len(bitstr):
        #     builder(qc, data_qubits + [control_idx])
        # else:
        #     builder(qc, anc_idxs, data_qubits + [control_idx])

        # try catch way to call builder
        # try builder(qc, data_qubits + [control_idx]) first
        # any errors call builder(qc, anc_idxs, data_qubits + [control_idx])
        try:

            builder(qc, anc_idxs, data_qubits + [control_idx])
        except Exception:
            builder(qc, data_qubits + [control_idx])

        # Initialize & evolve
        init = Statevector.from_label(bitstr)  # MSB..LSB string
        out = init.evolve(qc)

        # We read the output on the branch where ancillas and control equal the input bits.
        terms, _ = statevector_to_list(out)

        got = quantum_amps(terms)

        # Classical expected vector (size N = 2^(#data))
        # N = 1 << (target_count - 1)
        expected = classical_amps(M, j, N)
        # print(got)
        # print(expected)
        assert_close_vec(
            got,
            expected,
            ctx=f"init={bitstr}, j={j}, target_count={target_count}, N={N}",
        )

    def test_qct_type_I(self):
        for bitstr, j, tgt_cnt in TESTS_T1_QCT:
            with self.subTest(bitstr=bitstr):
                N = 1 << (tgt_cnt - 1)
                M = t1_cos_mat(N)
                self._run_case(qct_type_I, bitstr, j, tgt_cnt, M, N)

    def test_qct_type_II(self):
        for bitstr, j, tgt_cnt in TESTS_T2:
            with self.subTest(bitstr=bitstr):
                N = 1 << (tgt_cnt - 1)
                M = t2_cos_mat(N)
                self._run_case(qct_type_II, bitstr, j, tgt_cnt, M, N)

    def test_qct_type_IV(self):
        for bitstr, j, tgt_cnt in TESTS_T4:
            with self.subTest(bitstr=bitstr):
                N = 1 << (tgt_cnt - 1)
                M = t4_cos_mat(N)
                self._run_case(qct_type_IV, bitstr, j, tgt_cnt, M, N)

    def test_qst_type_I(self):
        for bitstr, j, tgt_cnt in TESTS_T1_QST_FAST:
            with self.subTest(bitstr=bitstr):
                N = 1 << (tgt_cnt - 1)
                M = t1_sin_mat(N)
                self._run_case(qst_type_I, bitstr, j, tgt_cnt, M, N)

    def test_qst_type_II(self):
        for bitstr, j, tgt_cnt in TESTS_T2:
            with self.subTest(bitstr=bitstr):
                N = 1 << (tgt_cnt - 1)
                M = t2_sin_mat(N)
                self._run_case(qst_type_II, bitstr, j, tgt_cnt, M, N)

    def test_qst_type_IV(self):
        for bitstr, j, tgt_cnt in TESTS_T4:
            with self.subTest(bitstr=bitstr):
                N = 1 << (tgt_cnt - 1)
                M = t4_sin_mat(N)

                self._run_case(qst_type_IV, bitstr, j, tgt_cnt, M, N)

    # from .test_utils import assert_close_vec

    def test_qht_LCU(self):
        for n in range(1, 6):
            for bits in [f"{i:0{n}b}" for i in range(2**n)]:
                results = compare_qht_vs_classical_LCU(n, bits)
                # print("Quantum:", results["quantum"])
                # print("Classical:", results["classical"])
                assert_close_vec(
                    results["quantum"],
                    results["classical"],
                    ctx=f"init={bits}, j={n}",
                )

    def test_qht_recursive(self):

        for n in range(1, 6):
            for bits in [f"{i:0{n}b}" for i in range(2**n)]:
                results = compare_qht_vs_classical_recursive(n, bits)
                # print("Quantum:", results["quantum"])
                # print("Classical:", results["classical"])
                assert_close_vec(
                    results["quantum"],
                    results["classical"],
                    ctx=f"init={bits}, j={n}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
