from typing import Sequence, Optional, Union
from qiskit.circuit import Gate, QuantumCircuit
import numpy as np


class QSTGate(Gate):
    r"""Quantum Sine Transform (QST) Gate with two modes:

    1) Standalone gate (internal definition):
         QSTGate(num_qubits=..., type="I"|"II"|"IV")
       - num_qubits = data-qubit count
       - adds 1 control + ancillas per rules

    2) Embed into an existing circuit (operate in-place):
         QSTGate(circuit=qc, target_qubits=[...], ancilla_qubits=[...], type="I"|"II"|"IV")
       - `target_qubits` = [control, data0, data1, ...] in `qc`'s indexing
       - `ancilla_qubits` = indices in `qc`
       - `_define()` calls the type-specific builder directly on `qc`
    """

    def __init__(
        self,
        arg: Optional[Union[int, QuantumCircuit]] = None,  # num_qubits OR circuit
        type: str = "I",
        *,
        target_qubits: Optional[Sequence[int]] = None,  # control + data
        ancilla_qubits: Optional[Sequence[int]] = None,
    ):
        if type not in ("I", "II", "IV"):
            raise ValueError(f"Unsupported QST type: {type}")
        self.type = type

        # Detect mode
        if isinstance(arg, QuantumCircuit):
            circuit = arg
            num_qubits = None
        else:
            circuit = None
            num_qubits = arg

        self._external_circuit = circuit
        self._user_target_qubits = (
            list(target_qubits) if target_qubits is not None else None
        )
        self._user_ancilla_qubits = (
            list(ancilla_qubits) if ancilla_qubits is not None else None
        )

        if circuit is not None:
            # Embedding mode
            if self._user_target_qubits is None or self._user_ancilla_qubits is None:
                raise ValueError(
                    "In embedding mode, provide both target_qubits and ancilla_qubits."
                )

            tgt_len = len(self._user_target_qubits)  # = control (1) + data
            if tgt_len < 2:
                raise ValueError(
                    "target_qubits must include 1 control + >=1 data (len >= 2)."
                )

            data_qubits = tgt_len - 1
            anc_len = len(self._user_ancilla_qubits)

            # Ancilla rules (QST)
            if type == "I" and anc_len != max(0, data_qubits - 2):
                raise ValueError(
                    f"Type I requires ancillas = data-2 ({data_qubits-2}), got {anc_len}."
                )
            if type == "II" and anc_len != max(0, data_qubits - 1):
                raise ValueError(
                    f"Type II requires ancillas = data-1 ({data_qubits-1}), got {anc_len}."
                )
            if type == "IV" and anc_len != 0:
                raise ValueError("Type IV requires no ancillas.")

            # Optional bounds check against provided circuit
            nq = circuit.num_qubits
            for i in (*self._user_target_qubits, *self._user_ancilla_qubits):
                if i < 0 or i >= nq:
                    raise ValueError(
                        f"Qubit index {i} out of range for circuit with {nq} qubits."
                    )

            self.data_qubits = data_qubits
            self.ancilla_qubits = anc_len
            self.control_qubits = 1

            total_qubits = data_qubits + anc_len + self.control_qubits
            super().__init__(
                name=f"qst-{type.lower()}", num_qubits=total_qubits, params=[]
            )

            # Apply directly on user circuit
            self._define()

        else:
            # Standalone mode
            if num_qubits is None:
                raise ValueError(
                    "Provide num_qubits when not embedding into a circuit."
                )
            data_qubits = int(num_qubits)
            if data_qubits < 1:
                raise ValueError("num_qubits (data qubits) must be >= 1")

            if type == "I":
                anc_ct = max(0, data_qubits - 2)
            elif type == "II":
                anc_ct = max(0, data_qubits - 1)
            else:  # IV
                anc_ct = 0

            self.data_qubits = data_qubits
            self.ancilla_qubits = anc_ct
            self.control_qubits = 1
            total_qubits = data_qubits + anc_ct + self.control_qubits

            super().__init__(
                name=f"qst-{type.lower()}", num_qubits=total_qubits, params=[]
            )

            # Build internal definition
            self._define()

    def __array__(self, dtype=complex, copy=None):
        """Return a numpy array representing the QST (using test_utils tX_sin_mat)."""
        if copy is False:
            raise ValueError(
                "unable to avoid copy while creating an array as requested"
            )
        from test.test_utils import t1_sin_mat, t2_sin_mat, t4_sin_mat

        N = 2**self.data_qubits
        mat = np.zeros((N, N), dtype=np.float64)
        if self.type == "I":

            for i in range(N):
                for j in range(N):
                    mat[i, j] = np.sin(np.pi * i * j / N)

        elif self.type == "II":

            for i in range(N):
                for j in range(N):
                    mat[i, j] = np.sin(np.pi * (i + 1) * (j + 1 / 2) / N)
                    if i == N - 1:
                        mat[i, j] *= 1 / np.sqrt(2)
        else:  # "IV"

            for i in range(N):
                for j in range(N):
                    mat[i, j] = np.cos(np.pi * (i) * (j + 1 / 2) / N)
                    if i == 0:
                        mat[i, j] *= 1 / np.sqrt(2)

        return np.array(mat, dtype=dtype)

    def _define(self):
        """Decompose QST. If an external circuit was provided, operate on it directly."""
        if self._external_circuit is not None:
            # Use the caller's circuit and indices as-is
            qc = self._external_circuit
            target_idxs = self._user_target_qubits
            ancilla_idxs = self._user_ancilla_qubits
        else:
            # Create our own definition circuit (standalone gate)
            qc = QuantumCircuit(self.num_qubits, name=self.name)
            total_targets = self.data_qubits + self.control_qubits
            target_idxs = list(range(total_targets))
            ancilla_idxs = list(range(total_targets, self.num_qubits))

        # Dispatch to the correct builder, operating on `qc`
        if self.type == "I":
            from qst_type_I_fast import qst_type_I

            qst_type_I(qc, ancilla_idxs, target_idxs)
        elif self.type == "II":
            from qct_qst_type_II import qst_type_II

            qst_type_II(qc, ancilla_idxs, target_idxs)
        else:  # "IV"
            from qct_qst_type_IV import qst_type_IV

            qst_type_IV(qc, target_idxs)

        # Only set self.definition when we're in standalone mode
        self.definition = qc
