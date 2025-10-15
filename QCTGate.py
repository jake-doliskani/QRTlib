from typing import Optional, Sequence, Union
from qiskit.circuit import Gate, QuantumCircuit
import numpy as np


class QCTGate(Gate):
    r"""Quantum Cosine Transform (QCT) Gate.

    Modes:
      1) Standalone gate:
         QCTGate(num_qubits=..., type="I"|"II"|"IV")
         - num_qubits = data-qubit count
         - adds 1 control + ancillas per rules
         - builds its own definition

      2) Embed into an existing circuit:
         QCTGate(circuit, type="I"|"II"|"IV",
                 target_qubits=[...], ancilla_qubits=[...])
         - `target_qubits` = [control, data0, data1, ...] indices in the circuit
         - `ancilla_qubits` = indices in the circuit
         - `_define` applies the QCT decomposition directly on the provided circuit
    """

    def __init__(
        self,
        arg: Optional[Union[int, QuantumCircuit]] = None,  # num_qubits OR circuit
        type: str = "I",
        *,
        target_qubits: Optional[Sequence[int]] = None,
        ancilla_qubits: Optional[Sequence[int]] = None,
    ):
        if type not in ("I", "II", "IV"):
            raise ValueError(f"Unsupported QCT type: {type}")
        self.type = type

        # detect mode
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

            tgt_len = len(self._user_target_qubits)  # = control + data
            if tgt_len < 2:
                raise ValueError("Need at least 1 control + 1 data qubit.")

            data_qubits = tgt_len - 1
            anc_len = len(self._user_ancilla_qubits)

            # Ancilla rules
            if type == "I" and anc_len != data_qubits:
                raise ValueError(
                    f"Type I requires ancillas = data ({data_qubits}), got {anc_len}."
                )
            if type == "II" and anc_len != max(0, data_qubits - 1):
                raise ValueError(
                    f"Type II requires ancillas = data-1 ({data_qubits-1}), got {anc_len}."
                )
            if type == "IV" and anc_len != 0:
                raise ValueError("Type IV requires no ancillas.")

            self.data_qubits = data_qubits
            self.ancilla_qubits = anc_len
            self.control_qubits = 1

            total_qubits = data_qubits + anc_len + self.control_qubits
            super().__init__(
                name=f"qct-{type.lower()}", num_qubits=total_qubits, params=[]
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
                anc_ct = data_qubits
            elif type == "II":
                anc_ct = max(0, data_qubits - 1)
            else:  # IV
                anc_ct = 0

            self.data_qubits = data_qubits
            self.ancilla_qubits = anc_ct
            self.control_qubits = 1
            total_qubits = data_qubits + anc_ct + self.control_qubits

            super().__init__(
                name=f"qct-{type.lower()}", num_qubits=total_qubits, params=[]
            )

            self._define()

    def __array__(self, dtype=complex, copy=None):
        """Return a numpy array for the QCTGate using test_utils tX_cos_mat."""
        if copy is False:
            raise ValueError(
                "unable to avoid copy while creating an array as requested"
            )

        N = 2**self.data_qubits
        if self.type == "I":

            mat = np.zeros((N + 1, N + 1), dtype=np.float64)

            for i in range(N + 1):
                for j in range(N + 1):
                    mat[i, j] = np.cos(np.pi * i * j / N)

                    if i == 0 or i == N:
                        mat[i, j] *= 1 / np.sqrt(2)
                    if j == 0 or j == N:
                        mat[i, j] *= 1 / np.sqrt(2)

        elif self.type == "II":
            mat = np.zeros((N, N), dtype=np.float64)

            for i in range(N):
                for j in range(N):
                    mat[i, j] = np.cos(np.pi * (i) * (j + 1 / 2) / N)
                if i == 0:
                    mat[i, j] *= 1 / np.sqrt(2)

        else:
            mat = np.zeros((N, N), dtype=np.float64)

            for i in range(N):
                for j in range(N):
                    mat[i, j] = np.cos(np.pi * (i + 1 / 2) * (j + 1 / 2) / N)
        return np.array(mat, dtype=dtype)

    def _define(self):
        """Decompose QCT. If external circuit provided, mutate it directly."""
        if self._external_circuit is not None:
            qc = self._external_circuit
            target_idxs = self._user_target_qubits
            ancilla_idxs = self._user_ancilla_qubits
        else:
            qc = QuantumCircuit(self.num_qubits, name=self.name)
            total_targets = self.data_qubits + self.control_qubits
            target_idxs = list(range(total_targets))
            ancilla_idxs = list(range(total_targets, self.num_qubits))

        if self.type == "I":
            from qct_qst_type_I import qct_type_I

            qct_type_I(qc, ancilla_idxs, target_idxs)
        elif self.type == "II":
            from qct_qst_type_II import qct_type_II

            qct_type_II(qc, ancilla_idxs, target_idxs)
        else:  # IV
            from qct_qst_type_IV import qct_type_IV

            qct_type_IV(qc, target_idxs)

        self.definition = qc
