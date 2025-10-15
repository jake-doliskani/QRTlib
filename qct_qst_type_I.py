from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector
from transform_utils import (
    compute_or_with_cleanup,
    ctrl_twos_complement,
)


def qct_type_I(
    circuit: QuantumCircuit, anc_qubits: list[int], target_qubits: list[int]
):
    """
    Type-I Quantum Cosine Transform (QCT-I) via the T–QFT–T† construction.

    Mathematical definition:
        C_N^{I} = sqrt(2 / N) [ k_m k_n cos(m n π / N) ],   for m, n = 0, 1, …, N
      where k_0 = 1 / sqrt(2) and k_r = 1 for r >= 1 (the usual DST/DCT-I scaling).

    Identity used (Klappenecker–Roetteler '01):
        T_N^* · QFT_{2N} · T_N  =  |0⟩⟨0| ⊗ QCT_N^{I}  +  i |1⟩⟨1| ⊗ QST_N^{I}

    Layout notes:
      - ctrl = target_qubits[-1] is the control qubit selecting cosine (|0⟩) vs sine (|1⟩).
      - T_gate implements T_N; QFTGate(len(target_qubits)) implements QFT_{2N};
        T_dagger_gate implements T_N^*.
      - This routine prepares |0⟩⟨0| ⊗ QCT_N^{I} on the data subspace when ctrl is |0⟩.

    Gate structure:
      1) T_N
      2) QFT_{2N}
      3) T_N^*
    """

    # 1) Apply T_N
    T_gate(circuit, target_qubits=target_qubits, anc_qubits=anc_qubits)

    # 2) Apply QFT_{2N}
    circuit.append(QFTGate(len(target_qubits)), target_qubits)

    # 3) Apply T_N^*
    T_inversed_gate(circuit, target_qubits=target_qubits, anc_qubits=anc_qubits)


def T_gate(circuit: QuantumCircuit, target_qubits: list[int], anc_qubits: list[int]):
    """
    Implementation of the unitary T_N = ctrl_twos_complement · D on (ctrl ⊗ data).

    Layout convention:
      - target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
        * ctrl = target_qubits[-1] is the control qubit.
      - anc_qubits = [final_result, a0, a1, ..., a_k]
        * final_result (index 0) and a0 (index 1) are used for the cleanup step.
        * anc_qubits[2:] is used as the carry chain for the controlled two’s complement.

    Mathematical definition:
        For x ∈ {1, …, N−1} and x' = (two’s complement of x):
            T_N |00⟩ = |00⟩
            T_N |0x⟩ = (|0x⟩ + |1x'⟩)/√2
            T_N |10⟩ = |10⟩
            T_N |1x⟩ = i(|0x⟩ − |1x'⟩)/√2

    Decomposition:
        - D_gate: applies a conditional rotation on the control qubit based on the data register.
        - ctrl_twos_complement: performs the two’s-complement mapping x → x', controlled by ctrl.
        - Cleanup: restores ancillas to |0⟩ through CCX and OR-based uncomputation.

    """
    ctrl = target_qubits[-1]  # control qubit

    # --- Decomposition:  T_N = ctrl_twos_complement · D -----------------------
    # Step 1: Apply D gate (conditional rotation on control)
    D_gate(circuit, target_qubits=target_qubits, anc_qubits=anc_qubits)

    # Step 2: Apply controlled two’s complement using anc_qubits[2:] as carry chain
    ctrl_twos_complement(
        circuit, anc_qubits=anc_qubits[2:], target_qubits=target_qubits
    )
    # --------------------------------------------------------------------------

    # Step 3: Cleanup ancillas used in D_gate and intermediate operations
    circuit.ccx(ctrl, anc_qubits[0], anc_qubits[1])
    compute_or_with_cleanup(circuit, target_qubits=target_qubits, anc_qubits=anc_qubits)


def T_inversed_gate(
    circuit: QuantumCircuit, target_qubits: list[int], anc_qubits: list[int]
):
    """
    Implementation of T_N* on (ctrl ⊗ data), following the inverse decomposition:
        T_N* = D_dagger_gate · ctrl_twos_complement

    Layout convention:
      - target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
        * ctrl = target_qubits[-1] is the control qubit.
      - anc_qubits = [final_result, a0, a1, ..., a_k]
        * final_result (index 0) and a0 (index 1) participate in the local CCX cleanup.
        * anc_qubits[2:] is the carry chain used by the controlled two’s complement.

    Steps (inverse of T_N = ctrl_twos_complement · D):
      1) Undo OR intermediates created by D_gate-related logic.
      2) Local CCX cleanup tied to the D/T realization.
      3) Apply ctrl_twos_complement (self-inverse block).
      4) Apply D_dagger_gate.
    """
    ctrl = target_qubits[-1]  # control qubit

    # 1) Uncompute OR intermediates and restore scratch ancillas
    compute_or_with_cleanup(circuit, target_qubits=target_qubits, anc_qubits=anc_qubits)

    # 2) Local CCX cleanup (matching the forward T_gate's final CCX)
    circuit.ccx(ctrl, anc_qubits[0], anc_qubits[1])

    # 3) Apply controlled two’s complement using the carry-chain slice
    ctrl_twos_complement(
        circuit, anc_qubits=anc_qubits[2:], target_qubits=target_qubits
    )

    # 4) Apply D* (inverse of the conditional rotation block)
    D_inversed_gate(circuit, target_qubits=target_qubits, anc_qubits=anc_qubits)


def D_gate(circuit: QuantumCircuit, target_qubits: list[int], anc_qubits: list[int]):
    """
    Conditional rotation block D acting on (ctrl ⊗ data).

    Layout:
      - target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
        * ctrl = target_qubits[-1] is the control qubit.
      - anc_qubits = [final_result, a0, a1, ..., a_k]
        * final_result (anc_qubits[0]) will hold OR(data) and is used to condition gates on data ≠ 0.
        * a0 (anc_qubits[1]) is used in the final CCX tidy-up.

    Mathematical action (for x ∈ {1,…,N−1}):
        D|00⟩ = |00⟩
        D|0x⟩ = (|0x⟩ + |1x⟩)/√2
        D|10⟩ = |10⟩
        D|1x⟩ = i(|0x⟩ − |1x⟩)/√2

      Equivalently: apply S then H on the control qubit, conditioned on (data ≠ 0).

    Implementation outline:
      1) compute_or_with_cleanup(...) → sets final_result = OR(data),
         cleaning only intermediate ancillas (final_result is preserved).
      2) Apply controlled-S and controlled-H to ctrl, controlled by final_result.
      3) CCX(ctrl, final_result, a0) — local disentangling step consistent with our D/T realization.
    """
    ctrl = target_qubits[-1]  # explicit alias for the control qubit

    # 1) Detect if data ≠ 0 and store the flag in anc_qubits[0] (final_result).
    #    Intermediates used during the OR reduction are uncomputed; final_result is kept.
    compute_or_with_cleanup(circuit, target_qubits=target_qubits, anc_qubits=anc_qubits)

    # 2) Apply S and H to ctrl, conditioned on (data ≠ 0) via final_result.
    circuit.cs(anc_qubits[0], ctrl)  # controlled-S on ctrl (control = final_result)
    circuit.ch(anc_qubits[0], ctrl)  # controlled-H on ctrl (control = final_result)

    # 3) Local cleanup step tied to this D realization.
    circuit.ccx(ctrl, anc_qubits[0], anc_qubits[1])


def D_inversed_gate(
    circuit: QuantumCircuit, target_qubits: list[int], anc_qubits: list[int]
):
    """
    Inverse of the conditional rotation block D (i.e., D*) acting on (ctrl ⊗ data).

    Layout:
      - target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
        * ctrl = target_qubits[-1] is the control qubit.
      - anc_qubits = [final_result, a0, a1, ..., a_k]
        * final_result = anc_qubits[0] holds OR(data) (flag for data ≠ 0).
        * a0 = anc_qubits[1] used in the local CCX step.

    Mathematical action (for x ∈ {1,…,N−1}):
        D*|00⟩ = |00⟩
        D*|0x⟩ = (|0x⟩ − |1x⟩)/√2
        D*|10⟩ = |10⟩
        D*|1x⟩ = −i(|0x⟩ + |1x⟩)/√2

      Equivalently: apply H then S† on the control qubit, conditioned on (data ≠ 0).

    Implementation outline (mirrors D, in reverse):
      1) Local CCX disentangling consistent with our D/D* realization.
      2) Apply controlled-H and controlled-S† to ctrl (controls on final_result).
      3) Uncompute OR intermediates (final_result flag may be preserved upstream).
    """
    ctrl = target_qubits[-1]  # explicit alias for the control qubit

    # 1) Local CCX step corresponding to the forward D cleanup structure
    circuit.ccx(ctrl, anc_qubits[0], anc_qubits[1])

    # 2) Apply H and then S† to ctrl, both conditioned on (data ≠ 0) via final_result
    circuit.ch(anc_qubits[0], ctrl)  # controlled-H on ctrl
    circuit.csdg(anc_qubits[0], ctrl)  # controlled-S† on ctrl

    # 3) Uncompute OR chain (cleans intermediate ancillas used to derive final_result)
    compute_or_with_cleanup(circuit, target_qubits=target_qubits, anc_qubits=anc_qubits)
