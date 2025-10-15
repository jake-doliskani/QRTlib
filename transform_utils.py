import numpy as np
from qiskit import QuantumCircuit
from typing import List


def or_gate(circuit: QuantumCircuit, data1: int, data2: int, result: int):
    """
    Two-qubit logical OR gate implemented reversibly using CNOT and Toffoli gates.

    Layout convention:
      - data1, data2: Input qubits whose logical OR is to be computed.
      - result: Target qubit (ancilla) where the OR result will be stored.

    Operation:
      This reversible OR operation follows the sequence:
        1. Apply CNOT(data1 → result) — flips the result if data1 = 1.
        2. Apply CNOT(data2 → result) — flips the result if data2 = 1.
        3. Apply CCX(data1, data2 → result) — flips result again if both are 1.

      Behavior summary:
        - If only one of (data1, data2) = 1 → result flips once → result = 1.
        - If both are 1 → result flips three times (CNOT, CNOT, Toffoli) → net flip = 1 → result = 1.
        - If both are 0 → result remains 0.

      Thus, result = data1 OR data2 (reversible form).

    Notes:
      - This gate assumes result is initialized to |0⟩.
    """
    # Step 1: flip result if data1 = 1
    circuit.cx(data1, result)

    # Step 2: flip result if data2 = 1
    circuit.cx(data2, result)

    # Step 3: flip result again if both inputs are 1
    circuit.ccx(data1, data2, result)


def compute_or_forward(
    circuit: QuantumCircuit,
    target_qubits: List[int],
    anc_qubits: List[int],
    store_to_final_result: bool = True,
) -> List[tuple[int, int, int]]:
    """
    Forward computation of the OR over all data qubits using a tree-style reduction.

    Layout convention:
      - target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
        * Data qubits: b0..b_{n-1} (little-endian)
        * ctrl (last element) is the **control qubit** (not used in this function directly)
      - anc_qubits = [(final_result), a0, a1, ..., a_k]
        * final_result (index 0) is an **optional** designated output qubit.
          If provided and `store_to_final_result=True`, the final OR value
          will be stored there.
        * a0, a1, ... are temporary ancillas for intermediate OR computations.

    Args:
        circuit: The quantum circuit.
        target_qubits: Data qubits followed by the control qubit (ctrl).
        anc_qubits: [final_result, a0, a1, ..., a_k] (little-endian).
                    The first element (final_result) may be omitted if not used.
        store_to_final_result: Whether to store the final OR result into
                               `final_result` (anc_qubits[0]) when available.
                               If False or no ancillas provided, the result
                               remains where it naturally lands.

    Returns:
        operation_log: A list of (q1, q2, target) tuples for each OR reduction performed.

    Notes:
        - Uses helper `or_gate(circuit, q1, q2, target)` that performs
          target ← (q1 OR q2) reversibly.
        - The control qubit (ctrl) is not touched here; it is included
          to maintain consistent register layout across controlled routines.
    """
    # Extract data qubits (exclude control)
    n = len(target_qubits) - 1
    data_qubits = target_qubits[:n]

    # Handle ancillas and optional final_result
    final_result_idx = anc_qubits[0] if anc_qubits else None
    scratch_a = anc_qubits[1:] if len(anc_qubits) > 1 else []

    # Edge cases
    if n == 0:
        return []  # No data → nothing to compute

    if n == 1:
        # Single input: optionally copy it to final_result
        if (
            store_to_final_result
            and final_result_idx is not None
            and data_qubits[0] != final_result_idx
        ):
            circuit.cx(data_qubits[0], final_result_idx)
        return []

    # Tree reduction: iteratively OR pairs into available targets
    current_layer = list(data_qubits)
    operation_log: List[tuple[int, int, int]] = []

    while len(current_layer) > 1:
        next_layer = []
        for i in range(0, len(current_layer), 2):
            if i + 1 < len(current_layer):
                q1, q2 = current_layer[i], current_layer[i + 1]

                # Prefer a free scratch ancilla; otherwise use final_result if available
                if scratch_a:
                    tgt = scratch_a.pop()
                elif final_result_idx is not None:
                    tgt = final_result_idx
                else:
                    raise ValueError(
                        "No available ancilla or final_result qubit to store intermediate OR result."
                    )

                or_gate(circuit, q1, q2, tgt)
                operation_log.append((q1, q2, tgt))
                next_layer.append(tgt)
            else:
                # Odd leftover propagates unchanged
                next_layer.append(current_layer[i])

        current_layer = next_layer

    # Final value location
    result_loc = current_layer[0]

    # If requested, move it into final_result (if it exists)
    if (
        store_to_final_result
        and final_result_idx is not None
        and result_loc != final_result_idx
    ):
        circuit.cx(result_loc, final_result_idx)

    return operation_log


def compute_or_backward(
    circuit: QuantumCircuit, operation_log: List[tuple[int, int, int]]
):
    """
    Uncompute the intermediate OR results stored in ancilla qubits.

    Purpose:
      This reverses the operations recorded by `compute_or_forward`, cleaning
      up ancilla qubits used for intermediate OR values.

    Important:
      - The reversible OR gate (`or_gate`) is self-inverse:
            or_gate == inverse(or_gate)
        so the same gate sequence can be reused for uncomputation.
      - This function only clears the temporary ancilla targets that appear
        in `operation_log`; it does **not** uncompute the optional final_result
        qubit if one was used in the forward pass.

    Args:
        circuit: The quantum circuit.
        operation_log: List of (q1, q2, target) tuples recorded during forward OR computation.
    """
    # Apply the same OR gates in reverse order to uncompute intermediate results
    for q1, q2, target in reversed(operation_log):
        or_gate(circuit, q1, q2, target)


def compute_or_with_cleanup(
    circuit: QuantumCircuit,
    target_qubits: List[int],
    anc_qubits: List[int],
    is_recovered: bool = True,
    store_to_final_result: bool = True,
):
    """
    Compute the OR of all data qubits and (optionally) clean up ancilla qubits.

    Layout convention:
      - target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
        * Data qubits: b0..b_{n-1} (little-endian)
        * ctrl (last element) is the control qubit (not used directly here)
      - anc_qubits = [final_result, a0, a1, ..., a_k]
        * final_result (index 0) is an **optional** output qubit; if present and
          `store_to_final_result=True`, the final OR value will be copied there.
        * a0, a1, ... are scratch ancillas for intermediate ORs.

    Args:
        circuit: The quantum circuit.
        target_qubits: Data qubits followed by the control qubit (ctrl).
        anc_qubits: [final_result, a0, a1, ..., a_k]. The first element may be
                    omitted if no dedicated final_result is used.
        is_recovered: If True, uncompute intermediate OR results to return all
                      scratch ancillas to |0⟩ (default True).
        store_to_final_result: If True and `final_result` exists (anc_qubits[0]),
                               copy the final OR value there; otherwise leave it
                               where it naturally lands.

    Notes:
        - Uses `compute_or_forward` to perform the tree reduction and (optionally)
          copy the result into `final_result`.
        - If `is_recovered=True`, calls `compute_or_backward` to clean only the
          intermediate ancillas recorded in the operation log. It does **not**
          uncompute any copy placed in `final_result`.
    """
    # Forward OR computation (records all intermediate OR steps)
    operation_log = compute_or_forward(
        circuit, target_qubits, anc_qubits, store_to_final_result
    )

    # Optionally uncompute intermediate ancillas back to |0⟩
    if is_recovered:
        compute_or_backward(circuit, operation_log)


def ctrl_inc_by_1(
    circuit: QuantumCircuit, anc_qubits: list[int], target_qubits: list[int]
):
    """
    Controlled increment-by-one on an n-qubit data register.

    Layout convention:
      - target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
        where b0 is the least-significant data qubit,
        and ctrl (last element) is the external control qubit.
      - anc_qubits = [a1, a2, ..., a_{n-2}] are carry ancillas
        required only when n >= 3.

    Semantics:
      If ctrl == 1: |b>|ctrl> → |b + 1 mod 2^n>|ctrl>
      If ctrl == 0: |b>|ctrl> → |b>|ctrl>

    The implementation follows the paper’s “ripple-carry with uncomputation” design:
      1. Compute carry qubits forward from LSB to MSB.
      2. Propagate conditional flips under control of ctrl.
      3. Uncompute all carries to restore ancillas to |0>.
    """

    # ----------------------------
    # Special case 1: 1-bit data
    # target_qubits = [b0, ctrl]
    # ----------------------------
    if len(target_qubits) == 2:
        b0, ctrl = target_qubits
        circuit.cx(ctrl, b0)  # if ctrl=1, flip LSB
        return

    # ----------------------------
    # Special case 2: 2-bit data
    # target_qubits = [b0, b1, ctrl]
    # ----------------------------
    if len(target_qubits) == 3:
        b0, b1, ctrl = target_qubits
        # Ripple-carry increment of 2-bit register [b1 b0], controlled by ctrl
        circuit.ccx(ctrl, b0, b1)  # carry into b1 when ctrl=1 and b0=1
        circuit.cx(ctrl, b0)  # flip LSB when ctrl=1
        return

    # ----------------------------
    # General case: n >= 3 data qubits
    # target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
    # anc_qubits = [a1, ..., a_{n-2}]
    # ----------------------------
    ctrl = target_qubits[-1]

    # Compute first carry: a1 = b0 AND b1
    circuit.ccx(target_qubits[0], target_qubits[1], anc_qubits[0])

    # Forward propagate carries: a_{i+1} = a_i AND b_{i+1}
    for i in range(len(anc_qubits) - 1):
        circuit.ccx(anc_qubits[i], target_qubits[i + 2], anc_qubits[i + 1])

    # Backward pass (ripple under ctrl):
    for i in range(len(anc_qubits) - 1, 0, -1):
        circuit.ccx(
            ctrl, anc_qubits[i], target_qubits[i + 2]
        )  # flip b_{i+2} if ctrl=1 and a_i=1
        circuit.ccx(
            anc_qubits[i - 1], target_qubits[i + 1], anc_qubits[i]
        )  # uncompute a_i

    # Handle b2 using a1, then clean a1
    circuit.ccx(ctrl, anc_qubits[0], target_qubits[2])  # flip b2 if ctrl=1 and a1=1
    circuit.ccx(target_qubits[0], target_qubits[1], anc_qubits[0])  # uncompute a1

    # Final two LSB operations under ctrl
    circuit.ccx(ctrl, target_qubits[0], target_qubits[1])  # propagate carry into b1
    circuit.cx(ctrl, target_qubits[0])  # flip LSB


def ctrl_dec_by_1(
    circuit: QuantumCircuit, anc_qubits: List[int], target_qubits: List[int]
):
    """
    Controlled decrement-by-one on an n-qubit data register.

    Layout convention:
      - target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
        where b0 is the least-significant data qubit,
        and ctrl (last element) is the external control qubit.
      - anc_qubits = [a1, a2, ..., a_{n-2}] are carry ancillas used during the ripple.

    Semantics (inverse of the controlled increment):
      If ctrl == 1: |b>|ctrl> → |b - 1 (mod 2^n)>|ctrl>
      If ctrl == 0: |b>|ctrl> → |b>|ctrl>
    """

    # ----------------------------
    # Special case 1: 1-bit data
    # target_qubits = [b0, ctrl]
    # ----------------------------
    if len(target_qubits) == 2:
        b0, ctrl = target_qubits
        circuit.cx(ctrl, b0)  # same as +1/-1 on a single bit under control
        return

    # ----------------------------
    # Special case 2: 2-bit data
    # target_qubits = [b0, b1, ctrl]
    # ----------------------------
    if len(target_qubits) == 3:
        b0, b1, ctrl = target_qubits
        # Inverse sequence of the 2-bit increment:
        circuit.cx(ctrl, b0)  # first undo the final LSB flip
        circuit.ccx(ctrl, b0, b1)  # then undo the carry into b1
        return

    # ----------------------------
    # General case: n >= 3 data qubits
    # target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
    # anc_qubits    = [a1, ..., a_{n-2}]
    # ----------------------------
    ctrl = target_qubits[-1]  # explicit handle for the control qubit

    # Flip LSB under ctrl (start undoing the increment tail)
    circuit.cx(ctrl, target_qubits[0])

    # Undo the carry into b1 under ctrl
    circuit.ccx(ctrl, target_qubits[0], target_qubits[1])

    # Start uncomputing the first carry a1 = b0 AND b1
    circuit.ccx(target_qubits[0], target_qubits[1], anc_qubits[0])

    # Propagate from top to bottom:
    # for i = 0..(len(anc_qubits)-2):
    #   flip b_{i+2} under (ctrl & a_i), then re-derive next carry into a_{i+1}
    for i in range(len(anc_qubits) - 1):
        circuit.ccx(
            ctrl, anc_qubits[i], target_qubits[i + 2]
        )  # conditional flip of b_{i+2}
        circuit.ccx(
            anc_qubits[i], target_qubits[i + 2], anc_qubits[i + 1]
        )  # re-derive a_{i+1}

    # Apply the last conditional flip at the MSB-1 using the last carry
    circuit.ccx(ctrl, anc_qubits[-1], target_qubits[-2])

    # T-chain back through the carries to fully uncompute them (top → bottom)
    for i in range(len(anc_qubits) - 1, 0, -1):
        circuit.ccx(anc_qubits[i - 1], target_qubits[i + 1], anc_qubits[i])

    # Finish uncomputing the first carry a1
    circuit.ccx(target_qubits[0], target_qubits[1], anc_qubits[0])


def ctrl_ones_complement(circuit: QuantumCircuit, target_qubits: List[int]):
    """
    Controlled one's complement operation.

    Layout convention:
      - target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
        where the **last element (target_qubits[-1]) is the control qubit**.
        The rest are data qubits.

    Behavior:
      If ctrl == 1: flip (X) all data qubits.
      If ctrl == 0: leave all data qubits unchanged.
    """
    # Apply controlled-X to each data qubit
    for data_idx in target_qubits[:-1]:
        # target_qubits[-1] is the control qubit
        circuit.cx(target_qubits[-1], data_idx)


def ctrl_twos_complement(
    circuit: QuantumCircuit, anc_qubits: List[int], target_qubits: List[int]
):
    """
    Controlled two's-complement using an ancilla-backed increment.

    What this implements:
      - Register layout: target_qubits = [b0, b1, ..., b_{n-1}, ctrl]
        where the **last element target_qubits[-1] is the control qubit**,
        and the preceding n elements are data bits (LSB-first).
      - Ancillas: anc_qubits provide the carry chain used by the controlled increment.
        For n >= 3 data bits, the increment routine expects exactly n-2 ancillas.

      - Operation (under control 'ctrl'):
          1) Controlled one's-complement: flip all data bits if ctrl == 1.
             (via `ctrl_ones_complement`, which uses target_qubits[-1] as the control)
          2) Controlled increment-by-1: add 1 to the data register if ctrl == 1,
             using ripple-carry logic that consumes/cleans anc_qubits.
             (via `ctrl_inc_by_1`, which requires the ancilla carry chain)

      - If ctrl == 0: the data register is left unchanged.

    Notes:
      - This routine intentionally relies on ancillas through `ctrl_inc_by_1`.
      - It assumes the helper functions `ctrl_ones_complement` and `ctr_inc_by_1`
        follow the same register conventions.
    """
    # Step 1: controlled one's-complement (target_qubits[-1] is the control)
    ctrl_ones_complement(circuit, target_qubits)

    # Step 2: controlled increment-by-1 using ancilla carry chain
    ctrl_inc_by_1(circuit, anc_qubits, target_qubits)


def V_N_gate(circuit: QuantumCircuit, target_qubits: List[int]):
    """
    Apply V gate to the circuit.

    V=pi_1(H )
    pi_1 is a controlled flip gate
    """
    # apply H gate to the control qubit
    circuit.h(target_qubits[-1])

    # apply controlled flip gate
    ctrl_ones_complement(circuit, target_qubits)


def D_One_gate(
    circuit: QuantumCircuit, target_qubits: List[int], is_L_adjoint: bool = False
):
    """
    Apply D_One gate to the target qubits

    D1 = (C ⊗ 1_N)(delta1 ⊕ delta2)

    delta1 = L_n ⊗ L_n-1 ⊗ ... ⊗ L_1

    delta2 = K_n ⊗ K_n-1 ⊗ ... ⊗ K_1

    L_i = diag(1, w^2^(j-1))
    K_i = diag(bar(w)^2^(j-1), 1)
    C = diag(1, bar(w))

    where w is the primitive root of unity.

    """
    theta = (2 * np.pi) / (4 * (2 ** (len(target_qubits) - 1)))
    # print(f"Applying D_One gate with theta: {theta}")
    # print(f"Target indices len: {len(target_qubits) - 1}")
    delta2(circuit, target_qubits, theta, is_L_adjoint)

    circuit.x(target_qubits[-1])  # apply X gate to the last target qubit
    delta1(circuit, target_qubits, theta)
    circuit.x(target_qubits[-1])  # apply X gate to the last target qubit

    # apply C gate
    circuit.p(-theta, target_qubits[-1])


def delta1(circuit: QuantumCircuit, target_qubits: List[int], theta: float):
    """
    Apply delta1 gate to the target qubits

    delta1 = L_n ⊗ L_n-1 ⊗ ... ⊗ L_1
    where L_i = diag(1, w^2^(i-1))
    the theta = 2 * pi / N, where N is the number of qubits.
    """
    # reversely apply L_i gates, L_1 for the last target qubit...
    for target_qubit in reversed(target_qubits[:-1]):
        # circuit.x(target_qubit)
        L_i(
            circuit,
            target_qubits[-1],
            target_qubit,
            target_qubit + 1,
            theta,
        )
        # circuit.x(target_qubit)


def delta2(
    circuit: QuantumCircuit,
    target_qubits: List[int],
    theta: float,
    is_L_adjoint: bool,
):
    """
    Apply delta2 gate to the target qubits

    delta2 = K_n ⊗ K_n-1 ⊗ ... ⊗ K_1
    where K_i = X L_i^dagger X
    the theta = 2 * pi / N, where N is the number of qubits.
    """
    # reversely apply K_i gates, K_1 for the last target qubit...
    if is_L_adjoint == True:
        for target_qubit in reversed(target_qubits[:-1]):
            L_i(circuit, target_qubits[-1], target_qubit, target_qubit + 1, -theta)
    else:
        for target_qubit in reversed(target_qubits[:-1]):
            K_i(circuit, target_qubits[-1], target_qubit, target_qubit + 1, theta)


def K_i(
    circuit: QuantumCircuit,
    control_qubit: int,
    target_qubit: int,
    i: int,
    theta: float,
):
    """
    Apply K_i gate to the target qubit

    K_i = X L_i^dagger X
    """
    circuit.x(target_qubit)
    # print(f"K_i: control {control_qubit}, target {target_qubit}, i {i}, theta {theta}")
    L_i(circuit, control_qubit, target_qubit, i, -theta)
    circuit.x(target_qubit)

    # phase = np.exp((2 ** (i - 1)) * (-theta) * 1j)
    # diag = [phase, 1]

    # k_gate = Diagonal(diag).control(1)  # Create a controlled diagonal gate
    # circuit.append(k_gate, [control_qubit, target_qubit])


def L_i(
    circuit: QuantumCircuit,
    control_qubit: int,
    target_qubit: int,
    i: int,
    theta: float,
):
    """
    Apply L_i gate to the target qubit

    L_i = diag(1, w^2^(i-1)), which is R_z(2 ** (i - 1) * theta)

    """
    circuit.cp((2 ** (i - 1)) * theta, control_qubit, target_qubit)
