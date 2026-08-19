import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import (
    Statevector,
    state_fidelity,
    partial_trace,
)

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from QHTGate import QHTGate


# ============================================================
# SETTINGS
# ============================================================

num_data_qubits = 4

noise_values = np.linspace(0.0, 0.05, 11)

basis_gates = ["u", "cx"]


# ============================================================
# CREATE THE TWO QHT GATES
# ============================================================

qht_lcu = QHTGate(
    num_qubits=num_data_qubits,
    type="LCU",
)

qht_rec = QHTGate(
    num_qubits=num_data_qubits,
    type="REC",
    swap=True,
)


print("LCU total qubits:", qht_lcu.num_qubits)
print("REC total qubits:", qht_rec.num_qubits)


# ============================================================
# CREATE CIRCUITS
# ============================================================

qc_lcu = QuantumCircuit(qht_lcu.num_qubits)

qc_lcu.append(
    qht_lcu,
    range(qht_lcu.num_qubits)
)


qc_rec = QuantumCircuit(qht_rec.num_qubits)

qc_rec.append(
    qht_rec,
    range(qht_rec.num_qubits)
)


# ============================================================
# IDEAL OUTPUTS
# ============================================================

# Full ideal output states
ideal_lcu_full = Statevector.from_instruction(qc_lcu)

ideal_rec_full = Statevector.from_instruction(qc_rec)


# Ancilla qubits that should be traced out
lcu_ancillas = list(
    range(
        num_data_qubits,
        qht_lcu.num_qubits
    )
)

rec_ancillas = list(
    range(
        num_data_qubits,
        qht_rec.num_qubits
    )
)


# Keep only the DATA-QUBIT ideal states
ideal_lcu = partial_trace(
    ideal_lcu_full,
    lcu_ancillas
)

ideal_rec = partial_trace(
    ideal_rec_full,
    rec_ancillas
)


# ============================================================
# DECOMPOSE TO THE SAME ELEMENTARY GATES
# ============================================================

qc_lcu = transpile(
    qc_lcu,
    basis_gates=basis_gates,
    optimization_level=0,
)

qc_rec = transpile(
    qc_rec,
    basis_gates=basis_gates,
    optimization_level=0,
)


# ============================================================
# PRINT CIRCUIT INFORMATION
# ============================================================

print()
print("LCU")
print("Gate counts:", qc_lcu.count_ops())
print("Depth:", qc_lcu.depth())

print()

print("Recursive")
print("Gate counts:", qc_rec.count_ops())
print("Depth:", qc_rec.depth())


# ============================================================
# ARRAYS FOR RESULTS
# ============================================================

fidelity_lcu = []
fidelity_rec = []


# ============================================================
# NOISE SWEEP
# ============================================================

for p in noise_values:

    print()
    print("Noise =", p)


    # --------------------------------------------------------
    # CREATE NOISE MODEL
    # --------------------------------------------------------

    noise_model = NoiseModel()

    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p, 1),
        ["u"],
    )

    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p, 2),
        ["cx"],
    )


    simulator = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
    )


    # ========================================================
    # LCU
    # ========================================================

    noisy_lcu = qc_lcu.copy()


    # Save ONLY the 4 data qubits
    noisy_lcu.save_density_matrix(
        qubits=list(range(num_data_qubits)),
        label="data",
    )


    result_lcu = simulator.run(
        noisy_lcu
    ).result()


    rho_lcu = result_lcu.data(0)[
        "data"
    ]


    F_lcu = state_fidelity(
        ideal_lcu,
        rho_lcu,
    )


    fidelity_lcu.append(
        F_lcu
    )


    # ========================================================
    # RECURSIVE
    # ========================================================

    noisy_rec = qc_rec.copy()


    # Save ONLY the 4 data qubits
    noisy_rec.save_density_matrix(
        qubits=list(range(num_data_qubits)),
        label="data",
    )


    result_rec = simulator.run(
        noisy_rec
    ).result()


    rho_rec = result_rec.data(0)[
        "data"
    ]


    F_rec = state_fidelity(
        ideal_rec,
        rho_rec,
    )


    fidelity_rec.append(
        F_rec
    )


    print("LCU fidelity:", F_lcu)
    print("REC fidelity:", F_rec)


# ============================================================
# SAVE FIGURE
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    noise_values,
    fidelity_lcu,
    marker="o",
    label="QHT LCU",
)

plt.plot(
    noise_values,
    fidelity_rec,
    marker="s",
    label="QHT Recursive",
)

plt.xlabel(
    "Depolarizing noise strength"
)

plt.ylabel(
    "State fidelity"
)

plt.title(
    "QHT Noise Robustness — 4 Data Qubits"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "qht_noise_4qubits.pdf",
    bbox_inches="tight",
)

plt.close()


print()
print("Saved: qht_noise_4qubits.pdf")