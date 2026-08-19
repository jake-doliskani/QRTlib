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

data_qubit_values = [3, 4, 5, 6]

noise_values = np.array([
    0.0,
    0.001,
    0.0025,
    0.005,
])

basis_gates = ["u", "cx"]


# ============================================================
# ARRAYS FOR RESULTS
# ============================================================

fidelity_lcu = np.zeros(
    (
        len(data_qubit_values),
        len(noise_values),
    )
)

fidelity_rec = np.zeros(
    (
        len(data_qubit_values),
        len(noise_values),
    )
)


# ============================================================
# LOOP OVER NUMBER OF DATA QUBITS
# ============================================================

for i, num_data_qubits in enumerate(data_qubit_values):

    print()
    print("=" * 60)
    print("DATA QUBITS =", num_data_qubits)
    print("=" * 60)


    # ========================================================
    # CREATE THE TWO QHT GATES
    # ========================================================

    qht_lcu = QHTGate(
        num_qubits=num_data_qubits,
        type="LCU",
    )

    qht_rec = QHTGate(
        num_qubits=num_data_qubits,
        type="REC",
        swap=True,
    )


    print(
        "LCU total qubits:",
        qht_lcu.num_qubits
    )

    print(
        "REC total qubits:",
        qht_rec.num_qubits
    )


    # ========================================================
    # CREATE CIRCUITS
    # ========================================================

    qc_lcu = QuantumCircuit(
        qht_lcu.num_qubits
    )

    qc_lcu.append(
        qht_lcu,
        range(qht_lcu.num_qubits)
    )


    qc_rec = QuantumCircuit(
        qht_rec.num_qubits
    )

    qc_rec.append(
        qht_rec,
        range(qht_rec.num_qubits)
    )


    # ========================================================
    # IDEAL OUTPUTS
    # ========================================================

    ideal_lcu_full = Statevector.from_instruction(
        qc_lcu
    )

    ideal_rec_full = Statevector.from_instruction(
        qc_rec
    )


    # Ancilla qubits
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


    # Keep only the ideal DATA-QUBIT state
    ideal_lcu = partial_trace(
        ideal_lcu_full,
        lcu_ancillas
    )

    ideal_rec = partial_trace(
        ideal_rec_full,
        rec_ancillas
    )


    # ========================================================
    # TRANSPILE TO SAME BASIS
    # ========================================================

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


    # ========================================================
    # PRINT CIRCUIT INFORMATION
    # ========================================================

    print()

    print("LCU")
    print(
        "Gate counts:",
        qc_lcu.count_ops()
    )
    print(
        "Depth:",
        qc_lcu.depth()
    )


    print()

    print("Recursive")
    print(
        "Gate counts:",
        qc_rec.count_ops()
    )
    print(
        "Depth:",
        qc_rec.depth()
    )


    # ========================================================
    # LOOP OVER NOISE VALUES
    # ========================================================

    for j, p in enumerate(noise_values):

        print()
        print(
            "Noise =",
            p
        )


        # ====================================================
        # CREATE NOISE MODEL
        # ====================================================

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


        # ====================================================
        # LCU
        # ====================================================

        noisy_lcu = qc_lcu.copy()


        # Save only DATA qubits
        noisy_lcu.save_density_matrix(
            qubits=list(
                range(num_data_qubits)
            ),
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


        fidelity_lcu[i, j] = F_lcu


        # ====================================================
        # RECURSIVE
        # ====================================================

        noisy_rec = qc_rec.copy()


        # Save only DATA qubits
        noisy_rec.save_density_matrix(
            qubits=list(
                range(num_data_qubits)
            ),
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


        fidelity_rec[i, j] = F_rec


        print(
            "LCU fidelity:",
            F_lcu
        )

        print(
            "REC fidelity:",
            F_rec
        )


# ============================================================
# CREATE GRID FOR 3D PLOT
# ============================================================

noise_grid, qubit_grid = np.meshgrid(
    noise_values,
    data_qubit_values,
)


# ============================================================
# CREATE 3D FIGURE
# ============================================================

fig = plt.figure(
    figsize=(11, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


# ============================================================
# LCU SURFACE
# ============================================================

ax.plot_surface(
    qubit_grid,
    noise_grid,
    fidelity_lcu,
    color="tab:blue",
    alpha=0.60,
    edgecolor="black",
    linewidth=0.3,
    antialiased=True,
)


# ============================================================
# RECURSIVE SURFACE
# ============================================================

ax.plot_surface(
    qubit_grid,
    noise_grid,
    fidelity_rec,
    color="tab:orange",
    alpha=0.60,
    edgecolor="black",
    linewidth=0.3,
    antialiased=True,
)


# ============================================================
# LABELS
# ============================================================

ax.set_xlabel(
    "Number of data qubits",
    fontsize=12,
    labelpad=12,
)

ax.set_ylabel(
    "Depolarizing noise strength",
    fontsize=12,
    labelpad=12,
)

ax.set_zlabel(
    "State fidelity",
    fontsize=12,
    labelpad=10,
)

ax.set_title(
    "QHT Noise Robustness",
    fontsize=14,
    pad=18,
)


# ============================================================
# AXIS TICKS
# ============================================================

# Only actual tested qubit values
ax.set_xticks(
    data_qubit_values
)

# Only actual tested noise values
ax.set_yticks(
    noise_values
)

ax.set_yticklabels(
    [f"{p:.4f}" for p in noise_values]
)

ax.set_zlim(
    0,
    1
)


# ============================================================
# BETTER VIEWING ANGLE
# ============================================================

ax.view_init(
    elev=25,
    azim=-55,
)


# ============================================================
# LEGEND
# ============================================================

from matplotlib.patches import Patch

legend_elements = [
    Patch(
        facecolor="tab:blue",
        edgecolor="black",
        label="QHT LCU",
        alpha=0.60,
    ),
    Patch(
        facecolor="tab:orange",
        edgecolor="black",
        label="QHT Recursive",
        alpha=0.60,
    ),
]

ax.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=11,
)


# ============================================================
# SAVE FIGURE
# ============================================================

plt.tight_layout()

plt.savefig(
    "qht_noise_3d_3to6.pdf",
    bbox_inches="tight",
)

plt.close()


print()
print(
    "Saved: qht_noise_3d_3to6.pdf"
)