import csv
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



data_qubit_values = list(range(3, 12))

noise_values = np.array([
    0.0,
    0.001,
    0.0025,
    0.005,
])

basis_gates = ["u", "cx"]

csv_filename = "qht_noise_results_3to11.csv"

plot_filename = "qht_noise_3d_3to11.pdf"



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



csv_file = open(
    csv_filename,
    "w",
    newline="",
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "data_qubits",
    "method",
    "total_qubits",
    "noise_strength",
    "fidelity",
    "depth",
    "u_count",
    "cx_count",
    "total_gate_count",
])



for i, num_data_qubits in enumerate(data_qubit_values):

    print()
    print("=" * 60)
    print("DATA QUBITS =", num_data_qubits)
    print("=" * 60)


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


    # Keep only DATA qubits
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
    # CIRCUIT INFORMATION
    # ========================================================

    counts_lcu = qc_lcu.count_ops()
    counts_rec = qc_rec.count_ops()

    depth_lcu = qc_lcu.depth()
    depth_rec = qc_rec.depth()

    u_lcu = counts_lcu.get("u", 0)
    cx_lcu = counts_lcu.get("cx", 0)

    u_rec = counts_rec.get("u", 0)
    cx_rec = counts_rec.get("cx", 0)

    total_gates_lcu = sum(
        counts_lcu.values()
    )

    total_gates_rec = sum(
        counts_rec.values()
    )


    print()

    print("LCU")
    print(
        "Gate counts:",
        counts_lcu
    )
    print(
        "Total gates:",
        total_gates_lcu
    )
    print(
        "Depth:",
        depth_lcu
    )


    print()

    print("Recursive")
    print(
        "Gate counts:",
        counts_rec
    )
    print(
        "Total gates:",
        total_gates_rec
    )
    print(
        "Depth:",
        depth_rec
    )



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
            method="matrix_product_state",
            noise_model=noise_model,
        )


        # ====================================================
        # LCU
        # ====================================================

        noisy_lcu = qc_lcu.copy()

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


        csv_writer.writerow([
            num_data_qubits,
            "LCU",
            qht_lcu.num_qubits,
            p,
            F_lcu,
            depth_lcu,
            u_lcu,
            cx_lcu,
            total_gates_lcu,
        ])



        csv_writer.writerow([
            num_data_qubits,
            "REC",
            qht_rec.num_qubits,
            p,
            F_rec,
            depth_rec,
            u_rec,
            cx_rec,
            total_gates_rec,
        ])


        
        csv_file.flush()




csv_file.close()



noise_grid, qubit_grid = np.meshgrid(
    noise_values,
    data_qubit_values,
)


fig = plt.figure(
    figsize=(11, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


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


ax.set_xticks(
    data_qubit_values
)

ax.set_yticks(
    noise_values
)

ax.set_yticklabels(
    [
        f"{p:.4f}"
        for p in noise_values
    ]
)

ax.set_zlim(
    0,
    1
)


ax.view_init(
    elev=25,
    azim=-55,
)


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



plt.tight_layout()

plt.savefig(
    plot_filename,
    bbox_inches="tight",
)

plt.close()



print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(
    "CSV saved:",
    csv_filename
)

print(
    "Plot saved:",
    plot_filename
)