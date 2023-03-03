"""
The purpose of this file is to reproduce Figure 4 of the paper
Good, Ian, et al. "Expanding the Design Space for Electrically-Driven Soft Robots Through Handed Shearing Auxetics."
2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022.
"""
import numpy as np
import matplotlib

matplotlib.use("Qt5Cairo")
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


# some settings
rod_names = {
    "4 rows": "HSA_ROD_FPU50_CLOSED_4ROWS",
    "6 rows": "HSA_ROD_FPU50_CLOSED_6ROWS",
    "8 rows": "HSA_ROD_FPU50_CLOSED_8ROWS",
    "10 rows": "HSA_ROD_FPU50_CLOSED_10ROWS",
    "12 rows": "HSA_ROD_FPU50_CLOSED_12ROWS",
}


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

# load real-world data
# load data for Fig 4(a)
blocked_force_data = {
    "4 rows": {
        "twist_angle": np.array([0, 30, 60, 90]),
        "blocked_force": np.array([0.0, 2.88, 7.15, 11.49]),
        "blocked_force_error": np.array(
            [
                [0.61, 0.278, 0.032, 0.15],
                [0.26, 0.1, 0.018, 0.29],
            ]
        ),
    },
    "6 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150]),
        "blocked_force": np.array([0.0, 1.27, 2.84, 4.51, 7.11, 11.13]),
        "blocked_force_error": np.array(
            [
                [0.67, 0.196, 0.054, 0.030, 0.097, 0.29],
                [0.22, 0.087, 0.025, 0.137, 0.264, 0.83],
            ]
        ),
    },
    "8 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150]),
        "blocked_force": np.array([0.0, 0.995, 2.085, 3.245, 3.801, 5.398]),
        "blocked_force_error": np.array(
            [
                [0.73, 0.23, 0.141, 0.038, 0.008, 0.035],
                [0.23, 0.10, 0.053, 0.003, 0.020, 0.123],
            ]
        ),
    },
    "10 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150, 180]),
        "blocked_force": np.array([0.0, 0.86, 1.68, 2.49, 3.27, 4.56, 6.45]),
        "blocked_force_error": np.array(
            [
                [0.775, 0.234, 0.167, 0.1, 0.006, 0.01, 0.15],
                [0.25, 0.107, 0.054, 0.02, 0.004, 0.017, 0.45],
            ]
        ),
    },
    "12 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150, 180]),
        "blocked_force": np.array([0.0, 0.582, 1.098, 1.69, 2.576, 4.27, 8.999]),
        "blocked_force_error": np.array(
            [
                [1.096, 0.298, 0.205, 0.169, 0.169, 0.243, 1.5525],
                [0.33, 0.140, 0.083, 0.066, 0.039, 0.861, 0.605],
            ]
        ),
    },
}

# load data for Fig 4(b)
spring_constant_data = {
    "4 rows": {
        "twist_angle": np.array([0, 30, 60, 90]),
        "spring_constant": np.array([1.73, 1.95, 2.35, 2.82]),
        "spring_constant_error": np.array(
            [
                [0.05, 0.031, 0.038, 0.065],
                [0.11, 0.084, 0.12, 0.166],
            ]
        ),
    },
    "6 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150]),
        "spring_constant": np.array([0.79, 0.91, 0.88, 0.96, 1.13, 1.23]),
        "spring_constant_error": np.array(
            [
                [0.01, 0.012, 0.0127, 0.0115, 0.005, 0.012],
                [0.027, 0.027, 0.036, 0.0356, 0.03, 0.008],
            ]
        ),
    },
    "8 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150]),
        "spring_constant": np.array([0.45, 0.498, 0.64, 0.65, 0.677, 0.725]),
        "spring_constant_error": np.array(
            [
                [0.0064, 0.0046, 0.006, 0.008, 0.009, 0.015],
                [0.0346, 0.013, 0.017, 0.025, 0.034, 0.025],
            ]
        ),
    },
    "10 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150, 180]),
        "spring_constant": np.array([0.39, 0.505, 0.497, 0.48, 0.51, 0.54, 0.56]),
        "spring_constant_error": np.array(
            [
                [0.0083, 0.0054, 0.0078, 0.01, 0.00667, 0.0125, 0.0144],
                [0.025, 0.0122, 0.018, 0.013, 0.0175, 0.0214, 0.0235],
            ]
        ),
    },
    "12 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150, 180]),
        "spring_constant": np.array([0.35, 0.37, 0.376, 0.372, 0.398, 0.393, 0.38]),
        "spring_constant_error": np.array(
            [
                [0.0078, 0.01, 0.003, 0.009, 0.008, 0.0167, 0.013],
                [0.030, 0.015, 0.012, 0.009, 0.012, 0.0196, 0.021],
            ]
        ),
    },
}

# load data for Fig 4(c)
minimum_energy_length_data = {
    "4 rows": {
        "twist_angle": np.array([0, 30, 60, 90]),
        "minimum_energy_length": np.array([74.86, 76.4, 78, 79.6]),
        "minimum_energy_length_error": np.array(
            [
                [2.79, 1.59, 0.64, 0.19],
                [0.3, 0.154, 0.088, 0.03],
            ]
        ),
    },
    "6 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150]),
        "minimum_energy_length": np.array([89, 90.3, 92.1, 94.1, 96.5, 98.5]),
        "minimum_energy_length_error": np.array(
            [
                [0.895, 1.02, 0.214, 0.0093, 0.097, 0.217],
                [0.28, 0.179, 0.084, 0.107, 0.41, 0.85],
            ]
        ),
    },
    "8 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150]),
        "minimum_energy_length": np.array(
            [100, 101.74, 103.49, 105.37, 107.57, 109.85]
        ),
        "minimum_energy_length_error": np.array(
            [
                [4.88, 2.72, 1.395, 0.386, 0.209, 0.007],
                [0.58, 0.46, 0.214, 0.141, 1.101, 2.275],
            ]
        ),
    },
    "10 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150, 180]),
        "minimum_energy_length": np.array(
            [111.8, 113.7, 115.7, 117.35, 119.4, 121.6, 123.9]
        ),
        "minimum_energy_length_error": np.array(
            [
                [1.921, 3.878, 2.52, 1.196, 0.55, 0.31, 0.154],
                [0.676, 0.43, 0.289, 0.217, 0.122, 1.4, 2.48],
            ]
        ),
    },
    "12 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150, 180]),
        "minimum_energy_length": np.array(
            [124, 126.4, 128.11, 129.79, 131.53, 133.36, 135.094]
        ),
        "minimum_energy_length_error": np.array(
            [
                [9.79, 6.395, 4.81, 3.33, 1.72, 0.675, 0.499],
                [1.10, 0.60, 0.45, 0.38, 0.28, 0.196, 1.506],
            ]
        ),
    },
}

# load data for Fig 4(d)
holding_torque_data = {
    "4 rows": {
        "twist_angle": np.array([0, 30, 60, 90]),
        "holding_torque": np.array([0.0, 27.9, 55.35, 87.4]),
        "holding_torque_error": np.array(
            [
                [5.71, 2.32, 1.0, 1.51],
                [1.74, 0.33, 0.31, 0.77],
            ]
        ),
    },
    "6 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150]),
        "holding_torque": np.array([0.0, 15.26, 30.3, 45.9, 58.65, 67.8]),
        "holding_torque_error": np.array(
            [
                [3.86, 1.73, 0.247, 0.467, 1.256, 2.89],
                [1.366, 0.489, 0.0242, 0.991, 2.774, 6.31],
            ]
        ),
    },
    "8 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150]),
        "holding_torque": np.array([0.0, 9.92, 20.34, 30.88, 42.85, 53.25]),
        "holding_torque_error": np.array(
            [
                [3.85, 1.78, 1.33, 0.594, 0.485, 0.904],
                [1.18, 0.43, 0.13, 0.078, 0.351, 1.264],
            ]
        ),
    },
    "10 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150, 180]),
        "holding_torque": np.array([0.0, 8.3, 17, 25.7, 34.85, 44.1, 52.05]),
        "holding_torque_error": np.array(
            [
                [4.24, 1.76, 1.45, 1.06, 0.176, 0.441, 0.954],
                [1.12, 0.44, 0.149, 0.0551, 0.06, 0.293, 1.129],
            ]
        ),
    },
    "12 rows": {
        "twist_angle": np.array([0, 30, 60, 90, 120, 150, 180]),
        "holding_torque": np.array([0.0, 6.15, 12.93, 19.78, 26.56, 32.23, 32.99]),
        "holding_torque_error": np.array(
            [
                [4.24, 1.86, 1.55, 1.385, 1.00, 0.514, 0.607],
                [1.12, 0.376, 0.179, 0.069, 0.077, 0.366, 0.437],
            ]
        ),
    },
}

if __name__ == "__main__":
    # load simulated data
    simulated_data = {}
    for rod_tag, rod_name in rod_names.items():
        filepath = (
            Path(__file__).parent.resolve() / "simulated_data" / f"{rod_name}.pickle"
        )
        with open(str(filepath), "rb") as f:
            simulated_data[rod_tag] = pickle.load(f)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.5), layout="constrained")

    # plot Fig 4(a)
    # plot real-world data
    for rod_tag, rod_data in blocked_force_data.items():
        axes[0].errorbar(
            x=rod_data["twist_angle"],
            y=rod_data["blocked_force"],
            yerr=rod_data["blocked_force_error"],
            marker=".",
            capsize=3.0,
            label=rod_tag,
        )
    axes[0].set_prop_cycle(None)
    # plot simulated data
    for rod_tag, rod_data in simulated_data.items():
        axes[0].plot(
            rod_data["blocked_force_data"]["twist_angle"] / np.pi * 180,
            rod_data["blocked_force_data"]["blocked_force"],
            marker=".",
            linestyle="--",
            label=rod_tag,
        )
    axes[0].grid(True)
    axes[0].set_title("Blocked force (a)")
    axes[0].set_xlabel(r"Twist angle $\phi_0$ [deg]")
    axes[0].set_ylabel(r"Blocked force $F_\mathrm{b}$ [N]")

    # plot Fig 4(b)
    # plot real-world data
    for rod_tag, rod_data in spring_constant_data.items():
        axes[1].errorbar(
            x=rod_data["twist_angle"],
            y=rod_data["spring_constant"],
            yerr=rod_data["spring_constant_error"],
            marker=".",
            capsize=3.0,
            label=rod_tag,
        )
    axes[1].set_prop_cycle(None)
    # plot simulated data
    for rod_tag, rod_data in simulated_data.items():
        axes[1].plot(
            rod_data["spring_constant_data"]["twist_angle"] / np.pi * 180,
            rod_data["spring_constant_data"]["spring_constant"] * 1e-3,
            marker=".",
            linestyle="--",
            label=rod_tag,
        )
    axes[1].grid(True)
    axes[1].set_title("Spring constant (b)")
    axes[1].set_xlabel(r"Twist angle $\phi_0$ [deg]")
    axes[1].set_ylabel(r"Spring constant $k$ [N/mm]")

    # plot Fig 4(c)
    # plot real-world data
    for rod_tag, rod_data in minimum_energy_length_data.items():
        axes[2].errorbar(
            x=rod_data["twist_angle"],
            y=rod_data["minimum_energy_length"],
            yerr=rod_data["minimum_energy_length_error"],
            marker=".",
            capsize=3.0,
            label=rod_tag,
        )
    axes[2].set_prop_cycle(None)
    # plot simulated data
    for rod_tag, rod_data in simulated_data.items():
        axes[2].plot(
            rod_data["minimum_energy_length_data"]["twist_angle"] / np.pi * 180,
            rod_data["minimum_energy_length_data"]["minimum_energy_length"] * 1e3,
            marker=".",
            linestyle="--",
            label=rod_tag,
        )
    axes[2].grid(True)
    axes[2].set_title("Minimum energy length (c)")
    axes[2].set_xlabel(r"Twist angle $\phi_0$ [deg]")
    axes[2].set_ylabel("Minimum energy length [mm]")

    # plot Fig 4(d)
    # plot real-world data
    for rod_tag, rod_data in holding_torque_data.items():
        axes[3].errorbar(
            x=rod_data["twist_angle"],
            y=rod_data["holding_torque"],
            yerr=rod_data["holding_torque_error"],
            marker=".",
            capsize=3.0,
            label=rod_tag,
        )
    axes[3].set_prop_cycle(None)
    # plot simulated data
    for rod_tag, rod_data in simulated_data.items():
        axes[3].plot(
            rod_data["holding_torque_data"]["twist_angle"] / np.pi * 180,
            rod_data["holding_torque_data"]["holding_torque"] * 1e3,
            marker=".",
            linestyle="--",
            label=rod_tag,
        )
    axes[3].grid(True)
    axes[3].set_title("Holding torque (d)")
    axes[3].set_xlabel(r"Twist angle $\phi_0$ [deg]")
    axes[3].set_ylabel(r"Holding torque $\tau_\mathrm{h}$ [Nmm]")

    handles, labels = axes[0].get_legend_handles_labels()
    plt.figlegend(
        handles[len(blocked_force_data.keys()) :],
        labels[len(blocked_force_data.keys()) :],
        loc="upper center",
        ncol=len(labels),
    )
    # plt.tight_layout()
    plt.subplots_adjust(
        top=0.82, bottom=0.16, left=0.06, right=0.99, hspace=0.2, wspace=0.305
    )
    plt.show()
