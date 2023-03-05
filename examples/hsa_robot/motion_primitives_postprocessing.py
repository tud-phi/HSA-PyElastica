from hsa_elastica.utils.data_mapping_utils import (
    simulation_diagnostic_arrays_to_transformation_matrices,
)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Dict
import yaml

from examples.visualization.scene import MatplotlibScene, PyvistaScene


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

sim_id = "motion_primitives_elongation-20221025_093020"
sim_id = "motion_primitives_elongation-20230303_183612"
# sim_id = "motion_primitives_bending-north-20221025_121420"
# sim_id = "motion_primitives_bending-south-20221025_122510"
# sim_id = "motion_primitives_bending-east-20221025_122602"
# sim_id = "motion_primitives_bending-west-20221025_122523"
# sim_id = "motion_primitives_twisting-cw-20221025_203925"
# sim_id = "motion_primitives_twisting-ccw-20221025_175354"


def plot_twist_angle(_rod_diagnostic_arrays: Dict):
    plt.figure(num="Twist angles")
    times = _rod_diagnostic_arrays["time"]
    twist_angles = _rod_diagnostic_arrays["twist_angle"]
    for j in range(twist_angles.shape[2]):
        plt.plot(
            times[:, 0, j],
            twist_angles[:, 0, j, 0] / np.pi * 180,
            label=rf"$\theta_{j}$",
        )
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$\theta$ [deg]")
    plt.show()


def plot_lengths(_rod_diagnostic_arrays: Dict):
    plt.figure(num="(Rest) lengths")
    times = _rod_diagnostic_arrays["time"]
    lengths = _rod_diagnostic_arrays["lengths"]
    rest_lengths = _rod_diagnostic_arrays["rest_lengths"]
    for j in range(lengths.shape[2]):
        plt.plot(
            times[:, 0, j],
            lengths[:, 0, j, :].sum(axis=1),
            label=r"$l_{" + str(j) + "}$",
        )
    # reset color cycle
    plt.gca().set_prop_cycle(None)
    for j in range(rest_lengths.shape[2]):
        plt.plot(
            times[:, 0, j],
            rest_lengths[:, 0, j, :].sum(axis=1),
            label=r"$\hat{l}_{" + str(j) + "}$",
            linestyle="--",
        )
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$l$ [m]")
    plt.show()


def plot_z_coordinate(_rod_diagnostic_arrays: Dict, _platform_diagnostic_arrays: Dict):
    plt.figure(num="Z-coordinate")
    times = _rod_diagnostic_arrays["time"]
    positions = _rod_diagnostic_arrays["position"]
    for j in range(positions.shape[2]):
        plt.plot(
            times[:, 0, j], positions[:, 0, j, 2, -1], label=r"$z_{" + str(j) + "}$"
        )
    plt.plot(
        _platform_diagnostic_arrays["time"][:, 0, 0],
        _platform_diagnostic_arrays["position"][:, 0, 2, 0],
        label=r"$z_{p}$",
    )
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$z$ [m]")
    plt.show()


def plot_rod_distal_end_platform_position(
    _rod_diagnostic_arrays: Dict, _platform_diagnostic_arrays: Dict
):
    plt.figure(num="Platform position")
    rod_positions = _rod_diagnostic_arrays["position"]
    for j in range(rod_positions.shape[2]):
        plt.plot(
            _rod_diagnostic_arrays["time"][:, 0, j, 0],
            _rod_diagnostic_arrays["position"][:, 0, j, 0, -1],
            label=r"$x_{x," + str(j) + "}$",
        )
        plt.plot(
            _rod_diagnostic_arrays["time"][:, 0, j, 0],
            _rod_diagnostic_arrays["position"][:, 0, j, 1, -1],
            label=r"$x_{y," + str(j) + "}$",
        )
        plt.plot(
            _rod_diagnostic_arrays["time"][:, 0, j, 0],
            _rod_diagnostic_arrays["position"][:, 0, j, 2, -1],
            label=r"$x_{z," + str(j) + "}$",
        )
    plt.plot(
        _platform_diagnostic_arrays["time"][:, 0, 0],
        _platform_diagnostic_arrays["position"][:, 0, 0, -1],
        label=r"$x_{x,\mathrm{p}}$",
        linestyle="--",
    )
    plt.plot(
        _platform_diagnostic_arrays["time"][:, 0, 0],
        _platform_diagnostic_arrays["position"][:, 0, 1, -1],
        label=r"$x_{y,\mathrm{p}}$",
        linestyle="--",
    )
    plt.plot(
        _platform_diagnostic_arrays["time"][:, 0, 0],
        _platform_diagnostic_arrays["position"][:, 0, 2, -1],
        label=r"$x_{z,\mathrm{p}}$",
        linestyle="--",
    )
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$x$ [m]")
    plt.show()


def plot_rod_distal_end_platform_orientation(
    _rod_diagnostic_arrays: Dict, _platform_diagnostic_arrays: Dict
):
    plt.figure(num="Platform / Rod distal end orientation")
    rod_time = _rod_diagnostic_arrays["time"]
    rod_directors = _rod_diagnostic_arrays["directors"]
    for j in range(rod_directors.shape[2]):
        directors = rod_directors[:, 0, j, :, :, -1].transpose(0, 2, 1)
        euler_angles = Rotation.from_matrix(directors).as_euler("xyz", degrees=True)
        plt.plot(
            rod_time[:, 0, j, 0], euler_angles[:, 0], label=r"$\alpha_{" + str(j) + "}$"
        )
        plt.plot(
            rod_time[:, 0, j, 0], euler_angles[:, 1], label=r"$\beta_{" + str(j) + "}$"
        )
        plt.plot(
            rod_time[:, 0, j, 0], euler_angles[:, 2], label=r"$\gamma_{" + str(j) + "}$"
        )
    platform_rotation_matrices = _platform_diagnostic_arrays["directors"][
        :, 0, :, :, 0
    ].transpose(0, 2, 1)
    platform_euler_angles = Rotation.from_matrix(platform_rotation_matrices).as_euler(
        "xyz", degrees=True
    )
    plt.plot(
        _platform_diagnostic_arrays["time"][:, 0, 0],
        platform_euler_angles[:, 0],
        label=r"$\alpha_\mathrm{p}$",
        linestyle="--",
    )
    plt.plot(
        _platform_diagnostic_arrays["time"][:, 0, 0],
        platform_euler_angles[:, 1],
        label=r"$\beta_\mathrm{p}$",
        linestyle="--",
    )
    plt.plot(
        _platform_diagnostic_arrays["time"][:, 0, 0],
        platform_euler_angles[:, 2],
        label=r"$\gamma_\mathrm{p}$",
        linestyle="--",
    )
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"XYZ Euler angles [deg]")
    plt.show()


def plot_curvature_magnitudes(_rod_diagnostic_arrays: Dict):
    times = _rod_diagnostic_arrays["time"]
    kappas = _rod_diagnostic_arrays["kappa"]
    num_rods = kappas.shape[2]
    axes = plt.figure(num="Curvature magnitudes").subplots(num_rods, 1)
    for j in range(num_rods):
        ax = axes[j]
        kappa_x = kappas[:, 0, j, 0, :]
        kappa_y = kappas[:, 0, j, 1, :]
        kappa_mag = np.sqrt(kappa_x**2 + kappa_y**2)
        for k in range(kappa_mag.shape[-1]):
            ax.plot(
                times[:, 0, j, 0],
                kappa_mag[:, k],
                label=r"$\kappa_{" + str(k) + "}$",
            )
        ax.legend()
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$|\kappa|$ of rod " + str(j) + " [1/m]")
    plt.show()


def plot_distal_end_external_forces(_rod_diagnostic_arrays: Dict):
    plt.figure(num="Distal end external forces")
    times = _rod_diagnostic_arrays["time"]
    external_forces = _rod_diagnostic_arrays["external_forces"]
    for j in range(external_forces.shape[2]):
        plt.plot(
            times[:, 0, j],
            external_forces[:, 0, j, 0, -1],
            label=r"$F_{x," + str(j) + "}$",
        )
        plt.plot(
            times[:, 0, j],
            external_forces[:, 0, j, 1, -1],
            label=r"$F_{y," + str(j) + "}$",
        )
        plt.plot(
            times[:, 0, j],
            external_forces[:, 0, j, 2, -1],
            label=r"$F_{z," + str(j) + "}$",
        )
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$F$ [N]")
    plt.show()


def plot_distal_end_external_torques(_rod_diagnostic_arrays: Dict):
    plt.figure(num="Distal end external torques")
    times = _rod_diagnostic_arrays["time"]
    external_torques = _rod_diagnostic_arrays["external_torques"]
    for j in range(external_torques.shape[2]):
        plt.plot(
            times[:, 0, j],
            external_torques[:, 0, j, 0, -1],
            label=r"$\tau_{x," + str(j) + "}$",
        )
        plt.plot(
            times[:, 0, j],
            external_torques[:, 0, j, 1, -1],
            label=r"$\tau_{y," + str(j) + "}$",
        )
        plt.plot(
            times[:, 0, j],
            external_torques[:, 0, j, 2, -1],
            label=r"$\tau_{z," + str(j) + "}$",
        )
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$\tau$ [N]")
    plt.show()


def plot_platform_external_forces(_platform_diagnostic_arrays: Dict):
    plt.figure(num="Platform external forces")
    times = _platform_diagnostic_arrays["time"]
    external_forces = _platform_diagnostic_arrays["external_forces"]
    plt.plot(times[:, 0], external_forces[:, 0, 0, 0], label=r"$F_{x}$")
    plt.plot(times[:, 0], external_forces[:, 0, 1, 0], label=r"$F_{y}$")
    plt.plot(times[:, 0], external_forces[:, 0, 2, 0], label=r"$F_{z}$")
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$F$ [N]")
    plt.tight_layout()
    plt.show()


def plot_platform_external_torques(_platform_diagnostic_arrays: Dict):
    plt.figure(num="Platform external torques")
    times = _platform_diagnostic_arrays["time"]
    external_torques = _platform_diagnostic_arrays["external_torques"]
    plt.plot(times[:, 0], external_torques[:, 0, 0, 0], label=r"$\tau_{x}$")
    plt.plot(times[:, 0], external_torques[:, 0, 1, 0], label=r"$\tau_{y}$")
    plt.plot(times[:, 0], external_torques[:, 0, 2, 0], label=r"$\tau_{z}$")
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$\tau$ [N]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    log_dir = f"examples/logs/{sim_id}"
    with open(f"{log_dir}/robot_params.yaml", "r") as file:
        robot_params = yaml.safe_load(file)
    rod_diagnostic_arrays = np.load(f"{log_dir}/rod_diagnostic_arrays.npz")
    platform_diagnostic_arrays = np.load(f"{log_dir}/platform_diagnostic_arrays.npz")
    fps = int(
        np.round(
            (platform_diagnostic_arrays["time"].shape[0] - 1)
            / platform_diagnostic_arrays["time"][-1, 0, 0]
        )
    )

    plot_twist_angle(rod_diagnostic_arrays)
    plot_lengths(rod_diagnostic_arrays)
    # plot_z_coordinate(rod_diagnostic_arrays, platform_diagnostic_arrays)
    plot_rod_distal_end_platform_position(
        rod_diagnostic_arrays, platform_diagnostic_arrays
    )
    plot_rod_distal_end_platform_orientation(
        rod_diagnostic_arrays, platform_diagnostic_arrays
    )
    plot_curvature_magnitudes(rod_diagnostic_arrays)
    plot_distal_end_external_forces(rod_diagnostic_arrays)
    plot_distal_end_external_torques(rod_diagnostic_arrays)
    plot_platform_external_forces(platform_diagnostic_arrays)
    plot_platform_external_torques(platform_diagnostic_arrays)

    T_rod_ts, T_platform_ts = simulation_diagnostic_arrays_to_transformation_matrices(
        rod_diagnostic_arrays, platform_diagnostic_arrays
    )

    # plt_scene = MatplotlibScene(robot_params=robot_params)
    # plt_scene.animate(
    #     rod_diagnostic_arrays=rod_diagnostic_arrays,
    #     platform_diagnostic_arrays=platform_diagnostic_arrays,
    #     filepath=f"examples/videos/{sim_id}_plt.mp4",
    #     fps=fps,
    # )

    pv_scene = PyvistaScene(
        robot_params=robot_params,
        gt_settings=dict(
            num_orientation_arrows_per_rod=12,
            opacity=1.0,
            diffuse=1.0,
            ambient=1.0,
            specular=0.8,
        ),
        enable_shadows=True,
    )
    pv_scene.run(T_rod_ts[-1], T_platform_ts[-1])

    pv_scene.animate(
        T_rod_gt_ts=T_rod_ts,
        T_platform_gt_ts=T_platform_ts,
        filepath=f"examples/videos/{sim_id}_pv.mp4",
        sample_rate=fps,
        frame_rate=20,
    )
