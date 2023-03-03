from elastica.external_forces import NoForces, EndpointForces
from elastica.typing import SystemType
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from hsa_elastica.forcing import ProximalEndTorsion, LocalEndpointForces
from hsa_elastica.simulation import HsaRodSimulator

from examples.parameters.rod_params import ROD_ROBOT_SIM


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

# define some parameters
rod_params = ROD_ROBOT_SIM


def platform_fixed_constraint_test():
    sim = HsaRodSimulator(rod_params=rod_params, duration=10.0)
    sim.configure(
        finalize=False, constrain_extension=False, follow_auxetic_trajectory=True
    )

    # add ramp actuation torsional torque
    sim.add_forcing_to(sim.rod).using(
        ProximalEndTorsion,
        torsional_torque=rod_params["max_actuation_torque"],
        ramp_up_time=0.0,
    )

    # add endpoint forces
    sim.add_forcing_to(sim.rod).using(
        LocalEndpointForces,
        start_force=np.array([0.0, 0.0, 0.0]),
        end_force=np.array([0.0, 0.5, -0.3]),
        ramp_up_time=1.0,
    )

    sim.finalize()
    sim.run()

    time = sim.diagnostic_data["time"]
    position = sim.diagnostic_data["position"]
    directors = sim.diagnostic_data["directors"]
    twist_angle = sim.diagnostic_data["twist_angle"]
    rest_lengths = sim.diagnostic_data["rest_lengths"]

    plot_twist_angle(time, twist_angle)
    plot_elongation(time, rest_lengths)
    plot_distal_end_position(time, position)
    plot_distal_end_orientation(time, directors)


def plot_distal_end_position(_time: np.ndarray, _position: np.ndarray):
    plt.figure(num="Distal end position")
    plt.plot(_time, _position[:, 0, -1], label=r"$x_{x}$")
    plt.plot(_time, _position[:, 1, -1], label=r"$x_{y}$")
    plt.plot(_time, _position[:, 2, -1], label=r"$x_{z}$")
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$x$ [m]")
    plt.show()


def plot_distal_end_orientation(_time: np.ndarray, _directors: np.ndarray):
    plt.figure(num="Distal end orientation")
    rotation_matrices = _directors[:, ..., -1].transpose(0, 2, 1)
    euler_angles = Rotation.from_matrix(rotation_matrices).as_euler("xyz", degrees=True)
    plt.plot(_time, euler_angles[:, 0], label=r"roll")
    plt.plot(_time, euler_angles[:, 1], label=r"pitch")
    plt.plot(_time, euler_angles[:, 2], label=r"yaw")
    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"Euler angles [deg]")
    plt.show()


def plot_twist_angle(_time: np.ndarray, _twist_angle: np.ndarray):
    # plot twist angles vs. time
    plt.figure(num="twist angles vs. time")
    for link_idx in range(_twist_angle.shape[1]):
        plt.plot(
            _time,
            _twist_angle[:, link_idx] / np.pi * 180,
            label=r"$\theta_{" + str(link_idx) + "}$",
        )
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$\theta$ [deg]")
    plt.legend()
    plt.title(r"$\theta$ vs. $t$")
    plt.show()


def plot_elongation(_time: np.ndarray, _rest_lengths: np.ndarray):
    # time vs. elongation
    plt.figure(num="elongation vs. time")
    rest_length_cumsum = np.cumsum(_rest_lengths, axis=-1)
    for link_idx in range(_rest_lengths.shape[1]):
        plt.plot(
            _time,
            rest_length_cumsum[:, link_idx] - rest_length_cumsum[0, link_idx],
        )
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$\delta z$ [m]")
    plt.title(r"$\delta z$ vs. $t$")
    plt.show()


if __name__ == "__main__":
    platform_fixed_constraint_test()
