"""
Verify whether the simulated HSA can exhibit the same system behavior as characterized in:
Good, Ian, et al. "Expanding the Design Space for Electrically-Driven Soft Robots Through Handed Shearing Auxetics."
2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022.
"""
from elastica.external_forces import NoForces
from elastica.typing import SystemType
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle

from hsa_elastica.forcing import ProximalEndTorsion
from hsa_elastica.simulation import HsaRodSimulator

from examples.parameters.rod_params import (
    ROD_FPU50_CLOSED_4ROWS_PARAMS,
    ROD_FPU50_CLOSED_6ROWS_PARAMS,
    ROD_FPU50_CLOSED_8ROWS_PARAMS,
    ROD_FPU50_CLOSED_10ROWS_PARAMS,
    ROD_FPU50_CLOSED_12ROWS_PARAMS,
)


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

# define some parameters
rod_params = ROD_FPU50_CLOSED_6ROWS_PARAMS
desired_twist_angles = np.array([0, 30, 60, 90, 120, 150]) / 180 * np.pi


def verify_holding_torque(
    plot_characteristics: bool = False, plot_time_response: bool = False
):
    """
    Verify the holding torque (e.g. motor torque at base of rod) for a given rotation of the base
    to hold steady-state
    """
    torsional_torques = np.linspace(
        start=0.0,
        stop=rod_params["max_actuation_torque"],
        num=len(desired_twist_angles),
    )

    final_twist_angles = []
    final_holding_torques = []
    for it, torsional_torque in enumerate(torsional_torques):
        print(f"Iteration {it} with torsional torque {torsional_torque} Nm")

        sim = HsaRodSimulator(rod_params=rod_params, duration=10.0)
        # constrain extension of
        sim.configure(finalize=False, constrain_extension=False)

        # add ramp actuation torsional_torque
        sim.add_forcing_to(sim.rod).using(
            ProximalEndTorsion,
            torsional_torque=torsional_torque,
            ramp_up_time=0.0,
        )

        sim.finalize()
        sim.run()

        time = sim.diagnostic_data["time"]
        position = sim.diagnostic_data["position"]
        twist_angle = sim.diagnostic_data["twist_angle"]

        final_twist_angle = twist_angle[-1, 0]
        final_holding_torque = torsional_torque
        print(
            f"Final theta {final_twist_angle / np.pi * 180} deg, and holding torque {final_holding_torque * 1e3} Nmm"
        )

        final_twist_angles.append(final_twist_angle)
        final_holding_torques.append(final_holding_torque)

        if plot_time_response:
            # plot twist angles
            plt.figure(num="Holding torque: twist angles vs. time")
            for link_idx in range(twist_angle.shape[1]):
                plt.plot(
                    time,
                    twist_angle[:, link_idx] / np.pi * 180,
                    label=r"$\theta_{" + str(link_idx) + "}$",
                )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$\theta$ [deg]")
            plt.legend()
            plt.title(
                r"$\theta$ vs. $t$ for $\tau_z = " + str(torsional_torque) + r"$ Nm"
            )
            plt.show()

    final_twist_angles = np.array(final_twist_angles)
    final_holding_torques = np.array(final_holding_torques)

    if plot_characteristics:
        # plot blocked axial force vs. twist angle
        plt.figure(num="Holding torque: holding torque vs. twist angle")
        plt.plot(
            final_twist_angles / np.pi * 180,
            final_holding_torques * 1e3,
            linestyle="-",
            marker=".",
            color="red",
        )
        plt.xlabel(r"$\theta$ [deg]")
        plt.ylabel(r"$\tau_z$ [Nmm]")
        plt.title("holding torque vs. twist angle")
        plt.show()

    holding_torque_data = {
        "twist_angle": final_twist_angles,
        "holding_torque": final_holding_torques,
    }
    return holding_torque_data


def verify_minimum_energy_length(
    plot_characteristics: bool = False, plot_time_response: bool = False
):
    """
    Verify the minimum energy length for a given rotation of the base (e.g. a given torsional torque)
    We find the length by letting the system approach steady state
    This implementation tests out discretized torsional torques and records the steady state response.
    """
    final_twist_angles = []
    final_extensions = []
    final_lengths = []
    for it, desired_twist_angle in enumerate(desired_twist_angles):
        print(
            f"Iteration {it} with desired twist angle {desired_twist_angle / np.pi * 180} deg"
        )

        sim = HsaRodSimulator(rod_params=rod_params, duration=10.0)
        sim.configure(
            finalize=False,
            constrain_extension=False,
            actuation_angle=desired_twist_angle,
        )

        sim.finalize()
        sim.run()

        time = sim.diagnostic_data["time"]
        position = sim.diagnostic_data["position"]
        twist_angle = sim.diagnostic_data["twist_angle"]

        final_directors = sim.diagnostic_data["directors"][-1, ...]

        # final desired twist angle (Figure 4): 150 deg
        final_twist_angle = twist_angle[-1, 0]
        final_extension = position[-1, 2, -1] - sim.printed_length
        final_length = sim.printed_length + final_extension
        print(
            f"Final theta {final_twist_angle / np.pi * 180} deg, extension {final_extension * 1e3} mm, "
            f"and length {final_length *  1e3} mm"
        )

        final_twist_angles.append(final_twist_angle)
        final_extensions.append(final_extension)
        final_lengths.append(final_length)

        if plot_time_response:
            # plot twist angles
            plt.figure(num="Minimum energy length: twist angles vs. time")
            for link_idx in range(twist_angle.shape[1]):
                plt.plot(
                    time,
                    twist_angle[:, link_idx] / np.pi * 180,
                    label=r"$\theta_{" + str(link_idx) + "}$",
                )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$\theta$ [deg]")
            plt.legend()
            plt.title(
                r"$\theta$ vs. $t$ for $\theta^* = "
                + str(desired_twist_angle / np.pi * 180)
                + r"$ deg"
            )
            plt.show()

            # plot extension
            plt.figure(num="Minimum energy length: extension vs. time")
            for node_idx in range(position.shape[2]):
                plt.plot(
                    time,
                    position[:, 2, node_idx]
                    - node_idx / (position.shape[2] - 1) * sim.printed_length,
                    label=r"$\delta z_{" + str(node_idx) + "}$",
                )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$\delta z$ [m]")
            plt.legend()
            plt.title(
                r"$\delta z$ vs. $t$ for $\theta^* = "
                + str(desired_twist_angle / np.pi * 180)
                + "$ deg"
            )
            plt.show()

            # extension vs. twist angle
            plt.figure(num="Minimum energy length: extension vs. twist angle")
            plt.plot(
                twist_angle[:, 0] / np.pi * 180,
                (position[:, 2, -1] - sim.printed_length) * 1e3,
                color="red",
            )
            plt.xlabel(r"$\theta$ [deg]")
            plt.ylabel(r"$\delta z$ [mm]")
            plt.title(
                r"$\delta z$ vs. $\theta$ for $\theta^* = "
                + str(desired_twist_angle / np.pi * 180)
                + "$ deg"
            )
            plt.show()

    final_twist_angles = np.array(final_twist_angles)
    final_extensions = np.array(final_extensions)
    final_lengths = np.array(final_lengths)

    if plot_characteristics:
        # plot length vs. twist angle
        plt.figure(num="Minimum energy length: length vs. twist angle")
        plt.plot(
            final_twist_angles / np.pi * 180,
            final_lengths * 1e3,
            linestyle="-",
            marker=".",
            color="red",
        )
        plt.xlabel(r"$\theta$ [deg]")
        plt.ylabel(r"$L$ [mm]")
        plt.title("length vs. twist angle")
        plt.show()

    minimum_energy_length_data = {
        "twist_angle": final_twist_angles,
        "minimum_energy_length": final_lengths,
    }
    return minimum_energy_length_data


def verify_spring_constant(
    plot_characteristics: bool = False, plot_time_response: bool = False
):
    """
    Verify the minimum energy length for a given rotation of the base (e.g. a given torsional torque)
    We find the length by letting the system approach steady state
    This implementation tests out discretized torsional torques and records the steady state response.
    """
    twist_inertial_delay = 10  # [s]
    duration = 20.0

    final_twist_angles = []
    final_displacements = []
    final_spring_constants = []
    for it, desired_twist_angle in enumerate(desired_twist_angles):
        print(
            f"Iteration {it} with desired twist angle {desired_twist_angle / np.pi * 180} deg"
        )

        sim = HsaRodSimulator(rod_params=rod_params, duration=duration)
        sim.configure(
            finalize=False,
            constrain_extension=False,
            actuation_angle=desired_twist_angle,
        )

        # add delayed axial axial_force
        sim.add_forcing_to(sim.rod).using(
            DelayedAxialEndpointForce,
            axial_force=1e0,
            delay=twist_inertial_delay,
        )

        sim.finalize()
        sim.run()

        time = sim.diagnostic_data["time"]
        position = sim.diagnostic_data["position"]
        twist_angle = sim.diagnostic_data["twist_angle"]
        rest_lengths = sim.diagnostic_data["rest_lengths"]
        sigma = sim.diagnostic_data["sigma"]
        kappa = sim.diagnostic_data["kappa"]
        internal_forces = sim.diagnostic_data["internal_forces"]

        final_twist_angle = twist_angle[-1, 0]
        final_displacement = position[-1, 2, -1] - rest_lengths[-1, :].sum()
        final_axial_force = -internal_forces[-1, 2, -1]
        final_spring_constant = final_axial_force / final_displacement
        print(
            f"Final twist angle {final_twist_angle / np.pi * 180} deg, displacement {final_displacement * 1e3} mm, "
            f"axial force {final_axial_force} N, and spring constant {final_spring_constant * 1e-3} N / mm"
        )

        final_twist_angles.append(final_twist_angle)
        final_displacements.append(final_displacement)
        final_spring_constants.append(final_spring_constant)

        if plot_time_response:
            # plot twist angles vs. time
            plt.figure(num="Spring constant: twist angles vs. time")
            for link_idx in range(twist_angle.shape[1]):
                plt.plot(
                    time,
                    twist_angle[:, link_idx] / np.pi * 180,
                    label=r"$\theta_{" + str(link_idx) + "}$",
                )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$\theta$ [deg]")
            plt.legend()
            plt.title(
                r"$\theta$ vs. $t$ for $\theta^* = "
                + str(desired_twist_angle / np.pi * 180)
                + r"$ deg"
            )
            plt.show()

            # plot twist strain vs. time
            plt.figure(num="Spring constant: twist vs. time")
            for link_idx in range(kappa.shape[2]):
                plt.plot(
                    time,
                    kappa[:, 2, link_idx],
                    label=r"$\kappa_{z," + str(link_idx) + "}$",
                )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$\kappa_z$ [-]")
            plt.legend()
            plt.title(
                r"$\kappa_z$ vs. $t$ for $\theta^* = "
                + str(desired_twist_angle / np.pi * 180)
                + "$ deg"
            )
            plt.show()

            # plot stretch vs. time
            plt.figure(num="Spring constant: stretch vs. time")
            for node_idx in range(sigma.shape[2]):
                plt.plot(
                    time,
                    sigma[:, 2, node_idx],
                    label=r"$\sigma_{z," + str(node_idx) + "}$",
                )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$\sigma_z$ [-]")
            plt.legend()
            plt.title(
                r"$\sigma_z$ vs. $t$ for $\theta^* = "
                + str(desired_twist_angle / np.pi * 180)
                + "$ deg"
            )
            plt.show()

            # plot axial force vs. time
            plt.figure(num="Spring constant: axial force vs. time")
            plt.plot(
                time,
                -internal_forces[:, 2, -1],
                color="red",
            )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$f_z$ [N]")
            plt.title(
                r"$f_z$ vs. $t$ for $\theta^* = "
                + str(desired_twist_angle / np.pi * 180)
                + "$ deg"
            )
            plt.show()

            # axial force vs. displacement
            plt.figure(num="Spring constant: axial force vs. elongation")
            plt.plot(
                position[:, 2, -1] - rest_lengths[:, :].sum(),
                -internal_forces[:, 2, -1],
                color="red",
            )
            plt.xlabel(r"$\delta z$ [m]")
            plt.ylabel(r"$f_z$ [N]")
            plt.title(
                r"$\delta z$ vs. $\sigma_z$ for $\theta^* = "
                + str(desired_twist_angle)
                + "$ deg"
            )
            plt.show()

    final_twist_angles = np.array(final_twist_angles)
    final_displacements = np.array(final_displacements)
    final_spring_constants = np.array(final_spring_constants)

    if plot_characteristics:
        # plot spring constant vs. twist angle
        plt.figure(num="Spring constant: spring constant vs. twist angle")
        plt.plot(
            final_twist_angles / np.pi * 180,
            final_spring_constants * 1e-3,
            linestyle="-",
            marker=".",
            color="red",
        )
        plt.xlabel(r"$\theta$ [deg]")
        plt.ylabel(r"$k$ [N / mm]")
        plt.title("spring constant vs. twist angle")
        plt.show()

    spring_constant_data = {
        "twist_angle": final_twist_angles,
        "spring_constant": final_spring_constants,
    }
    return spring_constant_data


class DelayedAxialEndpointForce(NoForces):
    def __init__(self, axial_force: float, delay: float):
        super().__init__()

        self.axial_force = axial_force
        self.delay = delay

    def apply_forces(self, system: SystemType, time: np.float64 = 0.0):
        if time >= self.delay:
            force_local_frame = np.array([0, 0, self.axial_force])
            system.external_forces[2, -1] += (
                system.director_collection[..., -1].T @ force_local_frame
            )[2]


def verify_blocked_force(
    plot_characteristics: bool = False, plot_time_response: bool = False
):
    """
    Verify the blocked axial_force for a given rotation of the base (e.g. a given torsional torque)
    while constraining the elongation of the rod at steady-state.
    """

    final_twist_angles = []
    final_blocked_forces = []
    for it, desired_twist_angle in enumerate(desired_twist_angles):
        print(
            f"Iteration {it} with desired twist angle {desired_twist_angle / np.pi * 180} deg"
        )

        sim = HsaRodSimulator(rod_params=rod_params, duration=10.0)
        sim.configure(
            finalize=False,
            constrain_extension=True,
            actuation_angle=desired_twist_angle,
        )

        sim.finalize()
        sim.run()

        time = sim.diagnostic_data["time"]
        position = sim.diagnostic_data["position"]
        twist_angle = sim.diagnostic_data["twist_angle"]
        internal_forces = sim.diagnostic_data["internal_forces"]

        final_twist_angle = twist_angle[-1, 0]
        final_blocked_force = internal_forces[-1, 2, -1]
        print(
            f"Final theta {final_twist_angle / np.pi * 180} deg, and blocked axial force {final_blocked_force} N"
        )

        final_twist_angles.append(final_twist_angle)
        final_blocked_forces.append(final_blocked_force)

        if plot_time_response:
            # plot twist angles
            plt.figure(num="Blocked force: twist angles vs. time")
            for link_idx in range(twist_angle.shape[1]):
                plt.plot(
                    time,
                    twist_angle[:, link_idx] / np.pi * 180,
                    label=r"$\theta_{" + str(link_idx) + "}$",
                )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$\theta$ [deg]")
            plt.legend()
            plt.title(
                r"$\theta$ vs. $t$ for $\theta^* = "
                + str(desired_twist_angle / np.pi * 180)
                + r"$ deg"
            )
            plt.show()

            # plot blocked axial_force
            plt.figure(num="Blocked force: blocked axial force vs. time")
            plt.plot(
                time,
                internal_forces[:, 2, -1],
                color="red",
            )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$f_z$ [N]")
            plt.title(
                r"$f_z$ vs. $t$ for $\theta^* = "
                + str(desired_twist_angle / np.pi * 180)
                + "$ deg"
            )
            plt.show()

            # blocked axial_force vs. twist angle
            plt.figure(num="Blocked force: blocked axial force vs. twist angle")
            plt.plot(
                twist_angle[:, 0] / np.pi * 180,
                internal_forces[:, 2, -1],
                color="red",
            )
            plt.xlabel(r"$\theta$ [s]")
            plt.ylabel(r"$f_z$ [N]")
            plt.title(
                r"$f_z$ vs. $\theta$ for $\theta^* = "
                + str(desired_twist_angle / np.pi * 180)
                + "$ deg"
            )
            plt.show()

    final_twist_angles = np.array(final_twist_angles)
    final_blocked_forces = np.array(final_blocked_forces)

    if plot_characteristics:
        # plot blocked axial_force vs. twist angle
        plt.figure(num="Blocked force: blocked axial force vs. twist angle")
        plt.plot(
            final_twist_angles / np.pi * 180,
            final_blocked_forces,
            linestyle="-",
            marker=".",
            color="red",
        )
        plt.xlabel(r"$\theta$ [deg]")
        plt.ylabel(r"$f_z$ [N]")
        plt.title("blocked axial force vs. twist angle")
        plt.show()

    blocked_force_data = {
        "twist_angle": final_twist_angles,
        "blocked_force": final_blocked_forces,
    }
    return blocked_force_data


if __name__ == "__main__":
    rod_data = {}
    rod_data["holding_torque_data"] = verify_holding_torque(
        plot_characteristics=False, plot_time_response=False
    )
    rod_data["minimum_energy_length_data"] = verify_minimum_energy_length(
        plot_characteristics=False
    )
    rod_data["spring_constant_data"] = verify_spring_constant(
        plot_characteristics=False
    )
    rod_data["blocked_force_data"] = verify_blocked_force(plot_characteristics=False)

    filepath = (
        pathlib.Path(__file__).parent.resolve()
        / "simulated_data"
        / f"{rod_params['name']}.pickle"
    )
    with open(str(filepath), "wb") as f:
        pickle.dump(rod_data, f)

    print("Saved simulated rod data to ", filepath)
