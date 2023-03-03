"""
Verify whether the simulated HSA can exhibit the same system behavior as characterized in:
Good, Ian, et al. "Expanding the Design Space for Electrically-Driven Soft Robots Through Handed Shearing Auxetics."
2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022.
"""
from elastica.external_forces import NoForces
from elastica.typing import SystemType
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from hsa_elastica.forcing import ProximalEndTorsion
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


def verify_torsion_to_twist_with_extension(plot_time_response: bool = False):
    """
    Verify that torsion (torque) to twist (strain) ratio remains constant when the rod is extended.
    """
    rod_dilations = np.linspace(start=1.0, stop=1.2, num=3)
    final_dilations = []
    final_torsions = []
    final_twist_angles = []
    for it, rod_dilation in enumerate(rod_dilations):
        print(f"Iteration {it} with rod dilation {rod_dilation} %")

        sim = HsaRodSimulator(rod_params=rod_params, duration=10.0)
        # constrain extension of
        sim.configure(
            finalize=False, constrain_extension=False, follow_auxetic_trajectory=False
        )

        # add ramp actuation torsional_torque
        sim.add_forcing_to(sim.rod).using(
            ProximalEndTorsion,
            torsional_torque=rod_params["max_actuation_torque"],
            ramp_up_time=0.0,
        )

        rod = sim.rod

        # Compute the desired rod length by adhering to the desired dilation
        desired_rest_lengths = rod_dilation * rod.printed_lengths

        # set rest lengths of rod (e.g. equilibrium state of elastic material) to the desired rod lengths
        rod.rest_lengths += desired_rest_lengths - rod.rest_lengths

        # update rest voronoi lengths
        sim.rod.rest_voronoi_lengths += (
            0.5 * (rod.rest_lengths[1:] + rod.rest_lengths[:-1])
            - rod.rest_voronoi_lengths
        )

        sim.finalize()
        sim.run()

        time = sim.diagnostic_data["time"]
        position = sim.diagnostic_data["position"]
        twist_angle = sim.diagnostic_data["twist_angle"]
        rest_lengths = sim.diagnostic_data["rest_lengths"]
        sigma = sim.diagnostic_data["sigma"]
        internal_forces = sim.diagnostic_data["internal_forces"]

        final_dilations.append(rod_dilation)
        final_torsions.append(rod_params["max_actuation_torque"])
        final_twist_angles.append(twist_angle[-1, 0])

        print(
            f"Final torsion - twist ratio for rod dilation {rod_dilation} % "
            f"is {rod_params['max_actuation_torque'] / twist_angle[-1, 0]} Nm / deg"
        )

        if plot_time_response:
            # plot twist angles
            plt.figure(
                num="Verify torsion to twist with elongation: twist angles vs. time"
            )
            for link_idx in range(twist_angle.shape[1]):
                plt.plot(
                    time,
                    twist_angle[:, link_idx] / np.pi * 180,
                    label=r"$\theta_{" + str(link_idx) + "}$",
                )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$\theta$ [deg]")
            plt.legend()
            plt.title(r"$\theta$ vs. $t$ for $e = " + str(rod_dilation) + r"$ %")
            plt.show()

            # # plot rest lengths
            # plt.figure(
            #     num="Verify torsion to twist with elongation: rest lengths vs. time"
            # )
            # for link_idx in range(rest_lengths.shape[1]):
            #     plt.plot(
            #         time,
            #         rest_lengths[:, link_idx],
            #         label=r"$\hat{l}_{" + str(link_idx) + "}$",
            #     )
            # plt.xlabel(r"$t$ [s]")
            # plt.ylabel(r"$\hat{l}$ [m]")
            # plt.legend()
            # plt.title(r"$\hat{l}$ vs. $t$ for $e = " + str(rod_dilation) + r"$ %")
            # plt.show()
            #
            # # plot sigma_z
            # plt.figure(
            #     num="Verify torsion to twist with elongation: sigma vs. time"
            # )
            # for link_idx in range(sigma.shape[2]):
            #     plt.plot(
            #         time,
            #         sigma[:, 2, link_idx],
            #         label=r"$\sigma_{z," + str(link_idx) + "}$",
            #     )
            # plt.xlabel(r"$t$ [s]")
            # plt.ylabel(r"$\sigma_z$ [-]")
            # plt.legend()
            # plt.title(r"$\sigma_z$ vs. $t$ for $e = " + str(rod_dilation) + r"$ %")
            # plt.show()

            # plot node positions
            plt.figure(
                num="Verify torsion to twist with elongation: node positions vs. time"
            )
            for node_idx in range(position.shape[2]):
                plt.plot(
                    time,
                    position[:, 2, node_idx],
                    label=r"$x_{z," + str(node_idx) + "}$",
                )
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$x_z$ [m]")
            plt.legend()
            plt.title(r"$x_z$ vs. $t$ for $e = " + str(rod_dilation) + r"$ %")
            plt.show()

    final_dilations = np.array(final_dilations)
    final_torsions = np.array(final_torsions)
    final_twist_angles = np.array(final_twist_angles)

    # plot torque to twist ratio
    plt.figure(num="Verify torsion to twist with elongation")
    plt.plot(
        final_dilations * 100,
        final_torsions / final_twist_angles,
        linestyle="-",
        marker=".",
        color="red",
    )
    plt.xlabel(r"$e$ [\%]")
    plt.ylabel(r"$\frac{\tau_z}{\theta}$ [Nm / rad]")
    plt.title("torsion - twist ratio")
    plt.show()


def verify_holding_torque(plot_time_response: bool = False):
    """
    Verify the holding torque (e.g. motor torque at base of rod) for a given rotation of the base
    to hold steady-state
    """
    torsional_torques = np.linspace(
        start=0.0, stop=rod_params["max_actuation_torque"], num=3
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
            f"Final theta {final_twist_angle} deg, and holding torque {final_holding_torque} Nm"
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


def verify_minimum_energy_length(plot_time_response: bool = False):
    """
    Verify the minimum energy length for a given rotation of the base (e.g. a given torsional torque)
    We find the length by letting the system approach steady state
    This implementation tests out discretized torsional torques and records the steady state response.
    """
    torsional_torques = np.linspace(
        start=0.0, stop=rod_params["max_actuation_torque"], num=3
    )

    final_twist_angles = []
    final_extensions = []
    final_lengths = []
    for it, torsional_torque in enumerate(torsional_torques):
        if it != 2:
            continue
        print(f"Iteration {it} with torsional torque {torsional_torque} Nm")

        sim = HsaRodSimulator(rod_params=rod_params, duration=10.0)
        # constrain extension of
        sim.configure(finalize=False)

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

        # final desired twist angle (Figure 4): 150 deg
        final_twist_angle = twist_angle[-1, 0]
        final_extension = position[-1, 2, -1] - sim.printed_length
        final_length = sim.printed_length + final_extension
        print(
            f"Final theta {final_twist_angle} deg, extension {final_extension} m, and length {final_length} m"
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
                r"$\theta$ vs. $t$ for $\tau_z = " + str(torsional_torque) + r"$ Nm"
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
                r"$\delta z$ vs. $t$ for $\tau_z = " + str(torsional_torque) + "$ Nm"
            )
            plt.show()

            # extension vs. twist angle
            plt.figure(num="Minimum energy length: extension vs. twist angle")
            plt.plot(
                twist_angle[:, 0] / np.pi * 180,
                position[:, 2, -1] - sim.printed_length,
                color="red",
            )
            plt.xlabel(r"$\theta$ [s]")
            plt.ylabel(r"$\delta z$ [m]")
            plt.title(
                r"$\delta z$ vs. $\theta$ for $\tau_z = "
                + str(torsional_torque)
                + "$ Nm"
            )
            plt.show()

    final_twist_angles = np.array(final_twist_angles)
    final_extensions = np.array(final_extensions)
    final_lengths = np.array(final_lengths)

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


def verify_spring_constant(plot_time_response: bool = False):
    """
    Verify the minimum energy length for a given rotation of the base (e.g. a given torsional torque)
    We find the length by letting the system approach steady state
    This implementation tests out discretized torsional torques and records the steady state response.
    """
    twist_inertial_delay = 10  # [s]
    duration = 20.0

    torsional_torques = np.linspace(
        start=0.0, stop=rod_params["max_actuation_torque"], num=5
    )

    final_twist_angles = []
    final_displacements = []
    final_spring_constants = []
    for it, torsional_torque in enumerate(torsional_torques):
        print(f"Iteration {it} with torsional torque {torsional_torque} Nm")

        sim = HsaRodSimulator(rod_params=rod_params, duration=duration)
        # constrain extension of
        sim.configure(finalize=False)

        # add ramp actuation torsional_torque
        sim.add_forcing_to(sim.rod).using(
            ProximalEndTorsion,
            torsional_torque=torsional_torque,
            ramp_up_time=0.0,
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
            f"Final theta {final_twist_angle} deg, displacement {final_displacement} m, "
            f"axial force {final_axial_force} N, and spring constant {final_spring_constant} N / m"
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
                r"$\theta$ vs. $t$ for $\tau_z = " + str(torsional_torque) + r"$ Nm"
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
                r"$\kappa_z$ vs. $t$ for $\tau_z = " + str(torsional_torque) + "$ Nm"
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
                r"$\sigma_z$ vs. $t$ for $\tau_z = " + str(torsional_torque) + "$ Nm"
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
            plt.title(r"$f_z$ vs. $t$ for $\tau_z = " + str(torsional_torque) + "$ Nm")
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
                r"$\delta z$ vs. $\sigma_z$ for $\tau_z = "
                + str(torsional_torque)
                + "$ Nm"
            )
            plt.show()

    final_twist_angles = np.array(final_twist_angles)
    final_displacements = np.array(final_displacements)
    final_spring_constants = np.array(final_spring_constants)

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


def verify_blocked_force(plot_time_response: bool = False):
    """
    Verify the blocked axial_force for a given rotation of the base (e.g. a given torsional torque)
    while constraining the elongation of the rod at steady-state.
    """
    # TODO: Change CosseratRod behaviour so that this increase in max torsion is not necessary
    # Currently, higher torsional torques are necessary to achieve same theta,
    # when the rod is not extending at the same time
    torsional_torques = np.linspace(
        start=0.0, stop=1.36 * rod_params["max_actuation_torque"], num=5
    )

    final_twist_angles = []
    final_blocked_forces = []
    for it, torsional_torque in enumerate(torsional_torques):
        print(f"Iteration {it} with torsional torque {torsional_torque} Nm")

        sim = HsaRodSimulator(rod_params=rod_params, duration=10.0)
        # constrain extension of
        sim.configure(finalize=False, constrain_extension=True)

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
        internal_forces = sim.diagnostic_data["internal_forces"]

        final_twist_angle = twist_angle[-1, 0]
        final_blocked_force = internal_forces[-1, 2, -1]
        print(
            f"Final theta {final_twist_angle} deg, and blocked axial_force {final_blocked_force} N"
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
                r"$\theta$ vs. $t$ for $\tau_z = " + str(torsional_torque) + r"$ Nm"
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
            plt.title(r"$f_z$ vs. $t$ for $\tau_z = " + str(torsional_torque) + "$ Nm")
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
                r"$f_z$ vs. $\theta$ for $\tau_z = " + str(torsional_torque) + "$ Nm"
            )
            plt.show()

    final_twist_angles = np.array(final_twist_angles)
    final_blocked_forces = np.array(final_blocked_forces)

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


if __name__ == "__main__":
    verify_torsion_to_twist_with_extension()
    verify_holding_torque()
    verify_minimum_energy_length()
    verify_spring_constant()
    verify_blocked_force()
