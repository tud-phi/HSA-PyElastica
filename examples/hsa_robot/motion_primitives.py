from hsa_elastica import HsaRobotSimulator
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

from examples.hsa_robot.actuation_utils import (
    platform_configuration_to_actuation_angles,
)
from examples.parameters.robot_params import ONE_SEGMENT_ROBOT


# possible modes:
# "elongation",
# "bending-north", "bending-south", "bending-west", "bending-east"
# "twisting-cw", "twisting-ccw"
MODE = "elongation"
max_actuation_angle = 179.9 / 180 * np.pi  # [rad]


if __name__ == "__main__":
    robot_params = ONE_SEGMENT_ROBOT
    sim = HsaRobotSimulator(
        name=f"motion_primitives_{MODE}",
        robot_params=robot_params,
        duration=15.0,
        dt=4e-5,
        fps=100,
    )
    sim.configure(finalize=False, add_constraints=False)

    max_platform_magnitude = 4 * max_actuation_angle
    match MODE:
        case "elongation":
            q_p_des = 1.0 * max_platform_magnitude * np.array([0.0, 0.0, 0.0, 1.0])
        case "bending-north":
            q_p_des = 1.0 * max_platform_magnitude * np.array([0.0, 0.5, 0.0, 0.5])
        case "bending-south":
            q_p_des = 1.0 * max_platform_magnitude * np.array([0.0, -0.5, 0.0, 0.5])
        case "bending-west":
            q_p_des = 1.0 * max_platform_magnitude * np.array([-0.5, 0.0, 0.0, 0.5])
        case "bending-east":
            q_p_des = 1.0 * max_platform_magnitude * np.array([0.5, 0.0, 0.0, 0.5])
        case "twisting-cw":
            q_p_des = 1.0 * max_platform_magnitude * np.array([0.0, 0.0, -0.5, 0.5])
        case "twisting-ccw":
            q_p_des = 1.0 * max_platform_magnitude * np.array([0.0, 0.0, 0.5, 0.5])
        case _:
            raise NotImplementedError(f"Mode {MODE} not implemented")

    actuation_angles = platform_configuration_to_actuation_angles(
        q_p_des, max_actuation_angle
    )

    print(f"Applying actuation angles: {actuation_angles / np.pi * 180} deg")

    sim.add_controlled_boundary_conditions(
        # add segment-dimension to actuation angles
        actuation_angles=np.expand_dims(actuation_angles, axis=0),
        ramp_up_time=5.0,
    )

    sim.finalize()  # manually finalize the simulator
    sim.run()
    sim.save_diagnostic_data()
