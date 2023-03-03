from elastica.boundary_conditions import GeneralConstraint
from elastica.typing import SystemType
import numpy as np
from scipy.spatial.transform import Rotation


class HsaMotorFreeRotBC(GeneralConstraint):
    """
    Class for defining the boundary condition of proximal end of the rod where it is connected to the electric motor
    Allows for free rotation of the motor
    """

    def __init__(self, *args, constrain_yaw: bool = False, **kwargs):
        super().__init__(
            *args,
            translational_constraint_selector=np.array(
                [True, True, True]
            ),  # Block all translational DoF
            rotational_constraint_selector=np.array(
                [True, True, constrain_yaw]
            ),  # Allow for yaw
            **kwargs,
        )

        # saves the current director of the proximal node
        self._director = None

        # keeps track of the rotation of the motor [rad]
        self.rotation_angle: float = 0.0

    def constrain_values(self, system: SystemType, time: float) -> None:
        super().constrain_values(system, time)

        if self._director is not None:
            # use the relative rotation between the previous director and the current director
            # to determine the delta in the rotation of the motor
            # delta_{t-1 to t} = R_{t-1 to I} * R_{I to t}
            delta_rot_mat = system.director_collection[..., 0] @ self._director.T
            euler_angles = Rotation.from_matrix(delta_rot_mat).as_euler("xyz")
            self.rotation_angle += euler_angles[2]
        # update the current director
        self._director = system.director_collection[..., 0]
