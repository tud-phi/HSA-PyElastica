from elastica.boundary_conditions import GeneralConstraint
from elastica.typing import SystemType
from elastica.utils import Tolerance
import numpy as np
from scipy.spatial.transform import Rotation

from .hsa_motor_free_rot_bc import HsaMotorFreeRotBC


class HsaMotorControlledRotBC(HsaMotorFreeRotBC):
    """
    Class for defining the boundary condition of proximal end of the rod where it is connected to the electric motor
    Controls the motor by constraining the rotation of the proximal end of the rod it to the specified actuation angle
    """

    def __init__(
        self, *args, actuation_angle: float, ramp_up_time: float = 0.0, **kwargs
    ):
        super().__init__(*args, constrain_yaw=True, **kwargs)

        self.actuation_angle = actuation_angle
        self.ramp_up_time = ramp_up_time

    def constrain_values(self, system: SystemType, time: float) -> None:
        super().constrain_values(system, time)

        # factor to linearly scale the rotation of the motor towards the actuation angle using ramp up time
        factor = min(1.0, time / (self.ramp_up_time + Tolerance.atol()))
        # construct the rotation matrix from Euler angles
        rot_mat = Rotation.from_rotvec(
            np.array([0.0, 0.0, factor * self.actuation_angle])
        ).as_matrix()
        # update the director of the system with the transposed rotation matrix
        system.director_collection[..., 0] = rot_mat.T
