from elastica.external_forces import NoForces
from elastica.utils import Tolerance
import numpy as np


class ProximalEndTorsion(NoForces):
    """
    This class applies a constant torque to the proximal end (e.g. base) of a rod.

        Attributes
        ----------
        torque: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Total torque applied to a rod-like object.

    """

    def __init__(self, torsional_torque: float, ramp_up_time: float = 0.0):
        """

        Parameters
        ----------
        torsional_torque: float
            Torque magnitude applied to the proximal end of a rod-like object.
        ramp_up_time: float
            Applied torques are ramped up until ramp up time.
        """
        super(ProximalEndTorsion, self).__init__()
        self.torque = torsional_torque * np.array([0, 0, 1])
        self.ramp_up_time = ramp_up_time

    def apply_torques(self, system, time: np.float64 = 0.0):
        factor = min(1.0, time / (self.ramp_up_time + Tolerance.atol()))
        system.external_torques[:, 0] += np.dot(
            system.director_collection[..., 0], factor * self.torque
        )
