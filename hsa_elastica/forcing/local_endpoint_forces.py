from elastica.external_forces import NoForces
from elastica._linalg import _batch_matvec
import numpy as np
from numba import njit


class LocalEndpointForces(NoForces):
    """
    This class applies constant forces on the endpoint nodes.

        Attributes
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to first node of the rod-like object.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to last node of the rod-like object.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

    """

    def __init__(self, start_force, end_force, ramp_up_time):
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to first node of the rod-like object.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the rod-like object.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        super(LocalEndpointForces, self).__init__()
        self.start_force = start_force
        self.end_force = end_force
        assert ramp_up_time > 0.0
        self.ramp_up_time = ramp_up_time

    def apply_forces(self, system, time=0.0):
        factor = min(1.0, time / self.ramp_up_time)
        # rotate the local forces to global frame
        system.external_forces[..., 0] += np.dot(
            system.director_collection[..., 0].T, self.start_force * factor
        )
        system.external_forces[..., -1] += np.dot(
            system.director_collection[..., -1].T, self.end_force * factor
        )
