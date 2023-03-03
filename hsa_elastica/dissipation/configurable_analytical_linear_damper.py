from elastica.dissipation import DamperBase
from elastica.typing import SystemType
import numba
import numpy as np


class ConfigurableAnalyticalLinearDamper(DamperBase):
    """
    Configurable analytical linear damper class. Same principle as `AnalyticalLinearDamper`
    but with configurable damping constant for each strain direction.

    Examples
    --------
    How to set analytical linear damper for rod or rigid body:

    >>> simulator.dampen(rod).using(
    ...     ConfigurableAnalyticalLinearDamper,
    ...     translational_damping_coefficient=np.array([0.1, 0.2, 0.3]),
    ...     rotational_damping_coefficient=np.array([0.3, 0.2, 0.1]),
    ...     time_step = 1E-4,   # Simulation time-step
    ... )


    Attributes
    ----------
    translational_damping_coefficient: numpy.ndarray
        1D array of shape(3, ) containing data with 'float' type.
        Damping coefficient acting on translational velocity.
    rotational_damping_coefficient : numpy.ndarray
        1D array of shape (3, ) containing data with 'float' type.
        Damping coefficient acting on rotational velocity.
    """

    def __init__(
        self,
        translational_damping_constant: np.ndarray,
        rotational_damping_constant: np.ndarray,
        time_step,
        **kwargs,
    ):
        """
        Analytical linear damper initializer

        Parameters
        ----------
        translational_damping_constant : np.ndarray of shape (3,)
            Damping constant for translational velocities
        rotational_damping_constant : np.ndarray of shape (3,)
            Damping constant for rotational velocities
        time_step : float
            Time-step of simulation
        """
        super().__init__(**kwargs)
        # Compute the damping coefficient for translational velocity
        nodal_mass = self._system.mass
        self.translational_damping_coefficient = np.exp(
            -translational_damping_constant * time_step
        )

        # Compute the damping coefficient for exponential velocity
        element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
        element_mass[0] += 0.5 * nodal_mass[0]
        element_mass[-1] += 0.5 * nodal_mass[-1]
        rotational_damping_coefficient = np.einsum(
            "i,ik->ik",
            -rotational_damping_constant,
            np.diagonal(self._system.inv_mass_second_moment_of_inertia).T,
        )
        self.rotational_damping_coefficient = np.exp(
            time_step * element_mass * rotational_damping_coefficient
        )

    def dampen_rates(self, rod: SystemType, time: float):
        _dampen_rates(
            self.translational_damping_coefficient,
            self.rotational_damping_coefficient,
            rod.velocity_collection,
            rod.omega_collection,
            rod.dilatation,
        )


@numba.njit(cache=True)
def _dampen_rates(
    translational_damping_coefficient: np.ndarray,
    rotational_damping_coefficient: np.ndarray,
    velocity_collection: np.ndarray,
    omega_collection: np.ndarray,
    dilatation: np.ndarray,
):
    """
    Numba accelerated function to dampen the rates of the system

    Parameters
    ----------
    velocity_collection : numpy.ndarray
        2D array containing data with 'float' type.
        Collection of translational velocities of the system.
    omega_collection : numpy.ndarray
        2D array containing data with 'float' type.
        Collection of rotational velocities of the system.
    translational_damping_coefficient : numpy.ndarray
        1D array containing data with 'float' type.
        Damping coefficient acting on translational velocity.
    rotational_damping_coefficient : numpy.ndarray
        1D array containing data with 'float' type.
        Damping coefficient acting on rotational velocity.
    dilatation : numpy.ndarray
        1D array containing data with 'float' type.
        Collection of dilatation of the system.
    """
    for i in range(velocity_collection.shape[0]):
        for k in range(velocity_collection.shape[1]):
            velocity_collection[i, k] *= translational_damping_coefficient[i]

    for i in range(omega_collection.shape[0]):
        for k in range(omega_collection.shape[1]):
            omega_collection[i, k] *= np.power(
                rotational_damping_coefficient[i, k], dilatation[k]
            )
