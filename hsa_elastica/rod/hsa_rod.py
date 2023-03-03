__doc__ = """ HSA rod classes and implementation details """
__all__ = ["HsaRod"]


from elastica.rod import RodBase
from elastica.rod.cosserat_rod import (
    CosseratRod,
    _compute_damping_forces,
    _compute_damping_torques,
    _compute_dilatation_rate,
    _compute_internal_forces,
    _compute_geometry_from_state,
    _compute_shear_stretch_strains,
)
from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from elastica.rod.knot_theory import KnotTheory
from elastica.utils import MaxDimension, Tolerance
from elastica._calculus import (
    quadrature_kernel_for_block_structure,
    difference_kernel_for_block_structure,
    _difference,
    _average,
)
from elastica._rotations import _inv_rotate
import functools
import numba
import numpy as np
import typing

from hsa_elastica.rod.factory_function import allocate

position_difference_kernel = _difference
position_average = _average


@functools.lru_cache(maxsize=1)
def _get_z_vector():
    return np.array([0.0, 0.0, 1.0]).reshape(3, -1)


class HsaRod(CosseratRod, KnotTheory):
    """
    Rod class simulating an HSA rod based on the Cosserat rod model.

        Attributes
        ----------
        n_elems: int
            The number of elements of the rod.
        position_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node position vectors.
        velocity_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node velocity vectors.
        acceleration_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node acceleration vectors.
        omega_collection: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Array containing element angular velocity vectors.
        alpha_collection: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Array contining element angular acceleration vectors.
        director_collection: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Array containing element director matrices.
        density: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod elements densities.
        volume: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element volumes.
        mass: numpy.ndarray
            1D (n_nodes) array containing data with 'float' type.
            Rod node masses. Note that masses are stored on the nodes, not on elements.
        cross_sectional_area: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element cross-sectional area.
        second_moment_of_inertia: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element second moment of inertia (e.g. I_xx, I_yy, I_zz) in the local coordinate frame.
        mass_second_moment_of_inertia: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Rod element mass second moment of inertia.
        inv_mass_second_moment_of_inertia: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Rod element inverse mass moment of inertia.
        elastic_modulus: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Elastic modulii (e.g. Young's modulii) of rod elements in the unit [Pa] without any twist strain.
        elastic_modulus_scale_factor: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Factor to linearly scale the elastic modulus with the (positive) twist strain according to the equation:
                elastic_modulus = elastic_modulus_zero_twist_strain + twist_strain * elastic_modulus_scale_factor
            where the factor is in the unit [Pa m / rad] and the twist strain is clipped to positive values
            according to the handedness of the rod in the unit [rad / m].
        shear_modulus: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Elastic modulii (e.g. Young's modulii) of rod elements in the unit [Pa] without any twist strain.
        shear_modulus_scale_factor: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Factor to linearly scale the shear modulus with the (positive) twist strain according to the equation:
                shear_modulus = shear_modulus_zero_twist_strain + twist_strain * shear_modulus_scale_factor
            where the factor is in the unit [Pa m / rad] and the twist strain is clipped to positive values
            according to the handedness of the rod in the unit [rad / m].
        bend_rigidity: numpy.ndarray
            1D (n_elems - 1) array containing data with 'float' type.
            Bend rigidity between the HSA rod elements in the unit [Nm/(rad/m)] = Nm^2/rad.
            The bend stress in x-dir is computed as:
                internal_couple[0, :] = bend_rigidity * (kappa[0, :] - rest_kappa[0, :])
            where kappa[0, :] are the bending strains in the unit [rad/m].
        bend_rigidity_scale_factor: numpy.ndarray
            1D (n_elems - 1) array containing data with 'float' type.
            Factor to linearly scale the bend_rigidity with the (positive) twist strain according to the equation:
                bend_rigidity = bend_rigidity_zero_twist_strain + twist_strain * bend_rigidity_scale_factor
            where the factor is in the unit [(Nm^2/rad) / (m / rad)] and the twist strain is clipped to positive values
            according to the handedness of the rod in the unit [rad / m].
        twist_rigidity: numpy.ndarray
            1D (n_elems - 1) array containing data with 'float' type.
            Twist rigidity between the HSA rod elements in the unit [Nm/(rad/m)] = Nm^2/rad.
            The twist stress is computed as:
                internal_couple[2, :] = twist_rigidity * (kappa[2, :] - rest_kappa[2, :])
            where kappa[2, :] are the twist strains in the unit [rad/m].
        dissipation_constant_for_forces: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dissipation coefficient (nu).
        dissipation_constant_for_torques: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dissipation (nu).
            Can be customized by passing 'nu_for_torques'.
        internal_forces: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Rod node internal forces. Note that internal forces are stored on the node, not on elements.
        internal_torques: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element internal torques.
        external_forces: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            External forces acting on rod nodes.
        external_torques: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            External torques acting on rod elements.
        lengths: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths.
        rest_lengths: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths at rest configuration.
        rest_lengths_scale_factor: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Scaling factor to map (positive) twist strain to an extension of the rest lengths to mirror the auxetic
            system behaviour. After the handedness of the twist strain is considered and the strain is clipped to
            positive values, we compute the modified rest lengths as:
                rest_lengths = printed_lengths * (1 + rest_lengths_scale_factor * positive_twist_strain)
        printed_lengths: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths at printed configuration.
            If 3D-printed length is different from minimum energy length without externally acting forces and torques,
            please input this minimum energy lengths.
        tangents: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element tangent vectors.
        radius: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element radius.
        dilatation: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dilatation.
        voronoi_dilatation: numpy.ndarray
            1D (n_voronoi) array containing data with 'float' type.
            Rod dilatation on voronoi domain.
        dilatation_rate: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dilatation rates.
        rest_voronoi_lengths: numpy.ndarray
            1D (n_voronoi) array containing data with 'float' type.
            Rod lengths on the voronoi domain at the rest configuration.
            The rest voronoi lengths are re-computed when the rest lengths change.
        printed_voronoi_lengths: numpy.ndarray
            1D (n_voronoi) array containing data with 'float' type.
            Rod lengths on the voronoi domain at the printed (e.g. minimum-energy) configuration.
            The printed voronoi lengths remain constant over the entire simulation.
        auxetic: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Binary array (0.0 and 1.0 values) controlling whether we apply the auxetic characteristics to the
            respective elements.
            If set to 1.0, we modify the rest length by adding the scaled twist strain to the original printed length.
            If set to 0.0, we do not modify the rest length of the element.
        handedness: numpy.ndarray
            1D (n_elems - 1) array containing data with 'float' type.
            Handedness sign for twist strain between each rod element (1.0 for right-handed, -1.0 for left-handed).
            A positive handedness sign allows for twist strain caused by a positive torsional torque around the
            local z-axis and leads to a positive elongation of the elements (e.g. increased rest length).
    """

    def __init__(
        self,
        n_elements,
        position,
        velocity,
        omega,
        acceleration,
        angular_acceleration,
        directors,
        radius,
        cross_sectional_area,
        second_moment_of_inertia,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        elastic_modulus,
        elastic_modulus_scale_factor,
        shear_modulus,
        shear_modulus_scale_factor,
        bend_rigidity,
        bend_rigidity_scale_factor,
        twist_rigidity,
        shear_matrix,
        bend_matrix,
        density,
        volume,
        mass,
        dissipation_constant_for_forces,
        dissipation_constant_for_torques,
        internal_forces,
        internal_torques,
        external_forces,
        external_torques,
        lengths,
        rest_lengths,
        rest_lengths_scale_factor,
        printed_lengths,
        tangents,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        rest_voronoi_lengths,
        printed_voronoi_lengths,
        sigma,
        kappa,
        rest_sigma,
        rest_kappa,
        internal_stress,
        internal_couple,
        damping_forces,
        damping_torques,
        auxetic: np.ndarray,
        handedness: np.ndarray,
    ):
        self.n_elems = n_elements
        self.position_collection = position
        self.velocity_collection = velocity
        self.omega_collection = omega
        self.acceleration_collection = acceleration
        self.alpha_collection = angular_acceleration
        self.director_collection = directors
        self.radius = radius
        self.cross_sectional_area = cross_sectional_area
        self.second_moment_of_inertia = second_moment_of_inertia
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
        self.inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        self.elastic_modulus = elastic_modulus
        self.elastic_modulus_scale_factor = elastic_modulus_scale_factor
        self.shear_modulus = shear_modulus
        self.shear_modulus_scale_factor = shear_modulus_scale_factor
        self.bend_rigidity = bend_rigidity
        self.bend_rigidity_scale_factor = bend_rigidity_scale_factor
        self.twist_rigidity = twist_rigidity
        self.shear_matrix = shear_matrix
        self.bend_matrix = bend_matrix
        self.density = density
        self.volume = volume
        self.mass = mass
        self.dissipation_constant_for_forces = dissipation_constant_for_forces
        self.dissipation_constant_for_torques = dissipation_constant_for_torques
        self.internal_forces = internal_forces
        self.internal_torques = internal_torques
        self.external_forces = external_forces
        self.external_torques = external_torques
        self.lengths = lengths
        self.rest_lengths = rest_lengths
        self.rest_lengths_scale_factor = rest_lengths_scale_factor
        self.printed_lengths = printed_lengths
        self.tangents = tangents
        self.dilatation = dilatation
        self.dilatation_rate = dilatation_rate
        self.voronoi_dilatation = voronoi_dilatation
        self.rest_voronoi_lengths = rest_voronoi_lengths
        self.printed_voronoi_lengths = printed_voronoi_lengths
        self.sigma = sigma
        self.kappa = kappa
        self.rest_sigma = rest_sigma
        self.rest_kappa = rest_kappa
        self.internal_stress = internal_stress
        self.internal_couple = internal_couple
        self.damping_forces = damping_forces
        self.damping_torques = damping_torques
        self.auxetic = auxetic
        self.handedness = handedness

        # Compute shear stretch and strains.
        _compute_shear_stretch_strains(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.voronoi_dilatation,
            self.director_collection,
            self.sigma,
        )

        # Compute bending twist strains
        _compute_bending_twist_strains(
            self.director_collection,
            self.rest_voronoi_lengths,
            self.printed_voronoi_lengths,
            self.kappa,
        )

        _compute_shear_bending_matrices(
            self.auxetic,
            self.handedness,
            self.cross_sectional_area,
            self.second_moment_of_inertia,
            self.elastic_modulus,
            self.elastic_modulus_scale_factor,
            self.shear_modulus,
            self.shear_modulus_scale_factor,
            self.bend_rigidity,
            self.bend_rigidity_scale_factor,
            self.twist_rigidity,
            self.rest_lengths,
            self.kappa,
            self.shear_matrix,
            self.bend_matrix,
        )

        _compute_auxetic_rest_lengths(
            self.auxetic,
            self.handedness,
            self.cross_sectional_area,
            self.rest_lengths_scale_factor,
            self.kappa,
            self.volume,
            self.rest_lengths,
            self.printed_lengths,
            self.rest_voronoi_lengths,
            self.printed_voronoi_lengths,
        )

    @classmethod
    def straight_rod(
        cls,
        n_elements: int,
        start: np.ndarray,
        direction: np.ndarray,
        normal: np.ndarray,
        base_length: float,
        base_radius: float,
        density: float,
        *args,
        **kwargs,
    ):
        """
        Cosserat rod constructor for straight-rod geometry.


        Notes
        -----
        Since we expect the Cosserat Rod to simulate soft rod, Poisson's ratio is set to 0.5 by default.
        It is possible to give additional argument "shear_modulus" or "poisson_ratio" to specify extra modulus.


        Parameters
        ----------
        n_elements : int
            Number of element. Must be greater than 3. Generally recommended to start with 40-50, and adjust the resolution.
        start : NDArray[3, float]
            Starting coordinate in 3D
        direction : NDArray[3, float]
            Direction of the rod in 3D
        normal : NDArray[3, float]
            Normal vector of the rod in 3D
        base_length : float
            Total length of the rod
        base_radius : float
            Uniform outside radius of the rod
        density : float
            Density of the rod
        *args : tuple
            Additional arguments should be passed as keyword arguments.
            (e.g. shear_modulus, poisson_ratio)
        **kwargs : dict, optional
            The "position" and/or "directors" can be overrided by passing "position" and "directors" argument. Remember, the shape of the "position" is (3,n_elements+1) and the shape of the "directors" is (3,3,n_elements).

        Returns
        -------
        CosseratRod

        """

        (
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            cross_sectional_area,
            second_moment_of_inertia,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            elastic_modulus,
            elastic_modulus_scale_factor,
            shear_modulus,
            shear_modulus_scale_factor,
            bend_rigidity,
            bend_rigidity_scale_factor,
            twist_rigidity,
            shear_matrix,
            bend_matrix,
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            rest_lengths_scale_factor,
            printed_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            printed_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            damping_forces,
            damping_torques,
            handedness_sign,
            auxetic,
        ) = allocate(
            n_elements,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            *args,
            **kwargs,
        )

        return cls(
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            cross_sectional_area,
            second_moment_of_inertia,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            elastic_modulus,
            elastic_modulus_scale_factor,
            shear_modulus,
            shear_modulus_scale_factor,
            bend_rigidity,
            bend_rigidity_scale_factor,
            twist_rigidity,
            shear_matrix,
            bend_matrix,
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            rest_lengths_scale_factor,
            printed_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            printed_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            damping_forces,
            damping_torques,
            handedness=handedness_sign,
            auxetic=auxetic,
        )

    def compute_internal_forces_and_torques(self, time):
        """
        Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
        they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
        one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
        Saving internal forces and torques in a variable take some memory, but we will gain speed up.

        Parameters
        ----------
        time: float
            current time

        """
        # we update the shear and bend matrices based on the twist strain from the last time step
        _compute_shear_bending_matrices(
            self.auxetic,
            self.handedness,
            self.cross_sectional_area,
            self.second_moment_of_inertia,
            self.elastic_modulus,
            self.elastic_modulus_scale_factor,
            self.shear_modulus,
            self.shear_modulus_scale_factor,
            self.bend_rigidity,
            self.bend_rigidity_scale_factor,
            self.twist_rigidity,
            self.rest_lengths,
            self.kappa,
            self.shear_matrix,
            self.bend_matrix,
        )

        # we update the rest length based on the twist strain from the last time step
        _compute_auxetic_rest_lengths(
            self.auxetic,
            self.handedness,
            self.cross_sectional_area,
            self.rest_lengths_scale_factor,
            self.kappa,
            self.volume,
            self.rest_lengths,
            self.printed_lengths,
            self.rest_voronoi_lengths,
            self.printed_voronoi_lengths,
        )

        _compute_internal_forces(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.voronoi_dilatation,
            self.director_collection,
            self.sigma,
            self.rest_sigma,
            self.shear_matrix,
            self.internal_stress,
            self.velocity_collection,
            self.dissipation_constant_for_forces,
            self.damping_forces,
            self.internal_forces,
            self.ghost_elems_idx,
        )

        _compute_internal_torques(
            self.position_collection,
            self.velocity_collection,
            self.tangents,
            self.lengths,
            self.rest_lengths,
            self.director_collection,
            self.rest_voronoi_lengths,
            self.printed_voronoi_lengths,
            self.bend_matrix,
            self.rest_kappa,
            self.kappa,
            self.voronoi_dilatation,
            self.mass_second_moment_of_inertia,
            self.omega_collection,
            self.internal_stress,
            self.internal_couple,
            self.dilatation,
            self.dilatation_rate,
            self.dissipation_constant_for_torques,
            self.damping_torques,
            self.internal_torques,
            self.ghost_voronoi_idx,
        )


@numba.njit(cache=True)
def _compute_internal_shear_stretch_stresses_from_model(
    position_collection,
    volume,
    lengths,
    tangents,
    radius,
    rest_lengths,
    rest_voronoi_lengths,
    dilatation,
    voronoi_dilatation,
    director_collection,
    sigma,
    rest_sigma,
    shear_matrix,
    internal_stress,
):
    """
    Update <internal stress> given <shear matrix, sigma, and rest_sigma>.

    Linear force functional
    Operates on
    S : (3,3,n) tensor and sigma (3,n)
    """
    _compute_shear_stretch_strains(
        position_collection,
        volume,
        lengths,
        tangents,
        radius,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        voronoi_dilatation,
        director_collection,
        sigma,
    )
    internal_stress[:] = _batch_matvec(shear_matrix, sigma - rest_sigma)


@numba.njit(cache=True)
def _compute_bending_twist_strains(
    director_collection: np.ndarray,
    rest_voronoi_lengths: np.ndarray,
    printed_voronoi_lengths: np.ndarray,
    kappa: np.ndarray,
):
    """
    Update <curvature/twist (kappa)> given <director and rest_voronoi_length>.
    """
    temp = _inv_rotate(director_collection)
    blocksize = rest_voronoi_lengths.shape[0]
    for k in range(blocksize):
        kappa[0, k] = temp[0, k] / rest_voronoi_lengths[k]
        kappa[1, k] = temp[1, k] / rest_voronoi_lengths[k]
        kappa[2, k] = temp[2, k] / printed_voronoi_lengths[k]


@numba.njit(cache=True)
def _compute_internal_bending_twist_stresses_from_model(
    director_collection: np.ndarray,
    rest_voronoi_lengths: np.ndarray,
    printed_voronoi_lengths: np.ndarray,
    internal_couple: np.ndarray,
    bend_matrix: np.ndarray,
    kappa: np.ndarray,
    rest_kappa: np.ndarray,
):
    """
    Upate <internal couple> given <curvature(kappa) and bend_matrix>.

    Linear force functional
    Operates on
    B : (3,3,n) tensor and curvature kappa (3,n)
    """
    _compute_bending_twist_strains(
        director_collection, rest_voronoi_lengths, printed_voronoi_lengths, kappa
    )  # concept : needs to compute kappa

    blocksize = kappa.shape[1]
    temp = np.empty((3, blocksize))
    for i in range(3):
        for k in range(blocksize):
            temp[i, k] = kappa[i, k] - rest_kappa[i, k]

    internal_couple[:] = _batch_matvec(bend_matrix, temp)


@numba.njit(cache=True)
def _compute_internal_torques(
    position_collection: np.ndarray,
    velocity_collection: np.ndarray,
    tangents: np.ndarray,
    lengths: np.ndarray,
    rest_lengths: np.ndarray,
    director_collection: np.ndarray,
    rest_voronoi_lengths: np.ndarray,
    printed_voronoi_lengths: np.ndarray,
    bend_matrix: np.ndarray,
    rest_kappa: np.ndarray,
    kappa: np.ndarray,
    voronoi_dilatation: np.ndarray,
    mass_second_moment_of_inertia: np.ndarray,
    omega_collection: np.ndarray,
    internal_stress: np.ndarray,
    internal_couple: np.ndarray,
    dilatation: np.ndarray,
    dilatation_rate: np.ndarray,
    dissipation_constant_for_torques: np.ndarray,
    damping_torques: np.ndarray,
    internal_torques: np.ndarray,
    ghost_voronoi_idx,
):
    """
    Update <internal torque>.
    """
    # Compute \tau_l and cache it using internal_couple
    # Be careful about usage though
    _compute_internal_bending_twist_stresses_from_model(
        director_collection,
        rest_voronoi_lengths,
        printed_voronoi_lengths,
        internal_couple,
        bend_matrix,
        kappa,
        rest_kappa,
    )
    # Compute dilatation rate when needed, dilatation itself is done before
    # in internal_stresses
    _compute_dilatation_rate(
        position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
    )

    # FIXME: change memory overload instead for the below calls!
    voronoi_dilatation_inv_cube_cached = 1.0 / voronoi_dilatation**3
    # Delta(\tau_L / \Epsilon^3)
    bend_twist_couple_2D = difference_kernel_for_block_structure(
        internal_couple * voronoi_dilatation_inv_cube_cached, ghost_voronoi_idx
    )
    # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epsilon^3 ]
    bend_twist_couple_3D = quadrature_kernel_for_block_structure(
        _batch_cross(kappa, internal_couple)
        * rest_voronoi_lengths
        * voronoi_dilatation_inv_cube_cached,
        ghost_voronoi_idx,
    )
    # (Qt x n_L) * \hat{l}
    shear_stretch_couple = (
        _batch_cross(_batch_matvec(director_collection, tangents), internal_stress)
        * rest_lengths
    )

    # I apply common sub expression elimination here, as J w / e is used in both the lagrangian and dilatation
    # terms
    # TODO : the _batch_matvec kernel needs to depend on the representation of J, and should be coded as such
    J_omega_upon_e = (
        _batch_matvec(mass_second_moment_of_inertia, omega_collection) / dilatation
    )

    # (J \omega_L / e) x \omega_L
    # Warning : Do not do micro-optimization here : you can ignore dividing by dilatation as we later multiply by it
    # but this causes confusion and violates SRP
    lagrangian_transport = _batch_cross(J_omega_upon_e, omega_collection)

    # Note : in the computation of dilatation_rate, there is an optimization opportunity as dilatation rate has
    # a dilatation-like term in the numerator, which we cancel here
    # (J \omega_L / e^2) . (de/dt)
    unsteady_dilatation = J_omega_upon_e * dilatation_rate / dilatation

    _compute_damping_torques(
        damping_torques, omega_collection, dissipation_constant_for_torques
    )

    blocksize = internal_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            internal_torques[i, k] = (
                bend_twist_couple_2D[i, k]
                + bend_twist_couple_3D[i, k]
                + shear_stretch_couple[i, k]
                + lagrangian_transport[i, k]
                + unsteady_dilatation[i, k]
                - damping_torques[i, k]
            )


@numba.njit(cache=True)
def _compute_shear_bending_matrices(
    auxetic: np.ndarray,
    handedness: np.ndarray,
    cross_sectional_area: np.ndarray,
    second_moment_of_inertia: np.ndarray,
    elastic_modulus: np.ndarray,
    elastic_modulus_scale_factor: np.ndarray,
    shear_modulus: np.ndarray,
    shear_modulus_scale_factor: np.ndarray,
    bend_rigidity: np.ndarray,
    bend_rigidity_scale_factor: np.ndarray,
    twist_rigidity: np.ndarray,
    rest_lengths: np.ndarray,
    kappa: np.ndarray,
    shear_matrix: np.ndarray,
    bend_matrix: np.ndarray,
):
    # Clip twist strain according to handedness of HSA to be positive.
    # clipping is only necessary for closed HSA's
    voronoi_twist_strain = np.clip(-handedness * kappa[2, :], a_min=0.0, a_max=None)
    # Elevate dimension of clipped_positive_twist_strain from (n_elements - 1) to (n_elements)
    twist_strain = np.zeros_like(elastic_modulus)
    twist_strain[:-1] = 0.5 * voronoi_twist_strain
    twist_strain[1:] = 0.5 * voronoi_twist_strain

    # Compute the elastic and shear modulus for the current twist strain.
    elastic_modulus_twisted_state = (
        elastic_modulus + elastic_modulus_scale_factor * twist_strain
    )
    shear_modulus_twisted_state = (
        shear_modulus + shear_modulus_scale_factor * twist_strain
    )

    # Update shear and stretch rigidity matrix
    # Value taken based on best correlation for Poisson ratio = 0.5, from
    # "On Timoshenko's correction for shear in vibrating beams" by Kaneko, 1975
    alpha_c = 27.0 / 28.0
    blocksize = shear_matrix.shape[2]
    for k in range(blocksize):
        shear_matrix[0, 0, k] = (
            alpha_c * shear_modulus_twisted_state[k] * cross_sectional_area[k]
        )
        shear_matrix[1, 1, k] = (
            alpha_c * shear_modulus_twisted_state[k] * cross_sectional_area[k]
        )
        shear_matrix[2, 2, k] = (
            elastic_modulus_twisted_state[k] * cross_sectional_area[k]
        )

    # Update bend and twist rigidity matrix
    for k in range(bend_matrix.shape[2]):
        bend_matrix[0, 0, k] = (
            bend_rigidity[k] + bend_rigidity_scale_factor[k] * voronoi_twist_strain[k]
        )
        bend_matrix[1, 1, k] = (
            bend_rigidity[k] + bend_rigidity_scale_factor[k] * voronoi_twist_strain[k]
        )
        bend_matrix[2, 2, k] = twist_rigidity[k]


def _compute_auxetic_rest_lengths(
    auxetic: np.ndarray,
    handedness: np.ndarray,
    cross_sectional_area: np.ndarray,
    rest_lengths_scale_factor: np.ndarray,
    kappa: np.ndarray,
    volume: np.ndarray,
    rest_lengths: np.ndarray,
    printed_lengths: np.ndarray,
    rest_voronoi_lengths: np.ndarray,
    printed_voronoi_lengths: np.ndarray,
):
    # Clip twist strain according to handedness of HSA to be positive.
    # clipping is only necessary for closed HSA's
    voronoi_twist_strain = np.clip(-handedness * kappa[2, :], a_min=0.0, a_max=None)

    # Elevate dimension of clipped_positive_twist_strain from (n_elements - 1) to (n_elements)
    twist_strain = np.zeros_like(rest_lengths)
    twist_strain[:-1] += 0.5 * voronoi_twist_strain
    twist_strain[1:] += 0.5 * voronoi_twist_strain

    # the extension ratio is determined by the scale factor and the (positive) twist strain
    rest_lengths += auxetic * (
        (1.0 + rest_lengths_scale_factor * twist_strain) * printed_lengths
        - rest_lengths
    )

    # re-calculate voronoi rest lengths based on the new rest lengths
    rest_voronoi_lengths += (
        0.5
        * (auxetic[:-1] + auxetic[1:])
        * ((0.5 * (rest_lengths[1:] + rest_lengths[:-1]) - rest_voronoi_lengths))
    )

    # update the volume
    volume += auxetic * (cross_sectional_area * rest_lengths - volume)
