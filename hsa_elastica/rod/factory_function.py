__doc__ = """ Factory function to allocate variables for Cosserat Rod"""
__all__ = ["allocate"]
from typing import Optional, Tuple
import logging
import numpy as np
from numpy.testing import assert_allclose
from elastica.rod.factory_function import (
    _assert_dim,
    _assert_shape,
    _position_validity_checker,
    _directors_validity_checker,
)
from elastica.utils import MaxDimension, Tolerance
from elastica._linalg import _batch_cross, _batch_norm, _batch_dot


def allocate(
    n_elements,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    elastic_modulus: float,
    shear_modulus: float,
    nu_for_torques: Optional[float] = None,
    position: Optional[np.ndarray] = None,
    directors: Optional[np.ndarray] = None,
    base_inside_radius: float = 0.0,
    elastic_modulus_scale_factor: float = 0.0,
    shear_modulus_scale_factor: float = 0.0,
    bend_rigidity: Optional[float] = None,
    bend_rigidity_scale_factor: float = 0.0,
    twist_rigidity: Optional[float] = None,
    rest_lengths_scale_factor: float = 0.0,
    rest_sigma: Optional[np.ndarray] = None,
    rest_kappa: Optional[np.ndarray] = None,
    follow_auxetic_trajectory: bool = True,
    handedness: Optional[str] = None,
    *args,
    **kwargs,
):
    log = logging.getLogger()

    # sanity checks here
    assert n_elements > 1
    assert base_length > Tolerance.atol()
    assert np.sqrt(np.dot(normal, normal)) > Tolerance.atol()
    assert np.sqrt(np.dot(direction, direction)) > Tolerance.atol()

    # check if position is given.
    if position is None:  # Generate straight and uniform rod
        # Set the position array
        position = np.zeros((MaxDimension.value(), n_elements + 1))
        end = start + direction * base_length
        for i in range(0, 3):
            position[i, ...] = np.linspace(start[i], end[i], n_elements + 1)
    _position_validity_checker(position, start, n_elements)

    # Boolean to choose whether to follow the auxetic trajectory
    # (e.g. apply any deviations of standard Cosserat rod)
    auxetic = (
        np.ones((n_elements)) if follow_auxetic_trajectory else np.zeros((n_elements))
    )
    # if auxetic trajectory is deactivated, make sure that all scale factors are zero
    if not follow_auxetic_trajectory:
        assert (
            rest_lengths_scale_factor == 0.0
        ), "rest_lengths_scale_factor must be zero if auxetic trajectory is deactivated"
        assert (
            elastic_modulus_scale_factor == 0.0
        ), "elastic_modulus_scale_factor must be zero if auxetic trajectory is deactivated"
        assert (
            shear_modulus_scale_factor == 0.0
        ), "shear_modulus_scale_factor must be zero if auxetic trajectory is deactivated"
        assert (
            bend_rigidity_scale_factor == 0.0
        ), "bend_rigidity_scale_factor must be zero if auxetic trajectory is deactivated"

    # Compute rest lengths
    position_diff = position[..., 1:] - position[..., :-1]
    rest_lengths = _batch_norm(position_diff)
    printed_lengths = rest_lengths.copy()

    # Set rest lengths scale factor
    rest_lengths_scale_factor = np.ones_like(rest_lengths) * rest_lengths_scale_factor

    # Compute tangents and normal vector
    tangents = position_diff / rest_lengths
    normal /= np.linalg.norm(normal)

    if directors is None:  # Generate straight uniform rod
        # Set the directors matrix
        directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
        # Construct directors using tangents and normal
        normal_collection = np.repeat(normal[:, np.newaxis], n_elements, axis=1)
        # Check if rod normal and rod tangent are perpendicular to each other otherwise
        # directors will be wrong!!
        assert_allclose(
            _batch_dot(normal_collection, tangents),
            0,
            atol=Tolerance.atol(),
            err_msg=(" Rod normal and tangent are not perpendicular to each other!"),
        )
        directors[0, ...] = normal_collection
        directors[1, ...] = _batch_cross(tangents, normal_collection)
        directors[2, ...] = tangents
    _directors_validity_checker(directors, tangents, n_elements)

    # Set radius array
    outside_radius = np.zeros((n_elements))
    inside_radius = np.zeros((n_elements))
    # Check if the user input radius is valid
    radius_temp = np.array(base_radius)
    _assert_dim(radius_temp, 2, "radius")
    outside_radius[:] = radius_temp
    # Set inside radius
    inside_radius[:] = np.array(base_inside_radius)
    # Check if the elements of radius are greater than tolerance
    assert np.all(
        outside_radius > Tolerance.atol()
    ), " Radius has to be greater than 0."
    # Check that the inside radius is smaller than the outside radius
    assert np.all(
        inside_radius < outside_radius
    ), "Inside radius has to be smaller than outside radius."

    # Set density array
    density_array = np.zeros((n_elements))
    # Check if the user input density is valid
    density_temp = np.array(density)
    _assert_dim(density_temp, 2, "density")
    density_array[:] = density_temp
    # Check if the elements of density are greater than tolerance
    assert np.all(
        density_array > Tolerance.atol()
    ), " Density has to be greater than 0."

    # Area of the ring for each disk cross-section
    cross_sectional_area = (
        np.ones_like(rest_lengths) * np.pi * (outside_radius**2 - inside_radius**2)
    )

    # Second moment of inertia of a ring
    I0_1 = np.pi * (outside_radius**4 - inside_radius**4) / 4
    I0_2 = I0_1
    I0_3 = 2.0 * I0_2
    I0 = np.array([I0_1, I0_2, I0_3])
    second_moment_of_inertia = np.einsum("ji,i->ji", I0, rest_lengths)

    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    mass_second_moment_of_inertia_temp = np.einsum(
        "ji,i->ji", I0, density * rest_lengths
    )
    for i in range(n_elements):
        np.fill_diagonal(
            mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia_temp[:, i],
        )
    # sanity check of mass second moment of inertia
    if (mass_second_moment_of_inertia < Tolerance.atol()).all():
        message = "Mass moment of inertia matrix smaller than tolerance, please check provided radius, density and length."
        log.warning(message)

    # Inverse of second moment of inertia
    inv_mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements)
    )
    for i in range(n_elements):
        # Check rank of mass moment of inertia matrix to see if it is invertible
        assert (
            np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i])
            == MaxDimension.value()
        )
        inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            mass_second_moment_of_inertia[..., i]
        )

    # Create elastic modulus array
    elastic_modulus = np.ones((n_elements)) * elastic_modulus
    # Create shear modulus array
    shear_modulus = np.ones((n_elements)) * shear_modulus

    # Scaling factors for elastic modulus and shear modulus
    elastic_modulus_scale_factor = np.ones((n_elements)) * elastic_modulus_scale_factor
    shear_modulus_scale_factor = np.ones((n_elements)) * shear_modulus_scale_factor

    # Create bend rigidity array
    if bend_rigidity is None:
        # current implementation only works for symmetric bending rigidity in x- and y-direction
        assert np.array_equal(second_moment_of_inertia[0], second_moment_of_inertia[1])
        bend_rigidity = elastic_modulus * second_moment_of_inertia[0]
        # transform into voronoi domain
        bend_rigidity = (
            bend_rigidity[1:] * rest_lengths[1:] + bend_rigidity[:-1] * rest_lengths[:1]
        ) / (rest_lengths[1:] + rest_lengths[:-1])
    else:
        assert type(bend_rigidity) == float, "bend_rigidity must be a float!"
        bend_rigidity = np.ones(n_elements - 1) * bend_rigidity
    bend_rigidity_scale_factor = np.ones((n_elements - 1)) * bend_rigidity_scale_factor

    # Create twist rigidity array
    if twist_rigidity is None:
        twist_rigidity = (
            shear_modulus[:-1] * second_moment_of_inertia[2, :-1]
            + shear_modulus[1:] * second_moment_of_inertia[2, 1:]
        ) / 2
    else:
        assert type(twist_rigidity) == float, "twist_rigidity must be a float!"
        twist_rigidity = np.ones((n_elements - 1)) * twist_rigidity

    # Allocate memory for shear_matrix and bend_matrix
    shear_matrix = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
    bend_matrix = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements - 1))

    # Compute volume of elements (will be later updated at each time step)
    volume = cross_sectional_area * rest_lengths

    # Compute mass of elements
    mass = np.zeros(n_elements + 1)
    mass[:-1] += 0.5 * density * volume
    mass[1:] += 0.5 * density * volume

    # Initialize the dissipation to zero to remain compatible with v0.3.0
    dissipation_constant_for_forces = np.zeros((n_elements + 1))
    dissipation_constant_for_torques = np.zeros((n_elements))

    # Generate rest sigma and rest kappa, use user input if defined
    # set rest strains and curvature to be  zero at start
    # if found in kwargs modify (say for curved rod)
    if rest_sigma is None:
        rest_sigma = np.zeros((MaxDimension.value(), n_elements))
    _assert_shape(rest_sigma, (MaxDimension.value(), n_elements), "rest_sigma")

    if rest_kappa is None:
        rest_kappa = np.zeros((MaxDimension.value(), n_elements - 1))
    _assert_shape(rest_kappa, (MaxDimension.value(), n_elements - 1), "rest_kappa")

    # Compute rest voronoi length
    rest_voronoi_lengths = 0.5 * (rest_lengths[1:] + rest_lengths[:-1])
    printed_voronoi_lengths = rest_voronoi_lengths.copy()

    # Allocate arrays for Cosserat Rod equations
    velocities = np.zeros((MaxDimension.value(), n_elements + 1))
    omegas = np.zeros((MaxDimension.value(), n_elements))
    accelerations = 0.0 * velocities
    angular_accelerations = 0.0 * omegas

    internal_forces = 0.0 * accelerations
    internal_torques = 0.0 * angular_accelerations

    external_forces = 0.0 * accelerations
    external_torques = 0.0 * angular_accelerations

    lengths = np.zeros((n_elements))
    tangents = np.zeros((3, n_elements))

    dilatation = np.zeros((n_elements))
    voronoi_dilatation = np.zeros((n_elements - 1))
    dilatation_rate = np.zeros((n_elements))

    sigma = np.zeros((3, n_elements))
    kappa = np.zeros((3, n_elements - 1))

    internal_stress = np.zeros((3, n_elements))
    internal_couple = np.zeros((3, n_elements - 1))

    damping_forces = np.zeros((3, n_elements + 1))
    damping_torques = np.zeros((3, n_elements))

    # HSA handedness
    if handedness == "right":
        handedness_sign = 1
    elif handedness == "left":
        handedness_sign = -1
    elif handedness == None:
        handedness_sign = 0
    else:
        raise ValueError("handedness must be 'left' or 'right'")
    handedness_sign = np.ones((internal_couple.shape[1])) * handedness_sign

    return (
        n_elements,
        position,
        velocities,
        omegas,
        accelerations,
        angular_accelerations,
        directors,
        outside_radius,
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
        density_array,
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
    )
