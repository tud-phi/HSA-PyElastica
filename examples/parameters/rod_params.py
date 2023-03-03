ROD_BASE_PARAMS = dict(
    outside_radius=19e-3,  # m
    wall_thickness=2e-3,  # m
    poisson_ratio=0.5,
    radial_offset=16e-3,  # radial offset of rod from xy origin [m] measurement from our robot: 24 mm
    num_elements=10,
)

ROD_FPU50_CLOSED_4ROWS_PARAMS = ROD_BASE_PARAMS | dict(
    name="HSA_ROD_FPU50_CLOSED_4ROWS",
    material_density=1.05e3,  # kg/m^3
    printed_length=74.86e-3,  # m
    elastic_modulus=576.9e3,  # Pa
    elastic_modulus_scale_factor=36.1e3,  # Pa / [rad / m]
    twist_rigidity=3.75e-3,  # [Nm/(rad/m)] = Nm^2/rad
    rest_lengths_scale_factor=3.037e-3,  # [1/(rad/m) = m / rad]
    damping_constant=3e0,
    max_actuation_torque=87.42e-3,  # Nm
)

ROD_FPU50_CLOSED_6ROWS_PARAMS = ROD_BASE_PARAMS | dict(
    name="HSA_ROD_FPU50_CLOSED_6ROWS",
    material_density=1.05e3,  # kg/m^3
    printed_length=89e-3,  # m
    elastic_modulus=309.3e3,  # Pa
    elastic_modulus_scale_factor=13.05e3,  # Pa / [rad / m]
    twist_rigidity=2.13e-3,  # [Nm/(rad/m)] = Nm^2/rad
    rest_lengths_scale_factor=3.5e-3,  # [1/(rad/m) = m / rad]
    damping_constant=1e0,
    max_actuation_torque=68.81e-3,  # Nm
)

ROD_FPU50_CLOSED_8ROWS_PARAMS = ROD_BASE_PARAMS | dict(
    name="HSA_ROD_FPU50_CLOSED_8ROWS",
    material_density=1.05e3,  # kg/m^3
    printed_length=100e-3,  # m
    elastic_modulus=203.5e3,  # Pa
    elastic_modulus_scale_factor=10.59e3,  # Pa / [rad / m]
    twist_rigidity=1.831e-3,  # [Nm/(rad/m)] = Nm^2/rad
    rest_lengths_scale_factor=3.768e-3,  # [1/(rad/m) = m / rad]
    damping_constant=1e0,
    max_actuation_torque=53.25e-3,  # Nm
)

ROD_FPU50_CLOSED_10ROWS_PARAMS = ROD_BASE_PARAMS | dict(
    name="HSA_ROD_FPU50_CLOSED_10ROWS",
    material_density=1.05e3,  # kg/m^3
    printed_length=112e-3,  # m
    elastic_modulus=197.57e3,  # Pa
    elastic_modulus_scale_factor=7.50e3,  # Pa / [rad / m]
    twist_rigidity=1.67e-3,  # [Nm/(rad/m)] = Nm^2/rad
    rest_lengths_scale_factor=3.644e-3,  # [1/(rad/m) = m / rad]
    damping_constant=2e0,
    max_actuation_torque=52.06e-3,  # Nm
)

ROD_FPU50_CLOSED_12ROWS_PARAMS = ROD_BASE_PARAMS | dict(
    name="HSA_ROD_FPU50_CLOSED_12ROWS",
    material_density=1.05e3,  # kg/m^3
    printed_length=124e-3,  # m
    elastic_modulus=197.57e3,  # Pa
    elastic_modulus_scale_factor=2.4e3,  # Pa / [rad / m]
    twist_rigidity=1.244e-3,  # [Nm/(rad/m)] = Nm^2/rad
    rest_lengths_scale_factor=3.527e-3,  # [1/(rad/m) = m / rad]
    damping_constant=1e0,
    max_actuation_torque=35e-3,  # Nm
)

ROD_FPU50_OPEN_4ROWS_PARAMS = ROD_BASE_PARAMS | dict(
    name="HSA_ROD_FPU50_OPEN_4ROWS",
    material_density=1.05e3,  # kg/m^3
    printed_length=122.27e-3,  # m
    elastic_modulus=576.9e3,  # Pa
    elastic_modulus_scale_factor=0e3,  # Pa / [rad / m]
    twist_rigidity=2.544e-3,  # [Nm/(rad/m)] = Nm^2/rad
    rest_lengths_scale_factor=0.74e-3,  # [1/(rad/m) = m / rad]
    damping_constant=4e0,
    min_actuation_torque=-72.5e-3,  # Nm
    max_actuation_torque=0e-3,  # Nm
)

# Some kind of adjusted rod used for HSA robot  simulations
ROD_ROBOT_SIM = dict(
    name="HSA_ROD_ROBOT_SIM",
    outside_radius=25.4e-3 / 2,  # m
    wall_thickness=2.43e-3,  # m
    radial_offset=24e-3,  # radial offset of rod from xy origin [m] measurement from our robot: 24 mm
    material_density=1.05e3,  # kg/m^3
    printed_length=100e-3,  # m
    elastic_modulus=1e7,  # Pa
    elastic_modulus_scale_factor=0e0,  # Pa / [rad / m]
    shear_modulus=8e5,  # Pa
    # bend_rigidity=3e-2,  # [Nm/(rad/m)] = Nm^2/rad worked decently for bending
    bend_rigidity=2e-2,  # [Nm/(rad/m)] = Nm^2/rad
    twist_rigidity=14e-3,  # [Nm/(rad/m)] = Nm^2/rad
    rest_lengths_scale_factor=1e-2,  # [1/(rad/m) = m / rad]
    translational_damping_constant=[8e3, 8e3, 1e4],
    rotational_damping_constant=[5e0, 5e0, 7e0],
    max_actuation_torque=500e-3,  # Nm
    num_elements=25,
)
