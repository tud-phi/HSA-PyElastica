from collections import defaultdict
from elastica.boundary_conditions import GeneralConstraint
from elastica.dissipation import AnalyticalLinearDamper
from elastica.modules import (
    Connections,
    Constraints,
    Damping,
    Forcing,
    CallBacks,
)
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
import numpy as np
from typing import Dict, Optional, Union

from hsa_elastica.modules import BaseSystemCollection
from hsa_elastica.boundary_conditions import HsaMotorFreeRotBC, HsaMotorControlledRotBC
from hsa_elastica.callbacks import HsaRodDiagnosticCallback
from hsa_elastica.dissipation import ConfigurableAnalyticalLinearDamper
from hsa_elastica.rod import HsaRod


class HsaRodSimulator(
    BaseSystemCollection,
    CallBacks,  # Enabled to use callback
    Connections,  # Enabled to use FixedJoint
    Constraints,  # Enabled to use boundary conditions 'GeneralJoint'
    Damping,  # Enabled to use damping
    Forcing,  # Enabled to use forcing 'GravityForces'
):
    def __init__(
        self,
        rod_params: Dict,
        num_elements: int = 10,
        duration: float = 10.0,
        dt: float = 5e-5,
        fps: float = 100.0,
    ):
        """
        Initialize a simulator for single HSA rods.
        Args:
            rod_params: Dictionary of rod parameters. Needs to contain the following keys:
                - name: name / type of the rod
                - printed_length: original / printed length of the rod [m]
                - outside_radius: outside radius of the rod [m]
                - wall_thickness: wall thickness of the rod [m]
                - material_density: density of the rod material [kg/m^3]
                - elastic_modulus: elastic modulus of the rod material [Pa]
                - elastic_modulus_scale_factor: scale factor for elastic modulus [Pa / (rad/m)]
                    determines how the elastic modulus changes with the twist strain
                - poisson_ratio: poisson ratio of the rod material
                - twist_rigidity: rigidity of the rod material [Nm/(rad/m)] = Nm^2/rad
                - damping_constant: damping constant
                - max_actuation_torque: maximum torque that can be applied to the rod before it leaves the auxetic regime [Nm]
            num_elements: number of elements / links in the rod
            duration: duration of the simulation [s]
            dt: timestep of the simulation [s]
            fps: frequency of rendering / visualization animations and movies [Hz].
                Determines the frequency of diagnostics gathering from the simulation.
        """
        super().__init__()

        self.num_elements = num_elements

        self.rod_params = rod_params
        self.printed_length = rod_params["printed_length"]  # m
        self.outside_radius = rod_params["outside_radius"]  # m
        self.wall_thickness = rod_params["wall_thickness"]  # m

        # Density of rod wall
        self.density = rod_params["material_density"]

        # elastic modulus and damping constants
        self.elastic_modulus = rod_params["elastic_modulus"]  # Pa
        self.elastic_modulus_scale_factor = rod_params[
            "elastic_modulus_scale_factor"
        ]  # Pa / [rad / m]
        if "shear_modulus" in rod_params:
            self.shear_modulus = rod_params["shear_modulus"]
            self.shear_modulus_scale_factor = rod_params.get(
                "shear_modulus_scale_factor", 0.0
            )
        elif "poisson_ratio" in rod_params:
            self.shear_modulus = rod_params["elastic_modulus"] / (
                rod_params["poisson_ratio"] + 1.0
            )
            self.shear_modulus_scale_factor = self.elastic_modulus_scale_factor / (
                rod_params["poisson_ratio"] + 1.0
            )
        else:
            raise ValueError
        self.twist_rigidity = rod_params.get(
            "twist_rigidity", None
        )  # [Nm/(rad/m)] = Nm^2/rad
        # maps twist strain to rest length extension: [1/(rad/m) = m / rad]
        self.rest_lengths_scale_factor = rod_params["rest_lengths_scale_factor"]

        # timestepper settings
        self.duration, self.dt = duration, dt
        self.total_steps = int(self.duration / self.dt)
        self.timestepper = PositionVerlet()

        # frequency of rendering / visualization animations and movies
        # also determines the frequency of diagnostics gathering from the simulation
        # assert that fps is a divisor of (1 / dt)
        assert abs((1 / dt + 1e-6) % fps) <= 1e-5
        self.fps = fps

        # initialize
        self.rod: Optional[HsaRod] = None
        self.diagnostic_data_raw = defaultdict(list)
        self.diagnostic_data = {}

    def configure(
        self,
        finalize: bool = True,
        constrain_extension: bool = False,
        follow_auxetic_trajectory: bool = True,
        actuation_angle: float = 0.0,
        actuation_ramp_up_time: float = 0.0,
    ):
        """
        Configure the simulator.
        Args:
            finalize: boolean whether to finalize the configuration of the simulator
            constrain_extension: boolean whether to constrain the extension of the rod
            follow_auxetic_trajectory: boolean whether to follow the auxetic trajectory
            actuation_angle: twist angle at the base [rad]
            actuation_ramp_up_time: time to ramp up the actuation to its full magnitude [s]
        """
        self.rod = HsaRod.straight_rod(
            n_elements=self.num_elements,  # number of elements
            start=np.array([0.0, 0.0, 0.0]),  # Starting position of first node in rod
            direction=np.array([0.0, 0.0, 1.0]),  # Direction the rod extends
            normal=np.array([1.0, 0.0, 0.0]),  # normal vector of rod
            base_length=self.printed_length,  # original length of rod (m): simulate closed 6 rows
            base_radius=self.outside_radius,
            base_inside_radius=self.outside_radius - self.wall_thickness,
            density=self.density,  # density of rod (kg/m^3)
            elastic_modulus=self.elastic_modulus,
            elastic_modulus_scale_factor=self.elastic_modulus_scale_factor
            if follow_auxetic_trajectory
            else 0.0,
            shear_modulus=self.shear_modulus if follow_auxetic_trajectory else 0.0,
            shear_modulus_scale_factor=self.shear_modulus_scale_factor,
            twist_rigidity=self.twist_rigidity,
            rest_lengths_scale_factor=self.rest_lengths_scale_factor
            if follow_auxetic_trajectory
            else 0.0,
            follow_auxetic_trajectory=follow_auxetic_trajectory,
            handedness="right" if follow_auxetic_trajectory else None,
        )
        self.append(self.rod)

        if (
            "translational_damping_constant" in self.rod_params
            and "rotational_damping_constant" in self.rod_params
        ):
            self.dampen(self.rod).using(
                ConfigurableAnalyticalLinearDamper,
                translational_damping_constant=np.array(
                    self.rod_params["translational_damping_constant"]
                ),
                rotational_damping_constant=np.array(
                    self.rod_params["rotational_damping_constant"]
                ),
                time_step=self.dt,
            )
        else:
            self.dampen(self.rod).using(
                AnalyticalLinearDamper,
                damping_constant=self.rod_params[
                    "damping_constant"
                ],  # Energy dissipation of rod
                time_step=self.dt,
            )

        # Apply boundary conditions to rod
        # Use HsaMotorBc for proximal end
        if actuation_angle == 0.0:
            self.constrain(self.rod).using(
                HsaMotorFreeRotBC,
                constrained_position_idx=(0,),  # Node number to apply BC
                constrained_director_idx=(0,),  # Element number to apply BC
            )
        else:
            self.constrain(self.rod).using(
                HsaMotorControlledRotBC,
                constrained_position_idx=(0,),  # Node number to apply BC
                constrained_director_idx=(0,),  # Element number to apply BC
                actuation_angle=actuation_angle,
                ramp_up_time=actuation_ramp_up_time,
            )
        # Constrain twisting / yaw at the distal end
        self.constrain(self.rod).using(
            GeneralConstraint,
            constrained_position_idx=(-1,),
            constrained_director_idx=(-1,),
            translational_constraint_selector=np.array(
                [False, False, True if constrain_extension else False]
            ),  # Allow all translational DoF
            rotational_constraint_selector=np.array(
                [False, False, True]
            ),  # Restrict yaw
        )

        self.collect_diagnostics(self.rod).using(
            HsaRodDiagnosticCallback,
            step_skip=int(1.0 / (self.fps * self.dt)),
            callback_data=self.diagnostic_data_raw,
        )

        if finalize:
            self.finalize()

    def run(self):
        """
        Run the simulation
        """
        integrate(self.timestepper, self, self.duration, self.total_steps)
        self.postprocess_diagnostic_data()

    def postprocess_diagnostic_data(self):
        for key, value in self.diagnostic_data_raw.items():
            self.diagnostic_data[key] = np.array(value)
