from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from elastica.dissipation import AnalyticalLinearDamper
from elastica.experimental.connection_contact_joint.generic_system_type_connection import (
    GenericSystemTypeFixedJoint,
)
from elastica.external_forces import GravityForces, UniformTorques
from elastica.modules import (
    Connections,
    Constraints,
    Damping,
    Forcing,
    CallBacks,
)
from elastica.rigidbody import Cylinder
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Union
import yaml

from hsa_elastica.boundary_conditions import HsaMotorFreeRotBC, HsaMotorControlledRotBC
from hsa_elastica.callbacks.hsa_diagnostic_callbacks import (
    HsaRodDiagnosticCallback,
    HsaRigidBodyDiagnosticCallback,
)
from hsa_elastica.dissipation import ConfigurableAnalyticalLinearDamper
from hsa_elastica.forcing import ProximalEndTorsion
from hsa_elastica.modules import BaseSystemCollection
from hsa_elastica.rod import HsaRod


class HsaRobotSimulator(
    BaseSystemCollection,
    CallBacks,  # Enabled to use callback
    Connections,  # Enabled to use FixedJoint
    Constraints,  # Enabled to use boundary conditions 'GeneralJoint'
    Damping,  # Enabled to use damping
    Forcing,  # Enabled to use forcing 'GravityForces'
):
    segments = []

    def __init__(
        self,
        name: str,
        robot_params: Dict,
        duration: float = 10.0,
        dt: float = 5e-5,
        timestepper: object = None,
        add_gravity: bool = False,
        fps: float = 100.0,
        log_dir: str = None,
    ):
        """
        Initializes a HSA robot simulator.
        Args:
            name: name of the robot
            robot_params: dictionary of robot parameters
            duration: duration of the simulation [s]
            dt: timestep of the simulation [s]
            timestepper: timestepper object.
             By default, we use the `elastica.timestepper.symplectic_steppers.PositionVerlet` time stepper.
            add_gravity: boolean flag to add gravity to the simulation
            fps: frequency of rendering / visualization animations and movies.
                This determines the frequency of diagnostic data being saved.
            log_dir: directory to save the diagnostic data
        """
        super().__init__()

        self.name = name
        self.robot_params = robot_params
        self.num_segments = len(self.robot_params["segments"])
        self.num_rods_per_segment = self.robot_params["num_rods_per_segment"]
        self.num_rods = self.num_segments * self.num_rods_per_segment
        self.add_gravity = add_gravity

        # timestepper settings
        self.duration, self.dt = duration, dt
        self.total_steps = int(self.duration / self.dt)
        self.timestepper = timestepper
        if self.timestepper is None:
            self.timestepper = PositionVerlet()

        # frequency of rendering / visualization animations and movies
        # also determines the frequency of diagnostics gathering from the simulation
        # assert that fps is a divisor of (1 / dt)
        assert abs((1 / dt + 1e-6) % fps) <= 1e-5
        self.fps = fps

        # array for each segment, same structure as segments dict
        self.diagnostic_data_raw = []
        self.diagnostic_data = []
        # transformed diagnostic data as full numpy arrays
        # Dict of numpy arrays for each state (e.g. positions, directors etc)
        # shape of numpy arrays: num_timesteps x num_segments x num_rods x ... x blocksize
        # where ... are the dimensions of the state and `blocksize` the number of nodes / elements of the Cosserat Rod
        self.rod_diagnostic_arrays: Dict[str, np.array] = {}
        # Dict of numpy arrays for each state (e.g. positions, directors etc)
        # shape of numpy arrays: num_timesteps x num_segments x ...
        # where ... are the dimensions of the state
        self.platform_diagnostic_arrays: Dict[str, np.array] = {}

        now = datetime.now()  # current date and time
        self.sim_id = f"{self.name}-{now.strftime('%Y%m%d_%H%M%S')}"
        if log_dir is None:
            self.log_dir = Path("logs") / self.sim_id
        else:
            self.log_dir = Path(log_dir)
        # Create logdir if it doesn't exist yet
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def configure(
        self,
        add_bodies: bool = True,
        add_constraints: bool = True,
        add_joints: bool = True,
        add_external_forces: bool = True,
        finalize: bool = True,
    ):
        """
        Configures the robot simulator.
        Args:
            add_bodies: boolean flag to add bodies to the simulator
            add_constraints: boolean flag to add constraints to the simulator
            add_joints: boolean flag to add joints to the simulator
            add_external_forces: boolean flag to add external forces to the simulator
            finalize: boolean flag to finalize the simulator
        """
        if add_bodies:
            self.add_bodies()
        if add_constraints:
            self.add_constraints()
        if add_joints:
            self.add_joints()
        if add_external_forces:
            self.add_external_forces()

        self.configure_diagnostics()

        if finalize:
            self.finalize()

    def add_bodies(self):
        self.segments = []
        center_base_position = np.array([0.0, 0.0, 0.0])
        for i in range(1, (self.num_segments + 1)):
            segment_params = self.robot_params["segments"][i - 1]
            segment = {}

            center_tip_position = center_base_position + segment_params[
                "printed_length"
            ] * np.array([0, 0, 1])

            # rods
            segment["rods"] = []
            for j in range(self.num_rods_per_segment):
                rod_params = segment_params["rods"][j]

                phi = rod_params["phi"]
                rod_base_offset = (
                    np.array([np.cos(phi), np.sin(phi), 0.0])
                    * rod_params["radial_offset"]
                )
                rod_base_position = center_base_position + rod_base_offset
                # inside radius
                inside_radius = (
                    rod_params["outside_radius"] - rod_params["wall_thickness"]
                )

                # set elastic and shear modulus params
                elastic_modulus_scale_factor = rod_params[
                    "elastic_modulus_scale_factor"
                ]
                if "shear_modulus" in rod_params:
                    shear_modulus = rod_params["shear_modulus"]
                    shear_modulus_scale_factor = rod_params.get(
                        "shear_modulus_scale_factor", 0.0
                    )
                elif "poisson_ratio" in rod_params:
                    shear_modulus = rod_params["elastic_modulus"] / (
                        rod_params["poisson_ratio"] + 1.0
                    )
                    shear_modulus_scale_factor = elastic_modulus_scale_factor / (
                        rod_params["poisson_ratio"] + 1.0
                    )
                else:
                    raise ValueError

                # Create rod
                rod = HsaRod.straight_rod(
                    n_elements=rod_params["num_elements"],  # number of elements
                    start=rod_base_position,  # Starting position of first node in rod
                    direction=np.array([0.0, 0.0, 1.0]),  # Direction the rod extends
                    normal=np.array([1.0, 0.0, 0.0]),  # normal vector of rod
                    base_length=rod_params["printed_length"],
                    base_radius=rod_params["outside_radius"],
                    base_inside_radius=inside_radius,
                    density=rod_params["material_density"],  # density of rod (kg/m^3)
                    elastic_modulus=rod_params["elastic_modulus"],
                    elastic_modulus_scale_factor=elastic_modulus_scale_factor,
                    shear_modulus=shear_modulus,
                    shear_modulus_scale_factor=shear_modulus_scale_factor,
                    bend_rigidity=rod_params.get("bend_rigidity", None),
                    bend_rigidity_scale_factor=rod_params.get(
                        "bend_rigidity_scale_factor", 0.0
                    ),
                    twist_rigidity=rod_params["twist_rigidity"],
                    rest_lengths_scale_factor=rod_params["rest_lengths_scale_factor"],
                    handedness=rod_params["handedness"],
                )
                # append rod
                self.append(rod)
                segment["rods"].append(rod)

                # add damping to rod
                if (
                    "translational_damping_constant" in rod_params
                    and "rotational_damping_constant" in rod_params
                ):
                    self.dampen(rod).using(
                        ConfigurableAnalyticalLinearDamper,
                        translational_damping_constant=np.array(
                            rod_params["translational_damping_constant"]
                        ),
                        rotational_damping_constant=np.array(
                            rod_params["rotational_damping_constant"]
                        ),
                        time_step=self.dt,
                    )
                else:
                    self.dampen(rod).using(
                        AnalyticalLinearDamper,
                        damping_constant=rod_params[
                            "damping_constant"
                        ],  # Energy dissipation of rod
                        time_step=self.dt,
                    )

            # platform
            platform = Cylinder(
                start=center_tip_position,
                direction=np.array([0.0, 0.0, 1.0]),
                normal=np.array([1.0, 0.0, 0.0]),
                base_length=segment_params["platform"]["thickness"],
                base_radius=segment_params["platform"]["radius"],
                density=segment_params["platform"]["density"],
            )
            self.append(platform)
            segment["platform"] = platform

            self.segments.append(segment)

            # update center base position
            center_base_position = center_tip_position + segment_params["platform"][
                "thickness"
            ] * np.array([0, 0, 1])

    def add_constraints(self):
        """
        Add constraints to the model
        At the moment, only the proximal end of the HSA is constrained.
        We constrain all DoFs except for the twist of the proximal end.
        """
        # constrains at proximal end of HSA
        for rod in self.segments[0]["rods"]:
            self.constrain(rod).using(
                HsaMotorFreeRotBC,
                constrained_position_idx=(0,),  # Node number to apply BC
                constrained_director_idx=(0,),  # Element number to apply BC
            )

    def add_controlled_boundary_conditions(
        self, actuation_angles: np.ndarray, ramp_up_time: float = 0.0
    ):
        """
        Add controlled boundary conditions to the model
        Move the proximal end of the HSA rod to a desired twist angle
        """
        # Attention: only works for the first segment
        for rod_idx, rod in enumerate(self.segments[0]["rods"]):
            self.constrain(rod).using(
                HsaMotorControlledRotBC,
                constrained_position_idx=(0,),  # Node number to apply BC
                constrained_director_idx=(0,),  # Element number to apply BC
                actuation_angle=actuation_angles[0, rod_idx],
                ramp_up_time=ramp_up_time,
            )

    def add_joints(self):
        """
        Add joints to the model
        At the moment, add fixed joints between the distal end of the HSA rods and the platform.
        """
        # introduce platform constraints
        fixed_joint_params = dict(
            # v1 parameters
            # k=4e4,  # translational joint stiffness
            # kt=3e1,  # rotational joint stiffness
            k=5e5,  # translational joint stiffness
            nu=0e0,  # translational damping coefficient
            kt=2e1,  # rotational joint stiffness
            nut=0e0,  # rotational damping coefficient
        )

        for i, segment in enumerate(self.segments, start=1):
            segment_params = self.robot_params["segments"][i - 1]
            segment_thickness = segment_params["platform"]["thickness"]

            if i >= 2:
                # attach proximal end of rods to platform of previous platform
                # TODO: allow for yawing (e.g. rotation of motors)
                previous_platform = self.segments[i - 2]["platform"]
                for j, rod in enumerate(segment["rods"]):
                    rod_params = segment_params["rods"][j]

                    phi = rod_params["phi"]
                    point_on_platform = np.array(
                        [
                            np.cos(phi) * rod_params["radial_offset"],
                            np.sin(phi) * rod_params["radial_offset"],
                            segment_thickness / 2,
                        ]
                    )
                    self.connect(
                        previous_platform,
                        rod,
                        first_connect_idx=0,
                        second_connect_idx=-1,
                    ).using(
                        GenericSystemTypeFixedJoint,
                        point_system_one=point_on_platform,
                        **fixed_joint_params,
                    )
            # attach distal end of rods to the platform
            for j, rod in enumerate(segment["rods"]):
                rod_params = segment_params["rods"][j]

                phi = rod_params["phi"]
                point_on_platform = np.array(
                    [
                        np.cos(phi) * rod_params["radial_offset"],
                        np.sin(phi) * rod_params["radial_offset"],
                        -segment_thickness / 2,
                    ]
                )
                self.connect(
                    rod, segment["platform"], first_connect_idx=-1, second_connect_idx=0
                ).using(
                    GenericSystemTypeFixedJoint,
                    point_system_two=point_on_platform,
                    **fixed_joint_params,
                )

    def add_external_forces(self):
        if self.add_gravity:
            gravitational_acc = -9.80665  # m/s^2
            for body in self._systems:
                self.add_forcing_to(body).using(
                    GravityForces, acc_gravity=np.array([0.0, 0.0, -gravitational_acc])
                )

    def add_constant_actuation_torque(self, actuation_torques: np.array):
        """
        Apply a constant torsional torque at the proximal end of the HSA rods
        Args:
            actuation_torques: array of shape (num_segments, num_rods_per_segment)
        """
        assert actuation_torques.shape == (self.num_segments, self.num_rods_per_segment)
        for i, segment in enumerate(self.segments, start=1):
            for j, rod in enumerate(segment["rods"]):
                self.add_forcing_to(rod).using(
                    ProximalEndTorsion,
                    torsional_torque=actuation_torques[i - 1, j],
                )

    def configure_diagnostics(self):
        diagnostic_params = dict(
            step_skip=int(1.0 / (self.fps * self.dt)),
            sim_total_steps=self.total_steps,
            # export_method="npz",  # numpy savez
        )

        for i, segment in enumerate(self.segments, 1):
            self.diagnostic_data_raw.append({"rods": []})
            for j, rod in enumerate(segment["rods"]):
                self.diagnostic_data_raw[i - 1]["rods"].append(defaultdict(list))
                self.collect_diagnostics(rod).using(
                    HsaRodDiagnosticCallback,
                    callback_data=self.diagnostic_data_raw[i - 1]["rods"][j],
                    # export_path=str(self.log_dir / f"s-{i}" / f"rod-{j}" / "diagnostic_data_raw"),
                    **diagnostic_params,
                )

            self.diagnostic_data_raw[i - 1]["platform"] = defaultdict(list)
            self.collect_diagnostics(segment["platform"]).using(
                HsaRigidBodyDiagnosticCallback,
                callback_data=self.diagnostic_data_raw[i - 1]["platform"],
                # export_path=str(self.log_dir / f"s-{i}" / "platform" / "diagnostic_data_raw"),
                **diagnostic_params,
            )

    def run(self):
        """
        Run the simulation
        """
        # save robot params as yaml file
        with open(str(self.log_dir / "robot_params.yaml"), "w+") as file:
            yaml.dump(self.robot_params, file)

        integrate(self.timestepper, self, self.duration, self.total_steps)

        self.postprocess_diagnostic_data()

    def postprocess_diagnostic_data(self):
        """
        Convert diagnostic data from List[np.array] to np.array of shape (N, ...)
        """
        # convert List[np.array] to np.array of shape (N, ...) where N is the number of time steps
        self.diagnostic_data = deepcopy(self.diagnostic_data_raw)
        for i, segment in enumerate(self.diagnostic_data, 1):
            for j, rod in enumerate(segment["rods"]):
                for key, value in rod.items():
                    rod[key] = np.array(value)
                platform = segment["platform"]
                for key, value in platform.items():
                    platform[key] = np.array(value)

        # populate rod_diagnostic_arrays
        rod_keys = self.diagnostic_data_raw[0]["rods"][0].keys()  # name of states
        for key in rod_keys:
            num_timesteps = len(self.diagnostic_data_raw[0]["rods"][0][key])

            sample_data = self.diagnostic_data_raw[0]["rods"][0][key][0]
            if type(sample_data) in [int, float]:
                data_shape = (1,)
            elif type(sample_data) == np.ndarray:
                data_shape = sample_data.shape
            else:
                raise ValueError(f"Unknown data type: {type(sample_data)}")

            # init numpy array
            self.rod_diagnostic_arrays[key] = np.zeros(
                (
                    num_timesteps,
                    len(self.diagnostic_data_raw),
                    len(self.diagnostic_data_raw[0]["rods"]),
                )
                + data_shape
            )
            for i, segment in enumerate(self.diagnostic_data, 1):
                for j, rod in enumerate(segment["rods"]):
                    rod_data = np.array(rod[key])
                    if rod_data.ndim == 1:
                        rod_data = rod_data.reshape((-1, 1))
                    self.rod_diagnostic_arrays[key][:, i - 1, j, ...] = rod_data

        # populate platform_diagnostic_arrays
        platform_keys = self.diagnostic_data_raw[0]["platform"].keys()  # name of states
        for key in platform_keys:
            num_timesteps = len(self.diagnostic_data_raw[0]["platform"][key])

            sample_data = self.diagnostic_data_raw[0]["platform"][key][0]
            if type(sample_data) in [int, float]:
                data_shape = (1,)
            elif type(sample_data) == np.ndarray:
                data_shape = sample_data.shape
            else:
                raise ValueError(f"Unknown data type: {type(sample_data)}")

            # init numpy array
            self.platform_diagnostic_arrays[key] = np.zeros(
                (num_timesteps, len(self.diagnostic_data_raw)) + data_shape
            )
            for i, segment in enumerate(self.diagnostic_data, 1):
                platform_data = np.array(segment["platform"][key])
                if platform_data.ndim == 1:
                    platform_data = platform_data.reshape((-1, 1))
                self.platform_diagnostic_arrays[key][:, i - 1, ...] = platform_data

    def get_diagnostic_data(self):
        return self.diagnostic_data

    def save_diagnostic_data(self):
        """
        Save diagnostic data to disk at self.log_dir
        """
        print("Saving diagnostic data to:", self.log_dir)
        with open(str(self.log_dir / "diagnostic_data.pkl"), "wb") as f:
            pickle.dump(self.diagnostic_data, f)

        np.savez(
            str(self.log_dir / "rod_diagnostic_arrays.npz"),
            **self.rod_diagnostic_arrays,
        )

        np.savez(
            str(self.log_dir / "platform_diagnostic_arrays.npz"),
            **self.platform_diagnostic_arrays,
        )
