# A plugin to PyElastica for the simulation of HSA robots

This repository contains a plugin for  [PyElastica](https://github.com/GazzolaLab/PyElastica) to simulate
robots based on Handed Shearing Auxetics (HSA).
Elastica makes use of the Discrete Cosserat Rod Model (DCM) to simulate the behaviour of slender rods.
We model the HSA rods as Cosserat rods, but perform several modifications to the DCM to account for the auxetic behaviour of HSAs.
Namely, we couple twist strains to an adjustment of the rest length of the rod.
This will let the rod extend when torsional torques are applied to it.
Additionally, the axial stiffness of the rod is modelled to be a linear function of the twist strain to account for a proportional 
increase of the spring constant with increasing twist angles, as shown by Good et al. [[1]](#1) in their characterization 
of the mechanical properties of HSAs.
We refer to the [publication](##Citation) for more details.

Currently, we are able to simulate the behaviour of closed HSAs
(i.e. the printed length is equal to the minimum length). Adaptations would be necessary to accommodate the simulation of 
semi-closed and open HSAs. Specifically, this would include non-linear couplings between the twist strains 
and the material properties such as the axial stiffness.

At the moment, the plugin is only able to simulate the behaviour of a single-segment HSA robot.
This corresponds to a parallel arrangement of HSAs, where each HSA is connected to an electric actuator at its proximal end.
All HSA rods are attached at their distal end to a (circular) platform.
An extension to multi-segment HSA robots would require the implementation of different kinds of constraints,
joints, etc.

## Citation
This simulator is part of the publication **Modelling Handed Shearing Auxetics:
Selective Piecewise Constant Strain Kinematics and Dynamic Simulation** presented at the 
_6th IEEE-RAS International Conference on Soft Robotics (RoboSoft 2023)_. 
You can find the publication online on ~~IEEE Xplore~~.

Please use the following citation if you use our method in your (scientific) work:

```bibtex
@inproceedings{stolzle2023modelling,
  title={Modelling Handed Shearing Auxetics: Selective Piecewise Constant Strain Kinematics and Dynamic Simulation},
  author={St{\"o}lzle, Maximilian and Chin, Lillian and Truby, Ryan L. and Rus, Daniela and Della Santina, Cosimo},
  booktitle={2023 IEEE 6th International Conference on Soft Robotics (RoboSoft)},
  year={2023},
  organization={IEEE}
}
```

## Installation
The plugin can be installed from PyPI:

```bash
pip install hsa-pyelastica
```

or locally from the source code:

```bash
pip install .
```

If you want to run the examples, please install some additional dependencies via:

```bash
pip install ".[examples]"
```

Please note that this plugin is currently compatible with the PyElastica version 0.3.x.
Any future changes in the private interface / API of PyElastica might break the plugin.

## Usage

We provide two simulators: `HsaRodSimulator` and `HsaRobotSim`` to simulate individual HSA rods and assembled 
HSA robots, respectively.

### HSA rod simulator

The `HsaRodSimulator` can be used to simulate the behaviour of individual HSAs.

```python
# import the simulator
from hsa_elastica.forcing import ProximalEndTorsion
from hsa_elastica.simulation import HsaRodSimulator

# define the rod parameters
rod_params = dict(
    name="HSA_ROD",
    outside_radius=25.4e-3 / 2,  # m
    wall_thickness=2.43e-3,  # m
    material_density=1.05e3,  # kg/m^3
    printed_length=100e-3,  # m
    elastic_modulus=1e7,  # Pa
    elastic_modulus_scale_factor=0e0,  # Pa / [rad / m]
    shear_modulus=8e5,  # Pa
    bend_rigidity=2e-2,  # [Nm/(rad/m)] = Nm^2/rad
    twist_rigidity=14e-3,  # [Nm/(rad/m)] = Nm^2/rad
    rest_lengths_scale_factor=1e-2,  # [1/(rad/m) = m / rad]
    translational_damping_constant=[8e3, 8e3, 1e4],
    rotational_damping_constant=[5e0, 5e0, 7e0],
    max_actuation_torque=500e-3,  # Nm
    num_elements=25,
)

# create the simulator (duration and time step in seconds)
# the fps parameter is used to control the frequency of storing 
# the state information of the simulator
sim = HsaRodSimulator(rod_params=rod_params, duration=10.0, dt=5e-5, fps=100)

# configure the simulator. 
# The twist of the distal end of the rod is automatically constrained
sim.configure(finalize=False)

# add ramp actuation torsional_torque
sim.add_forcing_to(sim.rod).using(
    ProximalEndTorsion,
    # 0.5 Nm at the end of the ramp
    torsional_torque=rod_params["max_actuation_torque"],
    ramp_up_time=1.0,  # 1 s ramp up time
)

# finalize the configuration of the simulator
sim.finalize()
# run the simulation
sim.run()

# array of shape (num_frames,), where num_frames is the 
# number of frames stored during the simulation
# num_frames = duration * fps
time = sim.diagnostic_data["time"]
# array of shape (num_frames, 3, num_elements + 1) 
# where num_elements is the number of links of the rod
position = sim.diagnostic_data["position"]
# array of shape (num_frames, num_elements - 1)
twist_angle = sim.diagnostic_data["twist_angle"]
# rest lengths of the rod of shape (num_frames, num_elements)
rest_lengths = sim.diagnostic_data["rest_lengths"]
# rotational strains of the rod of shape (num_frames, 3, num_elements-1)
kappa = sim.diagnostic_data["kappa"]
# linear strains of the rod of shape (num_frames, 3, num_elements)
sigma = sim.diagnostic_data["sigma"]
```

### HSA robot simulator

The `HsaRobotSimulator` can be used to simulate the behaviour of robots consisting of multiple HSAs in parallel configuration.

```python
from hsa_elastica import HsaRobotSimulator
import numpy as np

from examples.hsa_robot.actuation_utils import (
    platform_configuration_to_actuation_angles,
)

# choose the motion primitive you want to simulate
# possible modes:
# "elongation",
# "bending-north", "bending-south", "bending-west", "bending-east"
# "twisting-cw", "twisting-ccw"
MODE = "bending-north"
max_actuation_angle = 179.9 / 180 * np.pi  # [rad]

# define the rod parameters
rod_params = dict(
    name="HSA_ROD_ROBOT_SIM",
    outside_radius=25.4e-3 / 2,  # m
    wall_thickness=2.43e-3,  # m
    # radial offset of rod from xy origin [m] measurement from our robot: 24 mm
    radial_offset=24e-3,
    material_density=1.05e3,  # kg/m^3
    printed_length=100e-3,  # m
    elastic_modulus=1e7,  # Pa
    elastic_modulus_scale_factor=0e0,  # Pa / [rad / m]
    shear_modulus=8e5,  # Pa
    bend_rigidity=2e-2,  # [Nm/(rad/m)] = Nm^2/rad
    twist_rigidity=14e-3,  # [Nm/(rad/m)] = Nm^2/rad
    rest_lengths_scale_factor=1e-2,  # [1/(rad/m) = m / rad]
    translational_damping_constant=[8e3, 8e3, 1e4],
    rotational_damping_constant=[5e0, 5e0, 7e0],
    max_actuation_torque=500e-3,  # Nm
    num_elements=25,
)

# define the segment parameters
# currently, we only support one segment
segment_params = dict(
    # sum of HSA printed length and platform thickness
    L0=0.103,  # m
    platform=dict(
        density=700.0,  # kg/m^3
        radius=50e-3,  # m
        thickness=3e-3,  # m
    ),
    # printed / unextended length of the HSA
    printed_length=rod_params["printed_length"],  # m
    rods=[
        rod_params,
        rod_params,
        rod_params,
        rod_params,
    ]
)

# define the robot parameters
robot_params = dict(
    L0=segment_params["L0"],
    # total number of parallel HSAs 
    num_rods_per_segment=4,
    segments=[segment_params],
)

sim = HsaRobotSimulator(
    name=f"motion_primitives_{MODE}",
    robot_params=robot_params,
    duration=15.0,
    dt=4e-5,
    fps=100,
)

# do not finalize the simulator yet
# do not add constraints on the proximal end of the HSAs yet
sim.configure(finalize=False, add_constraints=False)

max_platform_magnitude = 4 * max_actuation_angle
# match requires Python 3.10
match MODE:
    # define the desired platform configuration
    # consists of bending around x-axis, y-axis, twist around z-axis and elongation
    case "elongation":
        q_p_des = 1.0 * max_platform_magnitude * np.array([0.0, 0.0, 0.0, 1.0])
    case "bending-north":
        q_p_des = 1.0 * max_platform_magnitude * np.array([0.0, 0.5, 0.0, 0.5])
    case "bending-south":
        q_p_des = 1.0 * max_platform_magnitude * np.array([0.0, -0.5, 0.0, 0.5])
    case "bending-west":
        q_p_des = 1.0 * max_platform_magnitude * np.array([-0.5, 0.0, 0.0, 0.5])
    case "bending-east":
        q_p_des = 1.0 * max_platform_magnitude * np.array([0.5, 0.0, 0.0, 0.5])
    case "twisting-cw":
        q_p_des = 1.0 * max_platform_magnitude * np.array([0.0, 0.0, -0.5, 0.5])
    case "twisting-ccw":
        q_p_des = 1.0 * max_platform_magnitude * np.array([0.0, 0.0, 0.5, 0.5])
    case _:
        raise NotImplementedError(f"Mode {MODE} not implemented")

# map the motion primitive to twisting angles of the proximal ends of the HSAs 
# (e.g. actuation angles)
actuation_angles = platform_configuration_to_actuation_angles(
    q_p_des, max_actuation_angle
)

# enforce the actuation angles by setting constrains on the proximal ends of the HSAs
# the ramp up time is used to smoothly ramp up the actuation angles from 0 to the desired value
sim.add_controlled_boundary_conditions(
    # add segment-dimension to actuation angles
    actuation_angles=np.expand_dims(actuation_angles, axis=0),
    ramp_up_time=5.0,
)

# manually finalize the simulator
sim.finalize()
# run the simulation
sim.run()
# save the diagnostic data to
sim.save_diagnostic_data()
```

## Getting started

We provide several scripts in the `examples` folder to get you started with the plugin.

### Verification of rod simulator against experimental results by Good et al. [[1]](#1)

The command

```bash
python examples/hsa_rod/verification_sim_against_good_et_al/hsa_rod_steady_state_characterisation.py
```

simulates the steady-state behaviour of individual HSA rods to identify several mechanical characteristics such as 
holding torque, minimum energy length, spring constant, and blocked force (e.g. Figure 3 of the paper).
We have tuned the parameters of the simulator to match the experimental results of Good et al. [[1]](#1) as closely as 
possible.
Finally, we compare the results of the simulation with the experimental results to verify the steady-state behaviour of the
simulator. This can be done by running the following command:

```bash
python examples/hsa_rod/verification_sim_against_good_et_al/plot_verification_sim_against_good_et_al.py
```

### Motion primitives of HSA robots

The typical motion primitives such as elongation, bending, and twisting can be simulated dynamically by running the 
script

```bash
python examples/hsa_robot/motion_primitives.py
```

Subsequently, the script

```bash
python examples/hsa_robot/motion_primitives_postprocessing.py
```

can be used to visualize and animate the simulation data as shown in Figure 2 of the paper.

### Generation of steady-state dataset

The command

```bash
python examples/hsa_robot/generate_steady_state_dataset.py
```

will generate a dataset of steady-state configurations of HSA robots. 
Namely, we explore configurations of the robot in its entire workspace, be it through the means of random sampling
or stepping through a specified grid.

In the paper, this steady-state dataset was used to fit the SPCS kinematic parametrization to the shape of the HSA rods
at steady-state and evaluate the accuracy of the reconstructed shape. The results are reported in Figure 5 of the paper.

## See also

You might also be interested in the following repositories:

 - The [`jax-spcs-kinematics`](https://github.com/tud-cor-sr/jax-spcs-kinematics) repository contains an implementation
 of the Selective Piecewise Constant Strain (SPCS) kinematics in JAX. We have shown in our paper that this kinematic 
model is suitable for representing the shape of HSA rods.
 - You can find code and datasets used for the verification of the SPCS model for HSA robots in the 
[`hsa-kinematic-model`](https://github.com/tud-cor-sr/hsa-kinematic-model) repository.

## References
<a id="1">[1]</a> Good, Ian, et al. 
"Expanding the Design Space for Electrically-Driven Soft Robots Through Handed Shearing Auxetics." 
2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022.
