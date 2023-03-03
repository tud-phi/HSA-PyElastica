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

Please note that this plugin is currently compatible with the PyElastica version 0.3.x.
Any future changes in the private interface / API of PyElastica might break the plugin.

## Usage

We provide two simulators: `HsaRodSimulator` and `HsaRobotSim`` to simulate individual HSA rods and assembled 
HSA robots, respectively.

### HsaRodSimulator

The `HsaRodSimulator` can be used to simulate the behaviour of individual HSAs.
First, we define the rod parameters:

```python
```

## Getting started

We provide several scripts in the `examples` folder to get you started with the plugin.

### Verification of simulator against experimental results by Good et al. [[1]](#1)

The command

```bash
python examples/hsa_rod/verification_sim_against_good_et_al/hsa_rod_steady_state_characterisation.py
```

simulates the steady-state behaviour of individual HSA rods to identify several mechanical characteristics such as 
holding torque, minimum energy length, spring constant, and blocked force.
We have tuned the parameters of the simulator to match the experimental results of Good et al. [[1]](#1) as closely as 
possible.
Finally, we compare the results of the simulation with the experimental results to verify the steady-state behaviour of the
simulator. This can be done by running the following command:

```bash
python examples/hsa_rod/verification_sim_against_good_et_al/plot_verification_sim_against_good_et_al.py
```

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
