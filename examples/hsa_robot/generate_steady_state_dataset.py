"""
Generate dataset consisting of steady state data for the HSA robot for a given set of actuation angles.
"""
from datetime import datetime
from hsa_elastica import HsaRobotSimulator
import numpy as np
import pandas as pd
from pathlib import Path

from .actuation_utils import generate_actuation_samples
from examples.parameters.robot_params import ONE_SEGMENT_ROBOT
from examples.visualization.scene import MatplotlibScene


# possible modes: "elongation", "bending", "lemniscate", "twisting", "combined"
mode = "elongation"
seed = 101
num_samples = 100
max_actuation_angle = 179.9 / 180 * np.pi  # [rad]
now = datetime.now()  # current date and time
dataset_name = (
    f"{now.strftime('%Y%m%d_%H%M%S')}_{mode}_seed-{seed}_{num_samples}-samples"
)
dataset_dir = Path(f"data/hsa_robot/kinematic_steady_state_datasets/{dataset_name}")


if __name__ == "__main__":
    robot_params = ONE_SEGMENT_ROBOT
    dataset_dir.mkdir(parents=True, exist_ok=True)

    q_p_ss, u_ss = generate_actuation_samples(
        num_samples=num_samples,
        max_actuation_angle=max_actuation_angle,
        mode=mode,
        seed=seed,
    )
    df = pd.DataFrame(
        {
            "q_p_ss_0": q_p_ss[:, 0],
            "q_p_ss_1": q_p_ss[:, 1],
            "q_p_ss_2": q_p_ss[:, 2],
            "q_p_ss_3": q_p_ss[:, 3],
            "u_ss_0": u_ss[:, 0],
            "u_ss_1": u_ss[:, 1],
            "u_ss_2": u_ss[:, 2],
            "u_ss_3": u_ss[:, 3],
        }
    )
    df.to_csv(
        str(dataset_dir / "actuation_specs.csv"), index=True, index_label="sample_idx"
    )

    for sample_idx in range(q_p_ss.shape[0]):
        print(f"Simulating sample {sample_idx + 1} / {q_p_ss.shape[0]}")
        sample_name = (
            f"{now.strftime('%Y%m%d_%H%M%S')}_{mode}_seed-{seed}_sample-{sample_idx}"
        )
        sample_dir = dataset_dir / sample_name
        sim = HsaRobotSimulator(
            name=sample_name,
            robot_params=robot_params,
            duration=15.0,
            dt=4e-5,
            fps=100,
            log_dir=str(sample_dir),
        )
        sim.configure(finalize=False, add_constraints=False)

        # if mode == "any":
        #     # randomly sample the mode for each sample
        #     it_mode = np.random.choice(["elongation", "bending", "torsion"])
        # else:
        #     it_mode = mode

        q_p, u = q_p_ss[sample_idx, :], u_ss[sample_idx, :]

        print(f"Apply actuation angles: {u / np.pi * 180} deg")

        sim.add_controlled_boundary_conditions(
            actuation_angles=np.expand_dims(u, axis=0),
            ramp_up_time=5.0,
        )

        sim.finalize()  # manually finalize the simulator
        sim.run()
        sim.save_diagnostic_data()

        print(f"Saved diagnostic data for sample {sample_idx + 1} to {sample_dir}")

        # extract final state
        rod_diagnostic_array_tf = {
            key: value[-1, ...] for key, value in sim.rod_diagnostic_arrays.items()
        }
        platform_diagnostic_array_tf = {
            key: value[-1, ...] for key, value in sim.platform_diagnostic_arrays.items()
        }

        plt_scene = MatplotlibScene(robot_params=robot_params)
        plt_scene.plot(
            t=platform_diagnostic_array_tf["time"].item(),
            rod_diagnostic_arrays=rod_diagnostic_array_tf,
            platform_diagnostic_arrays=platform_diagnostic_array_tf,
            show=False,
            filepath=str(sample_dir / f"final_state_plt.pdf"),
        )
