from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from examples.hsa_robot.actuation_utils import generate_actuation_samples

# possible modes: "elongation", "bending", "twisting", "combined", "lemniscate"
mode = "elongation"
seed = 101
max_actuation_angle = 1
num_samples = 100
now = datetime.now()  # current date and time
filename = f"{now.strftime('%Y%m%d_%H%M%S')}_actuation_specs_{mode}_seed-{seed}_{num_samples}-samples"
dataset_dir = Path(f"../hsa_actuation_matlab/actuation_specs")
filepath = dataset_dir / f"{filename}.csv"

if __name__ == "__main__":
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
    df.to_csv(str(filepath), index=True, index_label="sample_idx")

    print(
        f"Generated {num_samples} actuation samples of {mode} and saved them to {filepath.resolve()}"
    )
