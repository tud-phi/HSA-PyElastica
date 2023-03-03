import numpy as np
from typing import Dict

from .platform_params import *
from .rod_params import *


def construct_segment_params(
    num_rods: int,
    rod_params: Dict,
    platform_params: Dict,
) -> Dict:
    segment_params = {
        "rods": [],
        "platform": platform_params.copy(),
        "printed_length": rod_params["printed_length"],
        "L0": rod_params["printed_length"] + platform_params["thickness"],
    }
    for j in range(num_rods):
        rod_params_j = rod_params.copy()
        rod_params_j["rod_idx"] = j
        rod_params_j["phi"] = (0.5 + j) * 2 * np.pi / num_rods
        rod_params_j["handedness"] = "right" if j % 2 == 0 else "left"
        segment_params["rods"].append(rod_params_j)

    return segment_params


DEFAULT_SEGMENT_PARAMS = construct_segment_params(
    num_rods=4,
    rod_params=ROD_ROBOT_SIM,
    platform_params=PLATFORM_PARAMS,
)
