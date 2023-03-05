from typing import *

from .segment_params import DEFAULT_SEGMENT_PARAMS


def construct_robot_params(num_segments: int, segment_params: Dict) -> Dict:
    robot_params = {
        "segments": [],
        "num_rods_per_segment": len(segment_params["rods"]),
        "L0": 0.0,
    }
    for i in range(1, num_segments + 1):
        robot_params["segments"].append(segment_params.copy())
        robot_params["L0"] += segment_params["L0"]

    return robot_params


ONE_SEGMENT_ROBOT = construct_robot_params(
    num_segments=1, segment_params=DEFAULT_SEGMENT_PARAMS
)
