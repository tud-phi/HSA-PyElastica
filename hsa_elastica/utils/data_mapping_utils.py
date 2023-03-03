import numpy as np
from typing import *


def simulation_diagnostic_arrays_to_transformation_matrices(
    rod_diagnostic_arrays: Dict[str, np.ndarray] = None,
    platform_diagnostic_arrays: Dict[str, np.ndarray] = None,
) -> Tuple[np.array, np.array]:
    T_rod_ts = None
    T_platform_ts = None

    if rod_diagnostic_arrays is not None:
        position = rod_diagnostic_arrays["position"]
        directors = rod_diagnostic_arrays["directors"]

        # compute centers of the links from node positions
        t_rod_ts = 0.5 * position[..., :-1] + 0.5 * position[..., 1:]

        rot_mat = directors.transpose((0, 1, 2, 4, 3, 5))

        T_rod_ts = np.zeros(
            (
                directors.shape[0],  # number of time steps
                directors.shape[1],  # number of segments
                directors.shape[2],  # number of rods
                4,
                4,
                directors.shape[-1],  # number of links
            )
        )
        T_rod_ts[:, :, :, :3, :3, :] = rot_mat
        T_rod_ts[:, :, :, :3, 3, :] = t_rod_ts
        T_rod_ts[:, :, :, 3, 3, :] = 1.0

    if platform_diagnostic_arrays is not None:
        position = platform_diagnostic_arrays["position"]
        directors = platform_diagnostic_arrays["directors"]
        T_platform_ts = np.zeros((position.shape[0], position.shape[1], 4, 4))
        T_platform_ts[:, :, :3, :3] = directors[..., 0].transpose((0, 1, 3, 2))
        T_platform_ts[:, :, :3, 3] = position[..., 0]
        T_platform_ts[:, :, 3, 3] = 1.0

    return T_rod_ts, T_platform_ts
