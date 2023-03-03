import numpy as np
from typing import Tuple


# handedness of rods
h = np.array([1, -1, 1, -1])
# mapping from actuation-space to configuration-space
H_p = np.array(
    [
        [1, 1, -1, -1],
        [-1, 1, 1, -1],
        [1, -1, 1, -1],
        [1, 1, 1, 1],
    ]
) @ np.diag(h)


def platform_configuration_to_actuation_angles(
    q_p_des: np.ndarray, u_max: float, eps: float = 1e-6
) -> np.ndarray:
    """
    :param: q_p_des: desired configuration of the platform of shape (4,)
    :param: u_max: magnitude of maximum actuation angle of the rods
    :return: u: actuation angles of shape (4,)
    """
    u = np.linalg.inv(H_p) @ q_p_des
    min_extension = np.min(h * u)
    # sanitization as we cannot apply negative actuation angles
    u_tilde = u - (h * np.min(np.stack([0.0, min_extension])))
    # sanitization to not exceed maximum
    # actuation magnitude ratio
    am_ratio = u_max / (np.max(np.abs(u_tilde)) + eps)
    u_bar = u_tilde * np.min(np.stack([np.ones_like(am_ratio), am_ratio]), axis=0)
    # clip the actuation angles just to be sure we don't have any numerical issues
    u = u_bar.copy()
    u[h == 1] = np.clip(u[h == 1], a_min=0, a_max=u_max)
    u[h == -1] = np.clip(u[h == -1], a_min=-u_max, a_max=0)
    return u


def actuation_angles_to_platform_configuration(u: np.ndarray):
    q_p = H_p @ u
    return q_p


def generate_actuation_samples(
    num_samples: int, max_actuation_angle: float, mode="elongation", seed: int = 0
) -> Tuple[np.array, np.array]:
    """
    Generate a set of desired platform configurations for each mode
    :param: num_samples: number of desired platform configurations
    :param: max_actuation_angle: maximum actuation angle of the rods
    :param: mode: desired actuation mode. Allowed values: "elongation", "bending", "torsion", "any"
    :param: seed: random seed (important for "combined" mode)
    :return: desired platform configurations as np.array of shape (num_samples, 4)
    """
    rng = np.random.default_rng(seed=seed)

    max_platform_magnitude = 4 * max_actuation_angle
    q_p_ss = np.zeros((num_samples, 4))
    u_ss = np.zeros((num_samples, 4))
    if mode == "elongation":
        actuation_magnitudes = np.linspace(0, max_platform_magnitude, num_samples)
        for _sample_idx, actuation_magnitude in enumerate(actuation_magnitudes):
            q_p_d = actuation_magnitude * np.array([0, 0, 0, 1])
            q_p_ss[_sample_idx] = q_p_d
            u_ss[_sample_idx] = platform_configuration_to_actuation_angles(
                q_p_d, max_actuation_angle
            )
    elif mode == "bending":
        _sample_idx = 0
        azimuth_angles = np.linspace(0, 2 * np.pi, int(np.sqrt(num_samples)))

        bending_range_dim = int(np.sqrt(num_samples))
        max_bending_magnitude = max_platform_magnitude / (np.sqrt(2) * 2)
        bending_angles = np.linspace(
            start=0.2 * max_bending_magnitude,
            stop=max_bending_magnitude,
            num=bending_range_dim,
        )

        for bending_angle in bending_angles:
            for azimuth_angle in azimuth_angles:
                q_p_d = np.array(
                    [
                        bending_angle * np.cos(azimuth_angle),
                        bending_angle * np.sin(azimuth_angle),
                        0,
                        np.sqrt(2) * max_bending_magnitude,
                    ]
                )
                u = platform_configuration_to_actuation_angles(
                    q_p_d, max_actuation_angle
                )
                # actually reachable configuration
                q_p = actuation_angles_to_platform_configuration(u)
                q_p_ss[_sample_idx, :], u_ss[_sample_idx, :] = q_p, u

                _sample_idx += 1

        # cut-off left-over zero-entries
        q_p_ss = q_p_ss[: _sample_idx + 1]
        u_ss = u_ss[: _sample_idx + 1]
    elif mode == "twisting":
        max_twisting_magnitude = max_platform_magnitude / 2
        twist_angles = np.linspace(
            start=-max_twisting_magnitude, stop=max_twisting_magnitude, num=num_samples
        )
        for _sample_idx, twist_angle in enumerate(twist_angles):
            q_p_d = np.array([0, 0, twist_angle, max_twisting_magnitude])
            q_p_ss[_sample_idx] = q_p_d
            u_ss[_sample_idx] = platform_configuration_to_actuation_angles(
                q_p_d, max_actuation_angle
            )
    elif mode == "combined":
        max_bend_twist_magnitude = max_platform_magnitude / 2
        for _sample_idx in range(num_samples):
            bending_x = rng.uniform(-max_bend_twist_magnitude, max_bend_twist_magnitude)
            bending_y_magnitude = rng.uniform(
                0, max_bend_twist_magnitude - np.abs(bending_x)
            )
            bending_y = rng.choice([-bending_y_magnitude, bending_y_magnitude])
            twist_magnitude = rng.uniform(
                0, max_bend_twist_magnitude - np.abs(bending_x) - np.abs(bending_y)
            )
            twist_angle = rng.choice([-1, 1]) * twist_magnitude
            elongation = rng.uniform(
                low=np.abs(bending_x) + np.abs(bending_y) + np.abs(twist_angle),
                high=max_platform_magnitude
                - np.abs(bending_x)
                - np.abs(bending_y)
                - np.abs(twist_angle),
            )
            q_p_d = np.array(
                [
                    bending_x,
                    bending_y,
                    twist_angle,
                    elongation,
                ]
            )
            u = platform_configuration_to_actuation_angles(q_p_d, max_actuation_angle)
            # actually reachable configuration
            q_p = actuation_angles_to_platform_configuration(u)
            q_p_ss[_sample_idx, :], u_ss[_sample_idx, :] = q_p, u
    elif mode == "lemniscate":
        steps = np.linspace(0, 1, num_samples)
        bending_x = max_platform_magnitude / 4 * np.sin(1 * 2 * np.pi * steps)
        bending_y = max_platform_magnitude / 4 * np.sin(2 * 2 * np.pi * steps)
        for _sample_idx, step in enumerate(steps):
            q_p_d = np.array(
                [
                    bending_x[_sample_idx],
                    bending_y[_sample_idx],
                    0,
                    max_platform_magnitude / 2,
                ]
            )
            q_p_ss[_sample_idx] = q_p_d
            u_ss[_sample_idx] = platform_configuration_to_actuation_angles(
                q_p_d, max_actuation_angle
            )
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    return q_p_ss, u_ss
