from collections import defaultdict
from contextlib import contextmanager
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import progressbar
from typing import Dict, List, Optional, Union

from .base_scene import BaseScene
from ..matplotlib_3d_plot_utils import (
    _set_axes_equal,
    Arrow3D,
    _arrow3D,
)


class MatplotlibScene(BaseScene):
    def __init__(self, robot_params: Dict):
        self.robot_params = robot_params

        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self.lines: Dict = defaultdict(list)
        self.cmap = plt.get_cmap("tab10")
        self.movie_writer = None

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Romand"],
            }
        )
        setattr(Axes3D, "arrow3D", _arrow3D)

    @contextmanager
    def movie(self, *args, **kwargs):
        try:
            self._setup_movie(*args, **kwargs)
            yield self
        finally:
            self._finish_movie()

    def _setup_movie(self, filepath: str, fps: int = 1):
        num = "HSA Simulator Matplotlib Scene Animation"
        self._init_figure(num=num)

        # self.movie_writer = animation.writers['ffmpeg'](fps=frame_rate)
        metadata = dict(title="HSA Scene", artist="Maximilian Stölzle")
        self.movie_writer = FFMpegWriter(fps=fps, metadata=metadata)
        self.movie_writer.setup(self.fig, outfile=filepath, dpi=600)

    def _finish_movie(self):
        self.movie_writer.finish()

    def plot(
        self,
        t: float = None,
        rod_diagnostic_arrays: Dict[str, np.array] = None,
        platform_diagnostic_arrays: Dict[str, np.array] = None,
        show: bool = True,
        filepath: str = None,
        *args,
        **kwargs,
    ):
        time_str = " at time t = {:.2f} s".format(t) if t is not None else ""
        num = "HSA Simulator Matplotlib Scene" + time_str
        self._init_figure(num=num)

        self._draw_scene(
            rod_data=rod_diagnostic_arrays, platform_data=platform_diagnostic_arrays
        )

        if filepath is not None:
            plt.savefig(filepath)
        if show:
            plt.show()

    def animate(
        self,
        rod_diagnostic_arrays: Dict[str, np.array],
        platform_diagnostic_arrays: Dict[str, np.array],
        **kwargs,
    ):
        self._setup_movie(**kwargs)
        print("Rendering movie frames...")
        for time_idx in progressbar.progressbar(
            range(rod_diagnostic_arrays["time"].shape[0])
        ):
            rod_diagnostic_array_t = {
                key: value[time_idx, ...]
                for key, value in rod_diagnostic_arrays.items()
            }
            platform_diagnostic_array_t = {
                key: value[time_idx, ...]
                for key, value in platform_diagnostic_arrays.items()
            }
            self.run_step(
                rod_diagnostic_arrays=rod_diagnostic_array_t,
                platform_diagnostic_arrays=platform_diagnostic_array_t,
            )
        self._finish_movie()

    def run_step(
        self,
        rod_diagnostic_arrays: Dict[str, np.array] = None,
        platform_diagnostic_arrays: Dict[str, np.array] = None,
    ):
        if self.ax is None:
            self._draw_scene(
                rod_data=rod_diagnostic_arrays, platform_data=platform_diagnostic_arrays
            )
        else:
            self._update_scene(
                rod_data=rod_diagnostic_arrays, platform_data=platform_diagnostic_arrays
            )

        self.movie_writer.grab_frame()

    def _init_figure(self, num: Union[int, str, Figure] = None):
        self.fig = plt.figure(num, frameon=True, dpi=150)

    def _draw_scene(
        self,
        rod_data: Dict[str, np.array],
        platform_data: Dict[str, np.array] = None,
    ):
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect([1, 1, 1])
        _set_axes_equal(self.ax)  # IMPORTANT - this is also required
        # markers = list(MarkerStyle.markers.keys())

        for i in range(1, rod_data["step"].shape[0] + 1):
            segment_lines = defaultdict(list)
            printed_length = self.robot_params["segments"][i - 1]["printed_length"]
            for j in range(rod_data["step"].shape[1]):
                (backbone_line,) = self.ax.plot(
                    rod_data["position"][i - 1, j, 0, :],
                    rod_data["position"][i - 1, j, 1, :],
                    rod_data["position"][i - 1, j, 2, :],
                    linestyle="-",
                    linewidth=1,
                    marker="^" if j % 2 == 0 else "v",
                    markersize=2,
                    color=self.cmap(i - 1),
                    label=f"Rod {i}.{j}",
                )

                rod_lines = {"backbone_line": backbone_line, "orientation_arrows": []}
                segment_lines["rods"].append(rod_lines)

                num_elements = rod_data["directors"].shape[-1]
                for link_idx in [
                    0,
                    int(num_elements // 4),
                    int(num_elements // 2),
                    int(num_elements * 3 / 4),
                    -1,
                ]:
                    arrows = self._draw_link_orientation(
                        rod_data["position"][i - 1, j, :, link_idx],
                        rod_data["directors"][i - 1, j, ..., link_idx].T,
                        oal=0.1 * printed_length,
                    )
                    rod_lines["orientation_arrows"].append(
                        {"link_idx": link_idx} | arrows
                    )

            if platform_data is not None:
                self.plot_platform(
                    segment_idx=i,
                    segment_lines=segment_lines,
                    platform_data=platform_data,
                )

            self.lines["segments"].append(segment_lines)

        # set axis labels
        zlim_min = rod_data["position"][..., 2, :].min()
        zlim_max = 1.5 * rod_data["position"][..., 2, :].max()
        self.ax.set_xlim((-zlim_max / 2, zlim_max / 2))
        self.ax.set_ylim((-zlim_max / 2, zlim_max / 2))
        self.ax.set_zlim((zlim_min, zlim_max))

        # self.ax.legend()
        self.ax.set_xlabel("$x$ [m]")
        self.ax.set_ylabel("$y$ [m]")
        self.ax.set_zlabel("$z$ [m]")

        # set ticks
        plt.locator_params(axis="x", nbins=3)
        plt.locator_params(axis="y", nbins=3)

    def _draw_link_orientation(
        self, t: np.ndarray, R: np.ndarray, oal: float, opacity: float = 1.0
    ) -> Dict[str, Arrow3D]:
        """
        :param t: translation vector of the node
        :param R: rotation matrix of the link
        :param oal: orientation arrow length
        :return:
        """
        link_lines = {}

        normal = oal * R[:, 0]
        binormal = oal * R[:, 1]
        tangent = oal * R[:, 2]
        link_lines["normal"] = self.ax.arrow3D(
            x=t[0],
            y=t[1],
            z=t[2],
            dx=normal[0],
            dy=normal[1],
            dz=normal[2],
            color="red",
            mutation_scale=3,
            alpha=opacity,
        )
        link_lines["binormal"] = self.ax.arrow3D(
            x=t[0],
            y=t[1],
            z=t[2],
            dx=binormal[0],
            dy=binormal[1],
            dz=binormal[2],
            color="green",
            mutation_scale=3,
        )
        link_lines["tangent"] = self.ax.arrow3D(
            x=t[0],
            y=t[1],
            z=t[2],
            dx=tangent[0],
            dy=tangent[1],
            dz=tangent[2],
            color="blue",
            mutation_scale=3,
        )
        link_lines["oal"] = oal
        return link_lines

    def _update_scene(
        self,
        rod_data: Dict[str, np.array],
        platform_data: Dict[str, np.array] = None,
    ):
        for i, segment_lines in enumerate(self.lines["segments"], start=1):
            if "rods" in segment_lines:
                for j, rod_lines in enumerate(segment_lines["rods"]):
                    rod_lines["backbone_line"].set_data(
                        rod_data["position"][i - 1, j, 0, :],
                        rod_data["position"][i - 1, j, 1, :],
                    )
                    rod_lines["backbone_line"].set_3d_properties(
                        rod_data["position"][i - 1, j, 2, :]
                    )
                    for orientation_arrow in rod_lines["orientation_arrows"]:
                        link_idx = orientation_arrow["link_idx"]
                        # remove old arrows
                        orientation_arrow["normal"].remove()
                        orientation_arrow["binormal"].remove()
                        orientation_arrow["tangent"].remove()
                        new_arrows = self._draw_link_orientation(
                            rod_data["position"][i - 1, j, :, link_idx],
                            rod_data["directors"][i - 1, j, ..., link_idx].T,
                            oal=orientation_arrow["oal"],
                        )
                        # update dictionary with new arrows
                        orientation_arrow.update(new_arrows)

            if "platform_curved_surface" in segment_lines:
                segment_lines["platform_curved_surface"].remove()
            if "platform_bottom_surface" in segment_lines:
                segment_lines["platform_bottom_surface"].remove()
            if "platform_top_surface" in segment_lines:
                segment_lines["platform_top_surface"].remove()
            if platform_data is not None:
                self.plot_platform(
                    segment_idx=i,
                    segment_lines=segment_lines,
                    platform_data=platform_data,
                )

    def plot_platform(
        self,
        segment_idx: int,
        segment_lines: Dict,
        platform_data: Dict[str, np.array],
    ):
        i = segment_idx
        platform_params = self.robot_params["segments"][i - 1]["platform"]
        grid_x, grid_y, grid_z = data_for_cylinder_curved_surface(
            radius=platform_params["radius"],
            height=platform_params["thickness"],
            center=platform_data["position"][i - 1, ..., 0],
            directors=platform_data["directors"][i - 1, ..., 0],
        )
        segment_lines["platform_curved_surface"] = self.ax.plot_surface(
            grid_x,
            grid_y,
            grid_z,
            alpha=0.4,
            color=self.cmap(i - 1),
            label=f"Platform {i}",
        )
        grid_x, grid_y, grid_z = data_for_cylinder_face_surface(
            radius=platform_params["radius"],
            center=platform_data["position"][i - 1, ..., 0],
            directors=platform_data["directors"][i - 1, ..., 0],
            axis_offset=-platform_params["thickness"] / 2,
        )
        segment_lines["platform_bottom_surface"] = self.ax.plot_surface(
            grid_x,
            grid_y,
            grid_z,
            alpha=0.4,
            color=self.cmap(i - 1),
        )
        grid_x, grid_y, grid_z = data_for_cylinder_face_surface(
            radius=platform_params["radius"],
            center=platform_data["position"][i - 1, ..., 0],
            directors=platform_data["directors"][i - 1, ..., 0],
            axis_offset=platform_params["thickness"] / 2,
        )
        segment_lines["platform_top_surface"] = self.ax.plot_surface(
            grid_x,
            grid_y,
            grid_z,
            alpha=0.4,
            color=self.cmap(i - 1),
        )


def data_for_cylinder_curved_surface(
    radius: float,
    height: float,
    center: np.array,
    directors: np.ndarray,
    num_points: int = 50,
):
    # initialize cylinder in local frame
    z = np.linspace(-height / 2, height / 2, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    grid = np.stack((x_grid, y_grid, z_grid), axis=2)  # num_points x 3

    directors_repeated = np.repeat(
        directors.T.reshape(1, 3, 3), repeats=num_points * num_points, axis=0
    )

    # rotate into inertia frame
    grid = np.matmul(directors_repeated, grid.reshape((-1, 3, 1))).reshape(
        (num_points, num_points, 3)
    )

    # add translational offset
    grid += center

    return grid[..., 0], grid[..., 1], grid[..., 2]


def data_for_cylinder_face_surface(
    radius: float,
    center: np.array,
    directors: np.ndarray,
    axis_offset: float,
    num_points: int = 50,
):
    # initialize cylinder in local frame
    r = np.linspace(0, radius, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    theta_grid, r_grid = np.meshgrid(theta, r)

    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    z_grid = axis_offset * np.ones(x_grid.shape)
    grid = np.stack((x_grid, y_grid, z_grid), axis=2)  # num_points x 3

    directors_repeated = np.repeat(
        directors.T.reshape(1, 3, 3), repeats=num_points * num_points, axis=0
    )
    # rotate into inertia frame
    grid = np.matmul(directors_repeated, grid.reshape((-1, 3, 1))).reshape(
        (num_points, num_points, 3)
    )

    # add translational offset
    grid += center

    return grid[..., 0], grid[..., 1], grid[..., 2]
