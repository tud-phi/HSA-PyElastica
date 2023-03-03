from contextlib import contextmanager
import jax
import jax.numpy as jnp
import numpy as np
import progressbar
import pyvista as pv
import seaborn as sns
from typing import *
import warnings

from .base_scene import BaseScene
import spcs_kinematics.jax_math as jmath
from hsa_elastica.utils.check_freq_activation import check_freq_activation


class PyvistaScene(BaseScene):
    """
    A visualizer for the HSA robot using PyVista.
    """

    def __init__(
        self,
        robot_params: Dict,
        gt_settings: Dict = {},
        hat_settings: Dict = {},
        enable_shadows: bool = False,
        floor_size: Tuple = None,
        **kwargs,
    ):
        self.robot_params = robot_params
        self.L0 = robot_params["L0"]

        self.gt_settings, self.hat_settings = gt_settings, hat_settings

        self.enable_shadows = enable_shadows
        if enable_shadows:
            warnings.warn(
                "Shadows are not supported by PyVista / VTK for meshes with opacity"
            )
        self.floor_size = (
            floor_size if floor_size is not None else (2 * self.L0, 2 * self.L0)
        )
        self.backbone_radius = 0.002
        self.platform_resolution = 100

        self.filepath = None
        self.t, self.sample_rate, self.frame_rate = 0.0, 1.0, 1.0

        self.outputs_gt, self.outputs_hat = None, None

        # pyvista plotter
        self.pl = None

        # jax function for rapidly computing the rod mesh
        self.rod_mesh_points_fun = jax.jit(
            jax.vmap(
                fun=jmath.generate_infinitesimal_cylindrical_mesh_points,
                in_axes=(-1, None, None, None, None),
                out_axes=-1,
            ),
            static_argnames=("r_resolution", "phi_resolution"),
        )

    def run(
        self,
        T_rod_gt: np.ndarray = None,
        T_platform_gt: np.ndarray = None,
        T_rod_hat: np.ndarray = None,
        T_platform_hat: np.ndarray = None,
        filepath: str = None,
    ):
        self.draw_scene(
            T_rod_gt=T_rod_gt,
            T_platform_gt=T_platform_gt,
            T_rod_hat=T_rod_hat,
            T_platform_hat=T_platform_hat,
        )

        self.pl.show(auto_close=False)

        if filepath is not None:
            # self.pl.window_size = (2500, 2500)
            self.pl.ren_win.SetOffScreenRendering(True)
            self.pl.save_graphic(filepath)

        self.pl.close()

    @contextmanager
    def movie(self, *args, **kwargs):
        try:
            self._setup_movie(*args, **kwargs)
            yield self
        finally:
            self._finish_movie()

    def _setup_movie(
        self, filepath: str, sample_rate: float = 40, frame_rate: float = 20
    ):
        # this method should be run once at the start when creating a movie
        assert (
            frame_rate <= sample_rate
        ), "frame rate of movie should be less than or equal to sample rate"
        assert (
            sample_rate % frame_rate == 0
        ), "sample rate of movie should be a multiple of frame rate"

        self.filepath, self.sample_rate, self.frame_rate = (
            filepath,
            sample_rate,
            frame_rate,
        )

    def _finish_movie(self):
        self.close()

    def animate(
        self,
        T_rod_gt_ts: np.ndarray = None,
        T_platform_gt_ts: np.ndarray = None,
        T_rod_hat_ts: np.ndarray = None,
        T_platform_hat_ts: np.ndarray = None,
        **kwargs,
    ):
        """
        Animate a trajectory of the HSA robot in PyVista.
        :param T_rod_gt_ts: (n_t, n_s, n_r, 4, 4, N) array of ground truth rod poses
        :param T_platform_gt_ts: (n_t, n_s, 4, 4) array of ground truth platform poses
        :param T_rod_hat_ts: (n_t, n_s, n_r, 4, 4) array of estimated rod poses
        :param T_platform_hat_ts: (n_t, n_s, 4, 4, N) array of estimated platform poses
            where n_t is the number of timesteps, n_s is the number of segments, n_r is the number of rods,
            and N is the number of points on the backbone spline.
        """
        self._setup_movie(**kwargs)
        print("Rendering movie frames...")
        for time_idx in progressbar.progressbar(range(T_rod_gt_ts.shape[0])):
            self.run_timestep(
                None if T_rod_gt_ts is None else T_rod_gt_ts[time_idx],
                None if T_platform_gt_ts is None else T_platform_gt_ts[time_idx],
                None if T_rod_hat_ts is None else T_rod_hat_ts[time_idx],
                None if T_platform_hat_ts is None else T_platform_hat_ts[time_idx],
            )
        self._finish_movie()

    def run_timestep(
        self,
        T_rod_gt: np.ndarray = None,
        T_platform_gt: np.ndarray = None,
        T_rod_hat: np.ndarray = None,
        T_platform_hat: np.ndarray = None,
    ):
        if self.t == 0:
            # somehow diffusion is very inconsistent with overlapping meshes from frame to frame
            diffuse = 0.5
            if T_rod_gt is not None and T_rod_hat is not None:
                diffuse = 0.0

            self.draw_scene(
                T_rod_gt=T_rod_gt,
                T_platform_gt=T_platform_gt,
                T_rod_hat=T_rod_hat,
                T_platform_hat=T_platform_hat,
            )

            self.pl.open_movie(self.filepath, framerate=self.frame_rate, quality=9)

            self.pl.show(auto_close=False)

            self.pl.ren_win.SetOffScreenRendering(True)

            self.pl.write_frame()  # Write this frame

        elif check_freq_activation(self.t, 1 / self.frame_rate):
            self.update_scene(
                T_rod_gt=T_rod_gt,
                T_platform_gt=T_platform_gt,
                T_rod_hat=T_rod_hat,
                T_platform_hat=T_platform_hat,
            )

            self.pl.write_frame()  # Write this frame

        self.t = self.t + 1.0 / self.sample_rate

    def draw_scene(
        self,
        T_rod_gt: np.ndarray = None,
        T_platform_gt: np.ndarray = None,
        T_rod_hat: np.ndarray = None,
        T_platform_hat: np.ndarray = None,
    ):
        # create plotter
        plotter_kwargs = {"window_size": [1500, 1500], "lighting": "none"}
        self.pl = pv.Plotter(**plotter_kwargs)

        if T_rod_gt is not None and T_platform_gt is not None:
            self.outputs_gt = self.draw_meshes(
                T_rod=T_rod_gt,
                T_platform=T_platform_gt,
                show_backbone=self.gt_settings.get("show_backbone", False),
                show_hsa=self.gt_settings.get("show_hsa", True),
                show_platform=self.gt_settings.get("show_platform", True),
                show_orientation_arrows=self.gt_settings.get(
                    "show_orientation_arrows", True
                ),
                num_orientation_arrows_per_rod=self.gt_settings.get(
                    "num_orientation_arrows_per_rod", 8
                ),
                opacity=self.gt_settings.get("opacity", 1.0),
                orientation_arrows_opacity=self.gt_settings.get(
                    "orientation_arrows_opacity", None
                ),
                ambient=self.gt_settings.get("ambient", 1.0),
                diffuse=self.gt_settings.get("diffuse", 0.5),
                specular=self.gt_settings.get("specular", 0.0),
            )
        if T_rod_hat is not None and T_platform_hat is not None:
            self.outputs_hat = self.draw_meshes(
                T_rod=T_rod_hat,
                T_platform=T_platform_hat,
                show_backbone=self.hat_settings.get("show_backbone", False),
                show_hsa=self.hat_settings.get("show_hsa", True),
                show_platform=self.gt_settings.get("show_platform", True),
                show_orientation_arrows=self.hat_settings.get(
                    "show_orientation_arrows", True
                ),
                num_orientation_arrows_per_rod=self.hat_settings.get(
                    "num_orientation_arrows_per_rod", 8
                ),
                opacity=self.hat_settings.get("opacity", 0.5),
                orientation_arrows_opacity=self.hat_settings.get(
                    "orientation_arrows_opacity", None
                ),
                ambient=self.hat_settings.get("ambient", 1.0),
                diffuse=self.hat_settings.get("diffuse", 0.5),
                specular=self.hat_settings.get("specular", 0.0),
            )

        # add light
        light = pv.Light(
            position=(0, 5.0 * self.L0, 5 * self.L0),
            focal_point=(0.0, 0.0, 0.5 * self.L0),
            show_actor=False,
            positional=True,
            cone_angle=60,
            exponent=20,
            intensity=3,
        )
        self.pl.add_light(light)

        # add coordinate axis at origin of base frame
        # self.pl.add_axes_at_origin()
        marker_args = dict(
            cone_radius=0.6,
            shaft_length=0.7,
            tip_length=0.3,
            ambient=0.5,
            label_size=(0.25, 0.1),
        )
        _ = self.pl.add_axes(line_width=10, color="black", **marker_args)

        # add floor
        # floor = pl.add_floor(face='-z', opacity=0.5, lighting=True, pad=10.0)
        floor = pv.Plane(
            i_size=self.floor_size[0],
            j_size=self.floor_size[1],
            i_resolution=10,
            j_resolution=10,
        )
        self.pl.add_mesh(
            floor,
            ambient=0.0,
            diffuse=0.5,
            specular=0.8,
            color="white",
            opacity=1.0,
        )

        # display settings
        if self.enable_shadows:
            self.pl.enable_shadows()  # add shadows
        self.pl.set_background("white")
        self.pl.camera_position = "xz"
        self.pl.camera.elevation = (
            10.0  # slightly tilt upwards to look from above onto the robot
        )
        self.pl.camera.azimuth = (
            -90.0
        )  # rotate into (-y, -z) plane with x-axis coming out of the screen

    def update_scene(
        self,
        T_rod_gt: np.ndarray = None,
        T_platform_gt: np.ndarray = None,
        T_rod_hat: np.ndarray = None,
        T_platform_hat: np.ndarray = None,
    ):
        if T_rod_gt is not None and T_platform_gt is not None:
            self.update_meshes(
                T_rod=T_rod_gt,
                T_platform=T_platform_gt,
                outputs=self.outputs_gt,
            )
        if T_rod_hat is not None and T_platform_hat is not None:
            self.update_meshes(
                T_rod=T_rod_hat,
                T_platform=T_platform_hat,
                outputs=self.outputs_hat,
            )

    def draw_meshes(
        self,
        T_rod: np.ndarray,
        T_platform: np.ndarray,
        show_backbone: bool = False,
        show_hsa: bool = True,
        show_platform: bool = True,
        show_orientation_arrows: bool = True,
        num_orientation_arrows_per_rod: int = 8,
        opacity: float = 1.0,
        ambient: float = 1.0,
        diffuse: float = 0.5,
        specular: float = 0.0,
        orientation_arrows_opacity: float = None,
    ) -> Dict:
        outputs = {}

        # define colors
        # segment_colors = ["yellow", "orange", "red"]
        # segment_colors = sns.color_palette("rocket", n_colors=len(self.robot_params["segments"]))
        segment_colors = sns.dark_palette(
            "blueviolet", n_colors=max(len(self.robot_params["segments"]) + 1, 10)
        )[1:]

        outputs["segments"] = []
        for i in range(1, T_rod.shape[0] + 1):
            segment_meshes = {"rods": []}
            segment_params = self.robot_params["segments"][i - 1]
            for j in range(0, T_rod.shape[1]):
                rod_meshes = {}
                rod_params = segment_params["rods"][j]

                if show_backbone:
                    rod_meshes["backbone_mesh"] = self._generate_rod_mesh(
                        T_rod[i - 1, j], outside_radius=self.backbone_radius
                    )
                    rod_meshes["backbone_kwargs"] = dict(
                        color=segment_colors[i - 1],
                        opacity=opacity,
                        smooth_shading=True,
                        ambient=ambient,
                        diffuse=diffuse,
                        specular=specular,
                    )
                    rod_meshes["backbone_actor"] = self.pl.add_mesh(
                        rod_meshes["backbone_mesh"],
                        **rod_meshes["backbone_kwargs"],
                    )

                if show_hsa:
                    rod_meshes["hsa_mesh"] = self._generate_rod_mesh(
                        T_rod[i - 1, j],
                        outside_radius=rod_params["outside_radius"],
                        inside_radius=rod_params["outside_radius"]
                        - rod_params["wall_thickness"],
                    )
                    rod_meshes["hsa_kwargs"] = dict(
                        color=segment_colors[i - 1],
                        opacity=opacity,
                        smooth_shading=True,
                        ambient=ambient,
                        diffuse=diffuse,
                        specular=specular,
                    )
                    rod_meshes["hsa_actor"] = self.pl.add_mesh(
                        rod_meshes["hsa_mesh"], **rod_meshes["hsa_kwargs"]
                    )

                if show_orientation_arrows:
                    rod_meshes["rod_orientations"] = []
                    rod_meshes["orientation_indices"] = np.linspace(
                        start=1,
                        stop=T_rod.shape[-1] - 2,
                        num=min(num_orientation_arrows_per_rod, T_rod.shape[-1] - 2),
                        endpoint=True,
                        dtype=int,
                    )
                    for k in rod_meshes["orientation_indices"]:
                        po_dict = {}
                        (
                            po_dict["arrow_meshes"],
                            po_dict["arrow_mesh_kwargs"],
                            po_dict["arrow_actor_kwargs"],
                            po_dict["arrow_actors"],
                        ) = self._draw_orientation_arrows(
                            T_rod[i - 1, j, :, :, k],
                            arrow_selector=[True, True, False],
                            opacity=opacity,
                            ambient=ambient,
                            diffuse=diffuse,
                            specular=specular,
                            tip_length=0.2,
                            tip_radius=0.06,
                            shaft_radius=0.035,
                            scale=0.0185,
                        )
                        rod_meshes["rod_orientations"].append(po_dict)

                segment_meshes["rods"].append(rod_meshes)

            # add platform
            if show_platform:
                segment_meshes["platform_mesh"] = pv.Cylinder(
                    center=T_platform[i - 1, :3, 3],
                    direction=T_platform[i - 1, :3, 2],
                    radius=segment_params["platform"]["radius"],
                    height=segment_params["platform"]["thickness"],
                    resolution=self.platform_resolution,
                    capping=True,
                )
                segment_meshes["platform_kwargs"] = dict(
                    color=segment_colors[i - 1],
                    opacity=opacity,
                    smooth_shading=True,
                    ambient=ambient,
                    diffuse=diffuse,
                    specular=specular,
                )
                segment_meshes["platform_actor"] = self.pl.add_mesh(
                    segment_meshes["platform_mesh"], **segment_meshes["platform_kwargs"]
                )
                if show_orientation_arrows:
                    po_dict = {}
                    T_pot = np.identity(4)
                    T_pot[2, 3] = 0.6 * segment_params["platform"]["thickness"]
                    T_po = T_platform[i - 1] @ T_pot
                    (
                        po_dict["arrow_meshes"],
                        po_dict["arrow_mesh_kwargs"],
                        po_dict["arrow_actor_kwargs"],
                        po_dict["arrow_actors"],
                    ) = self._draw_orientation_arrows(
                        T_po,
                        opacity=opacity,
                        ambient=ambient,
                        diffuse=diffuse,
                        specular=specular,
                        scale=0.0125,
                    )
                    segment_meshes["platform_orientation"] = po_dict

            outputs["segments"].append(segment_meshes)

        return outputs

    def update_meshes(
        self,
        T_rod: np.ndarray,
        T_platform: np.ndarray,
        outputs: Dict,
    ):
        for i in range(1, T_rod.shape[0] + 1):
            segment_meshes = outputs["segments"][i - 1]
            segment_params = self.robot_params["segments"][i - 1]
            for j in range(0, T_rod.shape[1]):
                rod_meshes = segment_meshes["rods"][j]
                rod_params = segment_params["rods"][j]

                if "backbone_mesh" in rod_meshes:
                    self.pl.remove_actor(rod_meshes["backbone_actor"])
                    rod_meshes["backbone_mesh"] = self._generate_rod_mesh(
                        T_rod[i - 1, j],
                        outside_radius=self.backbone_radius,
                        grid=rod_meshes["backbone_mesh"],
                    )
                    rod_meshes["backbone_actor"] = self.pl.add_mesh(
                        rod_meshes["backbone_mesh"], **rod_meshes["backbone_kwargs"]
                    )

                if "hsa_mesh" in rod_meshes:
                    self.pl.remove_actor(rod_meshes["hsa_actor"])
                    rod_meshes["hsa_mesh"] = self._generate_rod_mesh(
                        T_rod[i - 1, j],
                        outside_radius=rod_params["outside_radius"],
                        inside_radius=rod_params["outside_radius"]
                        - rod_params["wall_thickness"],
                        grid=rod_meshes["hsa_mesh"],
                    )
                    rod_meshes["hsa_actor"] = self.pl.add_mesh(
                        rod_meshes["hsa_mesh"], **rod_meshes["hsa_kwargs"]
                    )

                if "rod_orientations" in rod_meshes:
                    for k, po_dict in zip(
                        rod_meshes["orientation_indices"],
                        rod_meshes["rod_orientations"],
                    ):
                        (
                            po_dict["arrow_meshes"],
                            po_dict["arrow_mesh_kwargs"],
                            po_dict["arrow_actor_kwargs"],
                            po_dict["arrow_actors"],
                        ) = self._update_orientation_arrows(
                            T_rod[i - 1, j, :, :, k],
                            arrow_mesh_kwargs=po_dict["arrow_mesh_kwargs"],
                            arrow_actor_kwargs=po_dict["arrow_actor_kwargs"],
                            arrow_actors=po_dict["arrow_actors"],
                            arrow_selector=[True, True, False],
                        )

            if "platform_mesh" in segment_meshes:
                self.pl.remove_actor(segment_meshes["platform_actor"])
                segment_meshes["platform_mesh"] = pv.Cylinder(
                    center=T_platform[i - 1, :3, 3],
                    direction=T_platform[i - 1, :3, 2],
                    radius=segment_params["platform"]["radius"],
                    height=segment_params["platform"]["thickness"],
                    resolution=self.platform_resolution,
                    capping=True,
                )
                segment_meshes["platform_actor"] = self.pl.add_mesh(
                    segment_meshes["platform_mesh"], **segment_meshes["platform_kwargs"]
                )

            if "platform_orientation" in segment_meshes:
                T_pot = np.identity(4)
                T_pot[2, 3] = 0.6 * segment_params["platform"]["thickness"]
                T_po = T_platform[i - 1] @ T_pot
                po_dict = segment_meshes["platform_orientation"]
                (
                    po_dict["arrow_meshes"],
                    po_dict["arrow_mesh_kwargs"],
                    po_dict["arrow_actor_kwargs"],
                    po_dict["arrow_actors"],
                ) = self._update_orientation_arrows(
                    T_po,
                    arrow_mesh_kwargs=po_dict["arrow_mesh_kwargs"],
                    arrow_actor_kwargs=po_dict["arrow_actor_kwargs"],
                    arrow_actors=po_dict["arrow_actors"],
                )

    def _generate_rod_mesh(
        self,
        T_s: np.ndarray,
        outside_radius: float,
        inside_radius: float = 0.0,
        r_resolution: int = 10,
        phi_resolution: int = 50,
        grid: pv.StructuredGrid = None,
    ) -> pv.StructuredGrid:
        mesh_points = self.rod_mesh_points_fun(
            jnp.array(T_s),
            outside_radius,
            inside_radius,
            r_resolution,
            phi_resolution,
        )

        # reshape T_r from
        #   (4, 4, r_resolution * phi_resolution, z_resolution)
        #   to
        #   (4, 4, r_resolution * phi_resolution * z_resolution)
        mesh_points = mesh_points.reshape((4, 4, -1), order="F")
        grid_points = np.array(mesh_points[:3, 3, :].T)

        if grid is None:
            grid = pv.StructuredGrid()
            grid.points = grid_points
            grid.dimensions = [r_resolution, phi_resolution, T_s.shape[-1]]
        else:
            grid.points = grid_points

        return grid

    def _draw_orientation_arrows(
        self,
        T: np.ndarray,
        arrow_mesh_kwargs: List[Dict] = None,
        arrow_actor_kwargs: List[Dict] = None,
        arrow_selector: List[bool] = None,
        opacity: float = 1.0,
        ambient: float = 1.0,
        diffuse: float = 0.5,
        specular: float = 0.0,
        **kwargs,
    ) -> Tuple[List, List[Dict], List[Dict], List]:
        if arrow_selector is None:
            num_arrows = 3
            arrow_selector = [True, True, True]
        else:
            num_arrows = 0
            for val in arrow_selector:
                if val:
                    num_arrows += 1

        if arrow_mesh_kwargs is None:
            arrow_mesh_kwargs = [kwargs for _ in range(num_arrows)]
        if arrow_actor_kwargs is None:
            common_kwargs = dict(
                opacity=opacity,
                smooth_shading=True,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
            )
            arrow_actor_kwargs = [
                common_kwargs | dict(color="red"),
                common_kwargs | dict(color="green"),
                common_kwargs | dict(color="blue"),
            ]

        arrow_meshes = []
        arrow_actors = []
        arrow_idx = 0
        if arrow_selector[0]:
            arrow_meshes.append(
                pv.Arrow(
                    start=T[:3, 3], direction=T[:3, 0], **arrow_mesh_kwargs[arrow_idx]
                )
            )
            arrow_actors.append(
                self.pl.add_mesh(
                    arrow_meshes[arrow_idx], **arrow_actor_kwargs[arrow_idx]
                ),
            )
            arrow_idx += 1
        if arrow_selector[1]:
            arrow_meshes.append(
                pv.Arrow(
                    start=T[:3, 3], direction=T[:3, 1], **arrow_mesh_kwargs[arrow_idx]
                )
            )
            arrow_actors.append(
                self.pl.add_mesh(
                    arrow_meshes[arrow_idx], **arrow_actor_kwargs[arrow_idx]
                ),
            )
            arrow_idx += 1
        if arrow_selector[2]:
            arrow_meshes.append(
                pv.Arrow(
                    start=T[:3, 3], direction=T[:3, 2], **arrow_mesh_kwargs[arrow_idx]
                )
            )
            arrow_actors.append(
                self.pl.add_mesh(
                    arrow_meshes[arrow_idx], **arrow_actor_kwargs[arrow_idx]
                ),
            )
            arrow_idx += 1

        return arrow_meshes, arrow_mesh_kwargs, arrow_actor_kwargs, arrow_actors

    def _update_orientation_arrows(
        self,
        T: np.ndarray,
        arrow_mesh_kwargs: List[Dict],
        arrow_actor_kwargs: List[Dict],
        arrow_actors: List,
        arrow_selector=None,
    ):
        for arrow_actor in arrow_actors:
            self.pl.remove_actor(arrow_actor)

        return self._draw_orientation_arrows(
            T,
            arrow_mesh_kwargs=arrow_mesh_kwargs,
            arrow_actor_kwargs=arrow_actor_kwargs,
            arrow_selector=arrow_selector,
        )

    def close(self):
        self.pl.close()
