from collections import defaultdict
from elastica.callback_functions import ExportCallBack, CallBackBaseClass
from elastica._rotations import _inv_rotate
import numpy as np
import pathlib
import sys


class HsaBaseDiagnosticCallback(ExportCallBack):
    """
    Call back function for continuum snake
    """

    def __init__(
        self,
        step_skip: int,
        sim_total_steps: int = None,
        callback_data: dict = None,  # if specified, we write all diagnostic data to memory and keep it there
        export_path: str = None,
        export_method: str = "npz",
        file_save_interval: int = 1e8,
    ):
        if export_path is not None:
            self.save_to_disk = True
            path = pathlib.Path(export_path)
            ExportCallBack.__init__(
                self,
                step_skip=step_skip,
                directory=str(path.parent),
                filename=str(path.name),
                method=export_method,
                file_save_interval=file_save_interval,
            )
        else:
            self.save_to_disk = False
            CallBackBaseClass.__init__(self)

        self.step_skip, self.sim_total_steps = step_skip, sim_total_steps
        self.callback_data = callback_data

        self.save_to_memory = True
        if self.callback_data is None:
            self.save_to_memory = False
            self.callback_data = defaultdict(list)

    def make_callback(self, system: object, time, current_step: int):
        """

        Parameters
        ----------
        system :
            Each part of the system (i.e. rod, rigid body, etc)
        time :
            simulation time unit
        current_step : int
            simulation step
        """
        if current_step % self.step_skip == 0:
            self.append_data(system, time, current_step)

            if self.save_to_disk:
                for key in self.callback_data.keys():
                    self.buffer[key].append(self.callback_data[key][-1])
                    self.buffer_size += sys.getsizeof(self.callback_data[key][-1])

        if self.save_to_disk:
            if (
                self.buffer_size > HsaBaseDiagnosticCallback.FILE_SIZE_CUTOFF
                or (current_step + 1) % self.file_save_interval == 0
                or (
                    self.sim_total_steps is not None
                    and (current_step + 1) == self.sim_total_steps
                )
            ):
                self._dump()

        if self.save_to_memory is False and len(self.callback_data.keys()) > 0:
            self.callback_data = defaultdict(list)

        return

    def append_data(self, system: object, time, current_step: int):
        self.callback_data["time"].append(time)
        self.callback_data["step"].append(current_step)
        self.callback_data["position"].append(system.position_collection.copy())
        self.callback_data["directors"].append(system.director_collection.copy())
        self.callback_data["velocity"].append(system.velocity_collection.copy())
        self.callback_data["external_forces"].append(system.external_forces.copy())
        self.callback_data["external_torques"].append(system.external_torques.copy())


class HsaRodDiagnosticCallback(HsaBaseDiagnosticCallback):
    def __init__(
        self,
        *args,
        position_center_of_mass: bool = False,
        velocity_center_of_mass: bool = False,
        rest_lengths: bool = True,
        sigma: bool = True,
        kappa: bool = True,
        internal_forces: bool = True,
        internal_torques: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.position_center_of_mass = position_center_of_mass
        self.velocity_center_of_mass = velocity_center_of_mass
        self.rest_lengths = rest_lengths
        self.sigma = sigma
        self.kappa = kappa
        self.internal_forces = internal_forces
        self.internal_torques = internal_torques

    def append_data(self, system: object, time, current_step: int):
        super().append_data(system, time, current_step)

        if self.position_center_of_mass:
            self.callback_data["position_center_of_mass"].append(
                system.compute_position_center_of_mass()
            )
        if self.velocity_center_of_mass:
            self.callback_data["velocity_center_of_mass"].append(
                system.compute_velocity_center_of_mass()
            )
        self.callback_data["lengths"].append(system.lengths.copy())
        if self.rest_lengths:
            self.callback_data["rest_lengths"].append(system.rest_lengths.copy())
        if self.sigma:
            self.callback_data["sigma"].append(system.sigma.copy())
        if self.kappa:
            self.callback_data["kappa"].append(system.kappa.copy())
        if self.internal_forces:
            self.callback_data["internal_forces"].append(system.internal_forces.copy())
        if self.internal_torques:
            self.callback_data["internal_torques"].append(
                system.internal_torques.copy()
            )

        # twist angle of base
        # we perform a cumulative sum from the distal end towards the proximal end
        twist_angle = -np.cumsum(_inv_rotate(system.director_collection)[2, :][::-1])[
            ::-1
        ]  # radians
        self.callback_data["twist_angle"].append(twist_angle)


class HsaRigidBodyDiagnosticCallback(HsaBaseDiagnosticCallback):
    def append_data(self, system: object, time, current_step: int):
        super().append_data(system, time, current_step)
