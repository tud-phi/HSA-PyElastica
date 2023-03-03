__doc__ = """
Base System
-----------

Basic coordinating for multiple, smaller systems that have an independently integrable
interface (i.e. works with symplectic or explicit routines `timestepper.py`.)
"""
from typing import Iterable, Callable, AnyStr

from elastica.rod.cosserat_rod import CosseratRod
from elastica.rigidbody import RigidBodyBase
from elastica.wrappers import BaseSystemCollection as ElasticaBaseSystemCollection

from hsa_elastica.rod import HsaRod
from hsa_elastica.modules.memory_block import construct_memory_block_structures


class BaseSystemCollection(ElasticaBaseSystemCollection):
    """
    Base System for simulator classes. Every simulation class written by the user
    must be derived from the BaseSystemCollection class; otherwise the simulation will
    proceed.

        Attributes
        ----------
        allowed_sys_types: tuple
            Tuple of allowed type rod-like objects. Here use a base class for objects, i.e. RodBase.
        _systems: list
            List of rod-like objects.

    """

    """
    Developer Note
    -----
    Note
    ----
    We can directly subclass a list for the
    most part, but this is a bad idea, as List is non abstract
    https://stackoverflow.com/q/3945940
    """

    def __init__(self):
        super(BaseSystemCollection, self).__init__()
        # List of system types/bases that are allowed
        self.allowed_sys_types = (CosseratRod, RigidBodyBase, HsaRod)

    def finalize(self):
        """
        This method finalizes the simulator class. When it is called, it is assumed that the user has appended
        all rod-like objects to the simulator as well as all boundary conditions, callbacks, etc.,
        acting on these rod-like objects. After the finalize method called,
        the user cannot add new features to the simulator class.
        """

        # This generates more straight-forward error.
        assert self._finalize_flag is not True, "The finalize cannot be called twice."

        # construct memory block
        self._memory_blocks = construct_memory_block_structures(self._systems)

        # Recurrent call finalize functions for all components.
        for finalize in self._feature_group_finalize:
            finalize()

        # Clear the finalize feature group, just for the safety.
        self._feature_group_finalize.clear()
        self._feature_group_finalize = None

        # Toggle the finalize_flag
        self._finalize_flag = True
