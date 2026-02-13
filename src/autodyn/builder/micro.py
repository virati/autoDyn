from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from autodyn.core.network import connectivity
import networkx as nx
from autodyn.utils.functions import unity


class microstruct:
    def __init__(self):
        pass

    @property
    def dynamics(self):
        if not self._dynamics:
            self._dynamics = unity

        return self._dynamics

    @dynamics.setter
    def dynamics(self, f: Callable):
        self._dynamics = f
