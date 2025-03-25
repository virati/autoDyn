from autodyn.core.dsys import dsys
import numpy as np


class control(dsys):
    def __init__(self):
        pass

    @property
    def u(self):
        self._u: control
        if self.u_func is None:
            return np.zeros_like(self.x)

        if not self._u:
            self._u = control(self.u_func)
        return self._u

    @u.setter
    def u(self, value: np.ndarray):
        self._u = control()
        self._u.raw = value
