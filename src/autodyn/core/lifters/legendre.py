from typing import Optional
import jax.numpy as np
import numpy as nnp


class lifter:
    def __init__(self):
        pass


def L_lift(x):
    a * x + a * x**3


def lift(
    trajectory: np.ndarray,
    M: int,
    U: Optional[np.ndarray] = None,
):
    if trajectory.ndim != 2:
        raise NotImplemented

    N, T = trajectory.shape
    if U is None:
        U = nnp.random.multivariate_normal(nnp.zeros(M), nnp.eye(M, N), size=(M, N))

    # need an array of callables
    U @ f(trajectory)
