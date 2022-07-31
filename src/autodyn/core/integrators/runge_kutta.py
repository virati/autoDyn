import numpy as np
from typing import Optional

# Integrator
def rk_integrator(
    f_dyn: callable,
    state: np.ndarray,
    u: Optional[np.ndarray] = None,
    dt: float = 0.01,
    **kwargs,
) -> np.ndarray:

    """
    Basic Runge-Kutta integrator with dynamics, state, and control parameters

    Inputs
    ------
        f_dyn : Callable
            The drift dynamics as a function

        state : np.ndarray
            The state vector

        u : Union[int, np.ndarray]
            The control signal

        dt

    Outputs
    -------
        new_state : np.ndarray
            The new state after integrating through the effects of the dynamics

    """
    if u is None:
        u = 0

    k1 = f_dyn(state, u=u, **kwargs) * dt
    k2 = f_dyn(state + 0.5 * k1, u=u, **kwargs) * dt
    k3 = f_dyn(state + 0.5 * k2, u=u, **kwargs) * dt
    k4 = f_dyn(state + k3, u=u, **kwargs) * dt

    new_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return new_state
