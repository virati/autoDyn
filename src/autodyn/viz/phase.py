import numpy as np
from fury import actor, window, ui
from fury.window import ShowManager
from autodyn.core.integrators.runge_kutta import rk_integrator


def _time_colors(n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.column_stack([t, np.zeros_like(t), 1 - t])  # blue -> red


def _run_sim(f, params: dict, T: float, dt: float, D: int, x0: np.ndarray) -> np.ndarray:
    tvect = np.arange(0, T, dt)
    x_state = x0.copy()
    raster = []
    for _ in tvect:
        x_state = rk_integrator(f, x_state, dt=dt, **params)
        raster.append(x_state)
    return np.array(raster).squeeze()


def _slider_range(v: float):
    v = float(v)
    if v == 0:
        return -1.0, 1.0
    lo = min(v * 0.1, v * 5.0)
    hi = max(v * 0.1, v * 5.0)
    return lo, hi


def render_phase(
    raster: np.ndarray,
    title: str = "Phase Portrait",
    f=None,
    params: dict = None,
    T: float = None,
    dt: float = 0.01,
):
    """FURY-based 3D rendering of a phase-space trajectory.

    If `f`, `params`, and `T` are supplied the window shows interactive
    sliders that re-simulate on every change.

    Parameters
    ----------
    raster : np.ndarray  shape (T_steps, 3)
    title  : window title
    f      : dynamics callable  ``f(x, **params) -> np.ndarray``
    params : dict of current parameter values (strings are coerced to float)
    T      : total simulation time for re-simulation
    dt     : integration step
    """
    if raster.ndim != 2 or raster.shape[1] != 3:
        raise ValueError(f"raster must be (T, 3), got {raster.shape}")

    interactive = f is not None and params is not None and T is not None

    scene = window.Scene()
    scene.background((0.05, 0.05, 0.05))

    line_actor = actor.line([raster], colors=[_time_colors(len(raster))])
    scene.add(line_actor)

    if not interactive:
        window.show(scene, title=title, size=(900, 700))
        return

    # ------------------------------------------------------------------ state
    current_params = {k: float(eval(str(v))) for k, v in params.items()}
    x0 = raster[0:1].T.copy()          # fix initial condition to avoid jumps
    actor_ref = [line_actor]            # mutable container for swap

    # --------------------------------------------------------------- helpers
    def rebuild():
        new_raster = _run_sim(f, current_params, T, dt, 3, x0)
        scene.rm(actor_ref[0])
        new_actor = actor.line([new_raster], colors=[_time_colors(len(new_raster))])
        scene.add(new_actor)
        actor_ref[0] = new_actor

    # Create ShowManager first — UI elements need the interactor to register callbacks
    show_manager = ShowManager(scene, title=title, size=(1000, 700))

    # --------------------------------------------------------------- panel
    n_params = len(current_params)
    panel_h = 50 + 65 * n_params
    panel = ui.Panel2D(size=(280, panel_h), color=(0.12, 0.12, 0.12), opacity=0.85)
    panel.center = (820, panel_h // 2 + 20)

    for idx, (name, val) in enumerate(current_params.items()):
        lo, hi = _slider_range(val)
        slider = ui.LineSlider2D(
            min_value=lo,
            max_value=hi,
            initial_value=val,
            length=220,
            text_template=f"{name}: {{value:.3f}}",
        )

        def _make_cb(param_name):
            def _cb(slider):
                current_params[param_name] = slider.value
                rebuild()
            return _cb

        slider.on_change = _make_cb(name)
        y_frac = 1.0 - (idx + 1) / (n_params + 1)
        panel.add_element(slider, (0.08, y_frac))

    scene.add(panel)
    show_manager.start()
