import time
import threading
import queue as _queue
from string import printable as _printable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from fury import actor, window, ui
from fury.window import ShowManager

from autodyn.core.integrators.runge_kutta import rk_integrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_colors(n: int, hue_index: int = 0, total_hues: int = 1) -> np.ndarray:
    """Time-gradient colours with a per-trajectory hue.

    Each trajectory gets a maximally-separated hue (golden-ratio spacing
    in HSV) so that even adjacent trajectories look drastically different.
    Brightness ramps from dim → bright along the trajectory to show time.
    """
    t = np.linspace(0.4, 1.0, n)  # value ramp (time → brightness)
    if total_hues <= 1:
        return np.column_stack([t, np.zeros_like(t), 1 - t])  # blue → red

    # Golden-ratio hue spacing gives maximally distinct colours
    golden = (1 + np.sqrt(5)) / 2
    hue = (hue_index * golden) % 1.0

    # HSV → RGB with S=1, V ramped by time
    # Using the sector formula directly to avoid importing colorsys per-point
    h6 = hue * 6.0
    sector = int(h6) % 6
    frac = h6 - int(h6)
    rgb_base = [
        (1.0, frac, 0.0),          # 0: red→yellow
        (1.0 - frac, 1.0, 0.0),    # 1: yellow→green
        (0.0, 1.0, frac),          # 2: green→cyan
        (0.0, 1.0 - frac, 1.0),    # 3: cyan→blue
        (frac, 0.0, 1.0),          # 4: blue→magenta
        (1.0, 0.0, 1.0 - frac),    # 5: magenta→red
    ][sector]

    r = np.full(n, rgb_base[0]) * t
    g = np.full(n, rgb_base[1]) * t
    b = np.full(n, rgb_base[2]) * t
    return np.column_stack([r, g, b])


def _pad_to_3d(raster: np.ndarray) -> np.ndarray:
    """Pad or project raster to exactly 3 columns for 3-D rendering."""
    D = raster.shape[1]
    if D == 3:
        return raster
    if D < 3:
        padding = np.zeros((raster.shape[0], 3 - D))
        return np.column_stack([raster, padding])
    # D > 3: keep first three dimensions
    return raster[:, :3]


def _smooth(raster: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-smooth each axis independently."""
    out = np.empty_like(raster)
    for i in range(raster.shape[1]):
        out[:, i] = gaussian_filter1d(raster[:, i], sigma=sigma)
    return out


def _glow_actors_multi(rasters: list, M: int = 1, uniform_color: bool = False):
    """Return glow line actors for *M* trajectories.

    When *uniform_color* is True, every trajectory uses the original
    blue→red time-gradient (identical to the single-trajectory look).
    Otherwise each trajectory gets a maximally-separated hue.
    """
    layers = [
        # (brightness_scale, line_width, opacity)
        (0.35, 12, 0.10),
        (0.65, 5,  0.28),
        (1.00, 1.5, 1.0),
    ]
    actors = []
    effective_hues = 1 if uniform_color else M
    for traj_idx, raster in enumerate(rasters):
        s = _smooth(raster)
        n = len(s)
        c = _time_colors(n, hue_index=0 if uniform_color else traj_idx,
                         total_hues=effective_hues)
        for scale, width, opacity in layers:
            a = actor.line([s], colors=np.clip(c * scale, 0, 1))
            a.GetProperty().SetLineWidth(width)
            a.GetProperty().SetOpacity(opacity)
            actors.append(a)
    return actors


def _run_sim(f, params: dict, T: float, dt: float, D: int, x0: np.ndarray) -> np.ndarray:
    x_state = x0.copy()
    raster = []
    for _ in np.arange(0, T, dt):
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

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _make_random_x0s(base_x0: np.ndarray, M: int, spread: float = 1.0):
    """Generate M initial conditions: the original plus M-1 random perturbations."""
    x0s = [base_x0.copy()]
    for _ in range(M - 1):
        perturb = base_x0 + np.random.normal(0, spread, base_x0.shape)
        x0s.append(perturb)
    return x0s


def render_phase(
    raster: np.ndarray,
    title: str = "Phase Portrait",
    f=None,
    params: dict = None,
    T: float = None,
    dt: float = 0.01,
    chat_callback=None,
    M: int = 1,
    uniform_color: bool = False,
):
    """FURY-based 3D rendering of a phase-space trajectory with glow.

    Parameters
    ----------
    raster        : np.ndarray  shape (T_steps, D)
    title         : window title
    f             : dynamics callable ``f(x, **params) -> np.ndarray``
    params        : current parameter values
    T             : total simulation time for re-simulation
    dt            : integration step
    chat_callback : callable(text: str) -> dict — enables the chat sidebar
    M             : number of trajectories from different initial conditions.
                    Total rendered points stays ~ constant (T is split across M).
    uniform_color : if True, all trajectories use the same blue→red colour
                    scheme (original single-trajectory look).
    """
    if raster.ndim != 2 or raster.shape[1] < 1:
        raise ValueError(f"raster must be (T, D) with D >= 1, got {raster.shape}")
    M = max(1, int(M))
    raw_D = raster.shape[1]
    raw_x0 = raster[0:1].T.copy()

    # Keep full T per trajectory; coarsen dt so total points ≈ original count
    dt_eff = dt * M
    interactive = f is not None and params is not None and T is not None

    scene = window.Scene()
    scene.background((0.02, 0.02, 0.06))  # deep navy for contrast

    # Build initial rasters: first from the provided raster (subsampled to
    # match the coarser step), remaining M-1 from fresh simulations.
    if M == 1 or not interactive:
        init_rasters = [_pad_to_3d(raster[::M])]
    else:
        x0s = _make_random_x0s(raw_x0, M)
        init_rasters = []
        for x0_i in x0s:
            r = _run_sim(f, {k: float(eval(str(v))) for k, v in params.items()},
                         T, dt_eff, raw_D, x0_i)
            init_rasters.append(_pad_to_3d(r))

    glow_refs = _glow_actors_multi(init_rasters, M, uniform_color=uniform_color)
    for a in glow_refs:
        scene.add(a)

    if not interactive:
        window.show(scene, title=title, size=(900, 700))
        return

    # ------------------------------------------------------------------ state
    current_params = {k: float(eval(str(v))) for k, v in params.items()}
    current_x0s = _make_random_x0s(raw_x0, M)

    def rebuild():
        rasters = []
        for x0_i in current_x0s:
            r = _run_sim(f, current_params, T, dt_eff, raw_D, x0_i)
            rasters.append(_pad_to_3d(r))
        for a in glow_refs:
            scene.rm(a)
        glow_refs.clear()
        for a in _glow_actors_multi(rasters, M, uniform_color=uniform_color):
            scene.add(a)
            glow_refs.append(a)

    # ShowManager must exist before adding UI elements
    win_h = 800 if chat_callback is not None else 700
    show_manager = ShowManager(
        scene, title=title, size=(1100, win_h), order_transparent=True
    )

    # ------------------------------------------------- debounced zoom control
    # VTK processes every scroll tick synchronously through OnMouseWheelForward/
    # Backward on the interactor style, causing zoom to keep going long after
    # you stop scrolling.  We replace those methods entirely so VTK never runs
    # its built-in zoom, then batch-apply our own after scrolling settles.
    _zoom_delta = [0]
    _zoom_last = [0.0]
    _ZOOM_IDLE_S = 0.10

    iren = show_manager.iren
    style = iren.GetInteractorStyle()

    # Monkey-patch the style so VTK's built-in zoom is completely dead
    style.OnMouseWheelForward = lambda: None
    style.OnMouseWheelBackward = lambda: None

    def _scroll_fwd(_obj, _event):
        _zoom_delta[0] += 1
        _zoom_last[0] = time.monotonic()

    def _scroll_bwd(_obj, _event):
        _zoom_delta[0] -= 1
        _zoom_last[0] = time.monotonic()

    # Priority 1.0 = fires before VTK's default handlers (priority 0)
    iren.AddObserver("MouseWheelForwardEvent", _scroll_fwd, 1.0)
    iren.AddObserver("MouseWheelBackwardEvent", _scroll_bwd, 1.0)

    def _flush_zoom(_obj, _event):
        if _zoom_delta[0] == 0:
            return
        if time.monotonic() - _zoom_last[0] < _ZOOM_IDLE_S:
            return  # still scrolling — wait
        delta = _zoom_delta[0]
        _zoom_delta[0] = 0
        cam = scene.GetActiveCamera()
        factor = 1.1 ** (-delta)
        cam.Dolly(factor)
        if hasattr(scene, "ResetCameraClippingRange"):
            scene.ResetCameraClippingRange()
        show_manager.render()

    show_manager.add_timer_callback(True, 50, _flush_zoom)

    # ---------------------------------------------------- param slider panel
    n_params = len(current_params)
    panel_h = 50 + 65 * n_params
    param_panel = ui.Panel2D(size=(280, panel_h), color=(0.12, 0.12, 0.12), opacity=0.85)
    param_panel.center = (970, win_h - panel_h // 2 - 20)

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
        param_panel.add_element(slider, (0.08, y_frac))

    scene.add(param_panel)

    # ------------------------------------------------------------ chat panel
    if chat_callback is None:
        show_manager.start()
        return

    update_queue = _queue.Queue()
    chat_log = []
    busy = [False]
    typed = [""]  # mutable text buffer

    CHAT_W, CHAT_H = 1080, 200
    chat_panel = ui.Panel2D(size=(CHAT_W, CHAT_H), color=(0.08, 0.08, 0.08), opacity=0.92)
    chat_panel.center = (CHAT_W // 2, CHAT_H // 2)

    history_block = ui.TextBlock2D(
        text="Ready. Just start typing and press Enter.",
        font_size=18,
        color=(0.78, 0.78, 0.85),
        size=(CHAT_W, 130),
    )
    chat_panel.add_element(history_block, (0.0, 0.35))

    input_display = ui.TextBlock2D(
        text="> _",
        font_size=18,
        color=(0.2, 1.0, 0.5),
        size=(CHAT_W, 34),
    )
    chat_panel.add_element(input_display, (0.0, 0.02))

    status_block = ui.TextBlock2D(
        text="", font_size=15, color=(1.0, 0.8, 0.2), size=(CHAT_W, 28),
    )
    chat_panel.add_element(status_block, (0.0, 0.18))

    def _refresh_input():
        input_display.message = "> " + typed[0] + "_"

    def _update_history():
        history_block.message = "\n".join(chat_log[-5:])

    def _send(msg: str):
        if busy[0] or not msg:
            return
        chat_log.append(f"You: {msg}")
        _update_history()
        status_block.message = "  Thinking..."
        busy[0] = True

        def _worker():
            try:
                new_params = chat_callback(msg)
                coerced = {k: float(eval(str(v))) for k, v in new_params.items()}
                summary = ", ".join(f"{k}={v:.3f}" for k, v in coerced.items())
                update_queue.put(("params", coerced, f"Agent: {summary}"))
            except Exception as e:
                update_queue.put(("error", None, f"Error: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    # Raw VTK key observer — no TextBox2D focus/activation required
    def _on_key(obj, _event):
        key = obj.GetKeySym()
        char = obj.GetKeyCode()
        if key == "Return":
            msg = typed[0].strip()
            typed[0] = ""
            _refresh_input()
            _send(msg)
        elif key in ("BackSpace", "Delete"):
            typed[0] = typed[0][:-1]
            _refresh_input()
        elif char and char in _printable:
            typed[0] += char
            _refresh_input()
        else:
            return
        show_manager.render()

    show_manager.add_iren_callback(_on_key, event="KeyPressEvent")

    def _process_queue(obj, event):
        while not update_queue.empty():
            kind, data, msg = update_queue.get()
            if kind == "params":
                current_params.update(data)
                rebuild()
            chat_log.append(msg)
            _update_history()
            status_block.message = ""
            busy[0] = False

    show_manager.add_timer_callback(True, 200, _process_queue)

    scene.add(chat_panel)
    show_manager.start()
