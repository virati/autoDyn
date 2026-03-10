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

def _time_colors(n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.column_stack([t, np.zeros_like(t), 1 - t])  # blue -> red


def _smooth(raster: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-smooth each axis independently."""
    out = np.empty_like(raster)
    for i in range(raster.shape[1]):
        out[:, i] = gaussian_filter1d(raster[:, i], sigma=sigma)
    return out


def _glow_actors(raster: np.ndarray):
    """Return three line actors that together produce a neon glow effect.

    Outer → wide + transparent  (halo)
    Mid   → medium              (bloom)
    Core  → thin + fully opaque (bright spine)
    """
    s = _smooth(raster)
    n = len(s)
    c = _time_colors(n)

    layers = [
        # (colors_scale, line_width, opacity)
        (0.35, 12, 0.10),
        (0.65, 5,  0.28),
        (1.00, 1.5, 1.0),
    ]
    actors = []
    for scale, width, opacity in layers:
        a = actor.line([s], colors=[c * scale])
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

def render_phase(
    raster: np.ndarray,
    title: str = "Phase Portrait",
    f=None,
    params: dict = None,
    T: float = None,
    dt: float = 0.01,
    chat_callback=None,
):
    """FURY-based 3D rendering of a phase-space trajectory with glow.

    Parameters
    ----------
    raster        : np.ndarray  shape (T_steps, 3)
    title         : window title
    f             : dynamics callable ``f(x, **params) -> np.ndarray``
    params        : current parameter values
    T             : total simulation time for re-simulation
    dt            : integration step
    chat_callback : callable(text: str) -> dict — enables the chat sidebar
    """
    if raster.ndim != 2 or raster.shape[1] != 3:
        raise ValueError(f"raster must be (T, 3), got {raster.shape}")

    interactive = f is not None and params is not None and T is not None

    scene = window.Scene()
    scene.background((0.02, 0.02, 0.06))  # deep navy for contrast

    # Initial glow actors
    glow_refs = _glow_actors(raster)
    for a in glow_refs:
        scene.add(a)

    if not interactive:
        window.show(scene, title=title, size=(900, 700))
        return

    # ------------------------------------------------------------------ state
    current_params = {k: float(eval(str(v))) for k, v in params.items()}
    x0 = raster[0:1].T.copy()

    def rebuild():
        new_raster = _run_sim(f, current_params, T, dt, 3, x0)
        for a in glow_refs:
            scene.rm(a)
        glow_refs.clear()
        for a in _glow_actors(new_raster):
            scene.add(a)
            glow_refs.append(a)

    # ShowManager must exist before adding UI elements
    win_h = 800 if chat_callback is not None else 700
    show_manager = ShowManager(
        scene, title=title, size=(1100, win_h), order_transparent=True
    )

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
        font_size=13,
        color=(0.78, 0.78, 0.85),
        size=(CHAT_W - 20, 130),
    )
    chat_panel.add_element(history_block, (0.01, 0.38))

    input_display = ui.TextBlock2D(
        text="> _",
        font_size=14,
        color=(0.2, 1.0, 0.5),  # green terminal cursor
        size=(CHAT_W - 20, 28),
    )
    chat_panel.add_element(input_display, (0.01, 0.05))

    status_block = ui.TextBlock2D(
        text="", font_size=12, color=(1.0, 0.8, 0.2), size=(400, 24),
    )
    chat_panel.add_element(status_block, (0.01, 0.20))

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
        elif char and char in _printable and char.strip():
            typed[0] += char
            _refresh_input()

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
