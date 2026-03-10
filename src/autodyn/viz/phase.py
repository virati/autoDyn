import numpy as np
from fury import actor, window


def render_phase(raster: np.ndarray, title: str = "Phase Portrait", colors=None):
    """FURY-based 3D rendering of a phase-space trajectory.

    Parameters
    ----------
    raster : np.ndarray
        Trajectory array of shape (T, 3).
    title : str
        Window title.
    colors : array-like, optional
        Per-point colors. Defaults to a blue-to-red colormap along time.
    """
    if raster.ndim != 2 or raster.shape[1] != 3:
        raise ValueError(f"raster must be (T, 3), got {raster.shape}")

    if colors is None:
        t = np.linspace(0, 1, len(raster))
        colors = np.column_stack([t, np.zeros_like(t), 1 - t])  # blue -> red

    # FURY line actor expects a list of lines
    line_actor = actor.line([raster], colors=[colors])

    scene = window.Scene()
    scene.add(line_actor)
    scene.background((0.05, 0.05, 0.05))

    window.show(scene, title=title, size=(900, 700))
