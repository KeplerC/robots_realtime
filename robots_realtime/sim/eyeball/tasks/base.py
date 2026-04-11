"""Task ABC and shared constants."""

from abc import ABC, abstractmethod

import mujoco
import numpy as np

try:
    import torch
except ImportError:
    torch = None


class Task(ABC):
    """Defines a manipulation task: scene construction, randomization, success metric."""

    z_offset: float = 0.0

    @abstractmethod
    def configure_scene(self, spec: mujoco.MjSpec):
        """Add task objects (meshes, bodies, joints) to the spec BEFORE compilation.

        Called once during EvalEnv.__init__. Use MeshAsset to add objects
        with convex decomposition.
        """

    @abstractmethod
    def setup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        """Cache body/joint/site IDs AFTER compilation. Called once."""

    @abstractmethod
    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        """Generate n deterministic eval configs. Same (n, base_seed) → same list.

        Config dicts must be JSON-serializable for logging and reproducibility.
        """

    @abstractmethod
    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        """Apply one config to the environment (set freejoint qpos, etc.)."""

    @abstractmethod
    def check_success(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> bool:
        """Check if the task is complete."""

    @property
    def stages(self) -> tuple[str, ...]:
        """Ordered sub-goal names. Last stage = full success.

        Override alongside check_stages() to define granular progress metrics.
        The eval loop tracks "ever achieved" per stage across the episode.
        """
        return ("success",)

    def check_stages(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, bool]:
        """Check all task stages. Returns {stage_name: bool}.

        Default wraps check_success() as a single stage. Override to define
        sub-goals — do all physics queries once here, not per-stage.
        """
        return {"success": self.check_success(mjm, mjd)}

    def post_warmup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        """Called after warmup physics steps, before the first observation.

        Override to re-place objects that may have shifted during warmup
        (e.g. items on a tray that topple during the 500-step gripper settle).
        Default is a no-op.
        """

    def get_gaze_targets(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, np.ndarray]:
        """Return {name: 3D_position} for all scene objects to track gaze against.

        Called per-step during eval. Positions should be in world frame.
        Default returns empty dict (no gaze tracking). Override in task subclasses.
        """
        return {}

    def get_spawn_polygons(self) -> dict[str, tuple[tuple[float, float], ...]] | None:
        """Return spawn polygon data for visualization. Override in subclasses."""
        return None

    def get_spawn_z(self) -> float:
        """Return z height for polygon visualization. Override in subclasses."""
        return 0.03

    # Polygon colors: semi-transparent RGBA for MuJoCo visual geoms
    _POLYGON_COLORS = [
        [0.95, 0.85, 0.2, 0.15],   # yellow
        [0.2, 0.8, 0.2, 0.15],     # green
        [0.2, 0.5, 0.9, 0.15],     # blue
        [0.9, 0.3, 0.3, 0.15],     # red
    ]

    def configure_polygon_geoms(self, spec: mujoco.MjSpec):
        """Add semi-transparent box geoms to visualize spawn polygons.

        Call after configure_scene() but before compile(). Adds non-colliding
        visual-only geoms so polygons are visible in all renderers (VR, Viser, eval).
        """
        polygons = self.get_spawn_polygons()
        if polygons is None:
            return
        spawn_z = self.get_spawn_z()
        thickness = 0.001  # thin slab

        for i, (name, verts) in enumerate(polygons.items()):
            arr = np.array(verts)
            xy_min = arr.min(axis=0)
            xy_max = arr.max(axis=0)
            center = [(xy_min[0] + xy_max[0]) / 2, (xy_min[1] + xy_max[1]) / 2, spawn_z]
            half_size = [(xy_max[0] - xy_min[0]) / 2, (xy_max[1] - xy_min[1]) / 2, thickness]
            rgba = self._POLYGON_COLORS[i % len(self._POLYGON_COLORS)]

            spec.worldbody.add_geom(
                name=f"_polygon_vis_{name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=center,
                size=half_size,
                rgba=rgba,
                contype=0,
                conaffinity=0,
                group=2,
            )

    @abstractmethod
    def get_clip_embedding(self, device: torch.device) -> torch.Tensor:
        """Return (512,) CLIP embedding for the task prompt."""

    @property
    @abstractmethod
    def prompt(self) -> str:
        """Text prompt for the task (e.g., 'a yellow tape roll')."""
