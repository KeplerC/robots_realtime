"""Composable OOD wrapper system for MuJoCo tasks.

Stack freely:
    TablePoseOOD(TapeHandoverTask(), dz=0.10)
    CameraOOD(VisualOOD(TapeHandoverTask()), translate_max=0.05)
    DistractorOOD(TablePoseOOD(PickUpTigerTask(), dz=0.10))
"""

from pathlib import Path

import mujoco
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from matplotlib.path import Path as MplPath

from .base import Task
from ..decompose_mesh import MeshAsset

_ASSETS_DIR = Path(__file__).parent.parent / "assets"


# ---------------------------------------------------------------------------
# TaskWrapper base — pure delegation with template method hooks
# ---------------------------------------------------------------------------

class TaskWrapper(Task):
    """Forwards all Task ABC methods to self._task.

    Subclasses override only what they change. configure_scene and setup use
    template method hooks (_extra variants) so stacking works without
    subclasses having to manually call self._task.method().
    """

    def __init__(self, task: Task):
        self._task = task

    @property
    def z_offset(self) -> float:
        return self._task.z_offset

    @z_offset.setter
    def z_offset(self, value: float):
        self._task.z_offset = value

    # --- template method hooks ---

    def configure_scene(self, spec: mujoco.MjSpec):
        self._task.configure_scene(spec)
        self.configure_scene_extra(spec)

    def configure_scene_extra(self, spec: mujoco.MjSpec):
        # Keyframes from the base XML have a fixed qpos size that becomes
        # invalid once task objects add freejoints.  Delete them so compile()
        # doesn't throw; mj_resetDataKeyframe is a no-op when nkey == 0.
        # MuJoCo 3.5+ doesn't enforce keyframe size and lacks k.delete().
        for k in list(spec.keys):
            if hasattr(k, 'delete'):
                k.delete()

    def setup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        self._task.setup(mjm, mjd)
        self.setup_extra(mjm, mjd)

    def setup_extra(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        pass

    # --- straight delegation ---

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        return self._task.generate_eval_configs(n, base_seed)

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        return self._task.apply_eval_config(mjm, mjd, config)

    def post_warmup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        return self._task.post_warmup(mjm, mjd, config)

    def check_success(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> bool:
        return self._task.check_success(mjm, mjd)

    def check_stages(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, bool]:
        return self._task.check_stages(mjm, mjd)

    @property
    def stages(self) -> tuple[str, ...]:
        return self._task.stages

    @property
    def prompt(self) -> str:
        return self._task.prompt

    def get_clip_embedding(self, device: torch.device) -> torch.Tensor:
        return self._task.get_clip_embedding(device)

    def get_gaze_targets(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, np.ndarray]:
        return self._task.get_gaze_targets(mjm, mjd)

    def get_spawn_polygons(self) -> dict[str, tuple[tuple[float, float], ...]] | None:
        return self._task.get_spawn_polygons()

    def get_spawn_z(self) -> float:
        return self._task.get_spawn_z()


# ---------------------------------------------------------------------------
# TablePoseOOD
# ---------------------------------------------------------------------------

class TablePoseOOD(TaskWrapper):
    """Randomize table position and orientation per episode.

    Each parameter accepts a (min, max) range sampled uniformly each episode.
    Pass a float as a shorthand for a fixed offset (i.e. (v, v)).

        TablePoseOOD(task, dz=(0, 0.15))          # raise table 0–15 cm
        TablePoseOOD(task, dyaw=(-0.3, 0.3))      # rotate ±0.3 rad
        TablePoseOOD(task, dx=(-0.05, 0.05), dy=(-0.05, 0.05))
    """

    _TABLE_CENTER_XY = np.array([0.47, -0.01])

    def __init__(
        self,
        task: Task,
        dx: float | tuple[float, float] = 0.0,
        dy: float | tuple[float, float] = 0.0,
        dz: float | tuple[float, float] = 0.0,
        dyaw: float | tuple[float, float] = 0.0,
    ):
        super().__init__(task)
        self.dx_range  = (dx,  dx)  if isinstance(dx,  (int, float)) else tuple(dx)
        self.dy_range  = (dy,  dy)  if isinstance(dy,  (int, float)) else tuple(dy)
        self.dz_range  = (dz,  dz)  if isinstance(dz,  (int, float)) else tuple(dz)
        self.dyaw_range = (dyaw, dyaw) if isinstance(dyaw, (int, float)) else tuple(dyaw)
        self._table_id: int = -1
        self._orig_pos: np.ndarray | None = None
        self._orig_quat: np.ndarray | None = None

    def setup_extra(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        self._table_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "table")
        if self._table_id >= 0:
            self._orig_pos  = mjm.body_pos[self._table_id].copy()
            self._orig_quat = mjm.body_quat[self._table_id].copy()

    def _apply_table_offset(self, mjm: mujoco.MjModel, dx: float, dy: float, dz: float, dyaw: float):
        if self._table_id < 0 or self._orig_pos is None:
            return
        mjm.body_pos[self._table_id]  = self._orig_pos + [dx, dy, dz]
        if dyaw != 0.0:
            c, s = np.cos(dyaw / 2), np.sin(dyaw / 2)
            ow, ox, oy, oz = self._orig_quat
            qw, qx, qy, qz = c, 0.0, 0.0, s
            mjm.body_quat[self._table_id] = [
                ow*qw - ox*qx - oy*qy - oz*qz,
                ow*qx + ox*qw + oy*qz - oz*qy,
                ow*qy - ox*qz + oy*qw + oz*qx,
                ow*qz + ox*qy - oy*qx + oz*qw,
            ]
        else:
            mjm.body_quat[self._table_id] = self._orig_quat.copy()

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        configs = self._task.generate_eval_configs(n, base_seed)
        rng = np.random.default_rng(base_seed)
        for cfg in configs:
            dx   = float(rng.uniform(*self.dx_range))
            dy   = float(rng.uniform(*self.dy_range))
            dz   = float(rng.uniform(*self.dz_range))
            dyaw = float(rng.uniform(*self.dyaw_range))
            cfg["table_pose"] = {"dx": dx, "dy": dy, "dz": dz, "dyaw": dyaw}
            c, s = np.cos(dyaw), np.sin(dyaw)
            for obj_cfg in cfg.get("objects", {}).values():
                pos = obj_cfg["pos"]
                if dyaw != 0.0:
                    xy = np.array([pos[0], pos[1]]) - self._TABLE_CENTER_XY
                    pos[0] = float(c * xy[0] - s * xy[1] + self._TABLE_CENTER_XY[0] + dx)
                    pos[1] = float(s * xy[0] + c * xy[1] + self._TABLE_CENTER_XY[1] + dy)
                else:
                    pos[0] += dx
                    pos[1] += dy
                pos[2] += dz
        return configs

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        self._task.apply_eval_config(mjm, mjd, config)
        pose = config.get("table_pose", {})
        dx, dy, dz, dyaw = pose.get("dx", 0.0), pose.get("dy", 0.0), pose.get("dz", 0.0), pose.get("dyaw", 0.0)
        self._apply_table_offset(mjm, dx, dy, dz, dyaw)
        self._task.z_offset = dz


# ---------------------------------------------------------------------------
# CameraOOD
# ---------------------------------------------------------------------------

class CameraOOD(TaskWrapper):
    """Randomly perturb the eye camera rig per episode.

    Stereo mode (elp_stereo body present):
        Perturbs mjm.body_pos / body_quat for the elp_stereo body.  Both
        elp_left and elp_right cameras follow automatically (they are parented
        to this body), and mjd.xpos[elp_stereo] updates after forward kinematics
        so _compute_stereo_eye_to_base_se3 sees the shifted position — keeping
        proprio and actions consistent with the new camera center.

    Mono fallback (no elp_stereo):
        Perturbs mjm.cam_pos / cam_quat for eye_camera directly.

    Offsets are sampled uniformly in [-max, max] and stored in eval configs,
    so perturbations are deterministic and reproducible.
    """

    _STEREO_BODY = "elp_stereo"
    _MONO_CAM    = "eye_camera"

    def __init__(self, task: Task, translate_max: float = 0.05, rotate_max: float = 0.1):
        super().__init__(task)
        self.translate_max = translate_max
        self.rotate_max    = rotate_max
        self._stereo_body_id: int = -1
        self._mono_cam_id:    int = -1
        self._orig_body_pos:  np.ndarray | None = None
        self._orig_body_quat: np.ndarray | None = None
        self._orig_cam_pos:   np.ndarray | None = None
        self._orig_cam_quat:  np.ndarray | None = None

    def setup_extra(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        self._stereo_body_id = mujoco.mj_name2id(
            mjm, mujoco.mjtObj.mjOBJ_BODY, self._STEREO_BODY)
        if self._stereo_body_id >= 0:
            self._orig_body_pos  = mjm.body_pos[self._stereo_body_id].copy()
            self._orig_body_quat = mjm.body_quat[self._stereo_body_id].copy()
        else:
            self._mono_cam_id = mujoco.mj_name2id(
                mjm, mujoco.mjtObj.mjOBJ_CAMERA, self._MONO_CAM)
            if self._mono_cam_id >= 0:
                self._orig_cam_pos  = mjm.cam_pos[self._mono_cam_id].copy()
                self._orig_cam_quat = mjm.cam_quat[self._mono_cam_id].copy()

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        configs = self._task.generate_eval_configs(n, base_seed)
        rng = np.random.default_rng(base_seed)
        for cfg in configs:
            t    = rng.uniform(-self.translate_max, self.translate_max, size=3)
            axis = rng.uniform(-1.0, 1.0, size=3)
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                axis /= norm
            angle = float(rng.uniform(-self.rotate_max, self.rotate_max))
            half  = angle / 2.0
            cfg["cam_translate"] = [float(v) for v in t]
            cfg["cam_rotate"]    = [float(np.cos(half)), *(float(v) for v in axis * np.sin(half))]
        return configs

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        ow, ox, oy, oz = q1
        qw, qx, qy, qz = q2
        q = np.array([
            ow*qw - ox*qx - oy*qy - oz*qz,
            ow*qx + ox*qw + oy*qz - oz*qy,
            ow*qy - ox*qz + oy*qw + oz*qx,
            ow*qz + ox*qy - oy*qx + oz*qw,
        ])
        n = np.linalg.norm(q)
        return q / n if n > 1e-8 else q

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        self._task.apply_eval_config(mjm, mjd, config)
        dt = np.array(config.get("cam_translate", [0.0, 0.0, 0.0]))
        dq = np.array(config.get("cam_rotate",    [1.0, 0.0, 0.0, 0.0]))
        if self._stereo_body_id >= 0 and self._orig_body_pos is not None:
            # Stereo: shift the whole rig body — cameras and proprio SE3 both follow
            mjm.body_pos[self._stereo_body_id]  = self._orig_body_pos + dt
            mjm.body_quat[self._stereo_body_id] = self._quat_mul(self._orig_body_quat, dq)
        elif self._mono_cam_id >= 0 and self._orig_cam_pos is not None:
            # Mono fallback: shift eye_camera directly
            mjm.cam_pos[self._mono_cam_id]  = self._orig_cam_pos + dt
            mjm.cam_quat[self._mono_cam_id] = self._quat_mul(self._orig_cam_quat, dq)


# ---------------------------------------------------------------------------
# VisualOOD
# ---------------------------------------------------------------------------

class VisualOOD(TaskWrapper):
    """Randomize table surface color per episode.

    Palette uses realistic table/surface colors (wood tones, greys, muted blues/greens).
    For lighting randomization, stack with LightingOOD.
    """

    # Realistic table surface colors: wood tones, greys, muted solids
    _TABLE_PALETTE: list[tuple[float, float, float, float]] = [
        (0.85, 0.80, 0.72, 1.0),  # light wood / birch
        (0.65, 0.50, 0.35, 1.0),  # medium wood / oak
        (0.40, 0.28, 0.18, 1.0),  # dark wood / walnut
        (0.90, 0.90, 0.90, 1.0),  # white / light grey
        (0.55, 0.55, 0.55, 1.0),  # mid grey
        (0.25, 0.25, 0.25, 1.0),  # dark grey / charcoal
        (0.45, 0.55, 0.60, 1.0),  # muted slate blue
        (0.40, 0.55, 0.45, 1.0),  # muted sage green
    ]

    def __init__(self, task: Task):
        super().__init__(task)
        self.n_colors = len(self._TABLE_PALETTE)
        self._color_palette = np.array(self._TABLE_PALETTE, dtype=np.float32)
        self._table_mat_id: int = -1

    def setup_extra(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        self._table_mat_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "table_mat")

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        configs = self._task.generate_eval_configs(n, base_seed)
        rng = np.random.default_rng(base_seed)
        for cfg in configs:
            cfg["table_color_idx"] = int(rng.integers(0, self.n_colors))
        return configs

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        self._task.apply_eval_config(mjm, mjd, config)
        if self._table_mat_id >= 0:
            mjm.mat_rgba[self._table_mat_id] = self._color_palette[config.get("table_color_idx", 0)]


# ---------------------------------------------------------------------------
# LightingOOD
# ---------------------------------------------------------------------------

class LightingOOD(TaskWrapper):
    """Randomize scene lighting per episode.

    Three independent axes of variation, each sampled per episode:

    1. brightness   — global diffuse/ambient scale (0.2=dim, 2.0=bright)
    2. ambient_ratio — ambient relative to diffuse; low = harsh shadows (0.0–1.0)
    3. dominant     — one random light is boosted to simulate a harsh point source;
                      its diffuse is multiplied by dominant_boost and shadow casting
                      is enabled on it while all other lights are dimmed by
                      dominant_dim_others.

    All original light state is restored before each episode's config is applied,
    so effects don't accumulate.
    """

    def __init__(
        self,
        task: Task,
        brightness_range: tuple[float, float] = (0.4, 1.8),
        ambient_ratio_range: tuple[float, float] = (0.0, 1.0),
        dominant_prob: float = 0.5,
        dominant_boost: float = 4.0,
        dominant_dim_others: float = 0.15,
    ):
        super().__init__(task)
        self.brightness_range     = brightness_range
        self.ambient_ratio_range  = ambient_ratio_range
        self.dominant_prob        = dominant_prob
        self.dominant_boost       = dominant_boost
        self.dominant_dim_others  = dominant_dim_others

        self._orig_diffuse:     np.ndarray | None = None
        self._orig_ambient:     np.ndarray | None = None
        self._orig_specular:    np.ndarray | None = None
        self._orig_castshadow:  np.ndarray | None = None
        self._orig_active:      np.ndarray | None = None

    def setup_extra(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        self._orig_diffuse    = mjm.light_diffuse.copy()
        self._orig_ambient    = mjm.light_ambient.copy()
        self._orig_specular   = mjm.light_specular.copy()
        self._orig_castshadow = mjm.light_castshadow.copy()
        self._orig_active     = mjm.light_active.copy()

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        configs = self._task.generate_eval_configs(n, base_seed)
        rng = np.random.default_rng(base_seed)
        nlight = len(self._orig_diffuse) if self._orig_diffuse is not None else 6
        for cfg in configs:
            cfg["lighting"] = {
                "brightness":    float(rng.uniform(*self.brightness_range)),
                "ambient_ratio": float(rng.uniform(*self.ambient_ratio_range)),
                "dominant":      int(rng.integers(0, nlight)) if rng.random() < self.dominant_prob else -1,
            }
        return configs

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        self._task.apply_eval_config(mjm, mjd, config)
        if self._orig_diffuse is None:
            return

        lc = config.get("lighting", {})
        brightness    = lc.get("brightness", 1.0)
        ambient_ratio = lc.get("ambient_ratio", 1.0)
        dominant_idx  = lc.get("dominant", -1)

        # Restore originals first
        mjm.light_diffuse[:]    = self._orig_diffuse
        mjm.light_ambient[:]    = self._orig_ambient
        mjm.light_specular[:]   = self._orig_specular
        mjm.light_castshadow[:] = self._orig_castshadow
        mjm.light_active[:]     = self._orig_active

        # Global brightness scale
        mjm.light_diffuse[:]  = np.clip(self._orig_diffuse * brightness, 0.0, 1.0)
        # ambient_ratio: 1.0 = original ratio, 0.0 = no ambient (harshest shadows)
        mjm.light_ambient[:]  = np.clip(self._orig_ambient * brightness * ambient_ratio, 0.0, 1.0)

        # Dominant point light: boost one, dim others, enable shadow
        if dominant_idx >= 0:
            for i in range(mjm.nlight):
                if i == dominant_idx:
                    mjm.light_diffuse[i]  = np.clip(self._orig_diffuse[i] * brightness * self.dominant_boost, 0.0, 1.0)
                    mjm.light_specular[i] = np.clip(self._orig_specular[i] * self.dominant_boost, 0.0, 1.0)
                    mjm.light_castshadow[i] = 1
                else:
                    mjm.light_diffuse[i] = np.clip(mjm.light_diffuse[i] * self.dominant_dim_others, 0.0, 1.0)
                    mjm.light_ambient[i] = np.clip(mjm.light_ambient[i] * self.dominant_dim_others, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ObjectOOD
# ---------------------------------------------------------------------------

class ObjectOOD(TaskWrapper):
    """Randomize task object size and color per episode.

    Scale is applied by modifying mesh vertex positions around each mesh's
    centroid — affects both visual and collision geometry.

    Color is applied by tinting the object's material RGBA. With
    flat_color=True the material texture is also disabled so you get a
    solid color instead of a tinted texture.

    object_body_names: bodies to affect. Defaults to keys of get_gaze_targets().
    scale_range: (min, max) uniform scale factor, e.g. (0.8, 1.3).
    color_prob: probability per episode of randomizing object color.
    flat_color: if True, disable texture on affected materials (solid color).
    """

    # A few recognizable solid colors for object recoloring
    _OBJECT_PALETTE: list[tuple[float, float, float, float]] = [
        (0.85, 0.15, 0.15, 1.0),  # red
        (0.10, 0.45, 0.85, 1.0),  # blue
        (0.15, 0.65, 0.20, 1.0),  # green
        (0.20, 0.20, 0.20, 1.0),  # black
        (0.90, 0.90, 0.90, 1.0),  # white
        (0.55, 0.55, 0.55, 1.0),  # grey
    ]

    def __init__(
        self,
        task: Task,
        object_body_names: list[str] | None = None,
        scale_range: tuple[float, float] = (0.8, 1.3),
        color_prob: float = 0.5,
        flat_color: bool = False,
    ):
        super().__init__(task)
        self._object_body_names = object_body_names
        self.scale_range  = scale_range
        self.color_prob   = color_prob
        self.flat_color   = flat_color

        self._color_palette = np.array(self._OBJECT_PALETTE, dtype=np.float32)

        # Populated in setup_extra
        # Each entry: (mesh_id, vert_adr, vert_num, centroid, orig_verts)
        self._mesh_data: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
        self._mat_ids: list[int] = []
        self._orig_mat_rgba: dict[int, np.ndarray] = {}
        self._orig_mat_texid: dict[int, int] = {}

    def setup_extra(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        # Resolve body names
        if self._object_body_names is not None:
            body_names = self._object_body_names
        else:
            try:
                body_names = list(self._task.get_gaze_targets(mjm, mjd).keys())
            except Exception:
                body_names = []

        body_ids = {
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in body_names
        } - {-1}

        geom_ids = [i for i in range(mjm.ngeom) if mjm.geom_bodyid[i] in body_ids]

        # Collect unique mesh IDs across all geoms (visual + collision)
        seen_meshes: set[int] = set()
        self._mesh_data = []
        for gid in geom_ids:
            if mjm.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = mjm.geom_dataid[gid]
            if mid < 0 or mid in seen_meshes:
                continue
            seen_meshes.add(mid)
            adr = mjm.mesh_vertadr[mid]
            num = mjm.mesh_vertnum[mid]
            orig = mjm.mesh_vert[adr : adr + num].copy()  # shape (num, 3)
            centroid = orig.mean(axis=0)
            self._mesh_data.append((mid, adr, num, centroid, orig))

        # Collect unique material IDs from visual geoms
        mat_ids: set[int] = set()
        for gid in geom_ids:
            mid = mjm.geom_matid[gid]
            if mid >= 0:
                mat_ids.add(mid)
        self._mat_ids = list(mat_ids)
        self._orig_mat_rgba  = {mid: mjm.mat_rgba[mid].copy() for mid in self._mat_ids}
        self._orig_mat_texid = {mid: int(mjm.mat_texid[mid, 0]) for mid in self._mat_ids}

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        configs = self._task.generate_eval_configs(n, base_seed)
        rng = np.random.default_rng(base_seed)
        for cfg in configs:
            scale     = float(rng.uniform(*self.scale_range))
            color_idx = int(rng.integers(0, len(self._color_palette))) if rng.random() < self.color_prob else -1
            cfg["object_ood"] = {"scale": scale, "color_idx": color_idx}
        return configs

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        self._task.apply_eval_config(mjm, mjd, config)
        ood       = config.get("object_ood", {})
        scale     = ood.get("scale", 1.0)
        color_idx = ood.get("color_idx", -1)

        # Scale mesh vertices around each mesh's centroid
        for mid, adr, num, centroid, orig in self._mesh_data:
            scaled = centroid + (orig - centroid) * scale  # (num, 3)
            mjm.mesh_vert[adr : adr + num] = scaled

        # Restore then optionally recolor materials
        for mid in self._mat_ids:
            mjm.mat_rgba[mid]    = self._orig_mat_rgba[mid]
            mjm.mat_texid[mid, 0] = self._orig_mat_texid[mid]

        if color_idx >= 0:
            color = self._color_palette[color_idx]
            for mid in self._mat_ids:
                mjm.mat_rgba[mid] = color
                if self.flat_color:
                    mjm.mat_texid[mid, 0] = -1


# ---------------------------------------------------------------------------
# DistractorOOD
# ---------------------------------------------------------------------------

_DISTRACTOR_POLY_X = (0.15, 0.55)
_DISTRACTOR_POLY_Y = (-0.30, 0.30)


class DistractorOOD(TaskWrapper):
    """Add randomly placed distractor objects to the scene.

    Bodies are injected at configure_scene time (required by MuJoCo) and
    repositioned per episode via freejoint qpos. Placement avoids task spawn
    regions via rejection sampling against get_spawn_polygons().
    """

    def __init__(
        self,
        task: Task,
        n_distractors: int = 2,
        asset_names: list[str] | None = None,
    ):
        super().__init__(task)
        self.n_distractors = n_distractors
        self.asset_names = asset_names if asset_names is not None else ["hammer"]
        self._distractor_jnt_adrs: list[int] = []

    def configure_scene_extra(self, spec: mujoco.MjSpec):
        col_kwargs = dict(
            friction=[1.0, 0.05, 0.01],
            solref=[0.002, 1],
            solimp=[0.99, 0.999, 0.001, 0.5, 2],
        )

        # add_meshes_to_spec and add_visual_assets_to_spec use the asset's own
        # stem names (not a tag prefix), so they must only be called once per
        # unique asset type — multiple bodies can share the registered assets.
        registered_assets: set[str] = set()

        for i in range(self.n_distractors):
            asset_name = self.asset_names[i % len(self.asset_names)]
            body_name = f"_distractor_{i}"

            asset = MeshAsset.load(_ASSETS_DIR / asset_name)

            if asset_name not in registered_assets:
                asset.add_meshes_to_spec(spec, asset_name)
                if asset.has_textures:
                    asset.add_visual_assets_to_spec(spec, asset_name)
                else:
                    spec.add_mesh(
                        name=f"{asset_name}_visual_mesh",
                        file=str(asset.visual_mesh_path),
                    )
                registered_assets.add(asset_name)

            body = spec.worldbody.add_body(name=body_name, pos=[0.0, 0.0, -10.0])
            body.add_freejoint(name=f"{body_name}_jnt")

            if asset.has_textures:
                asset.add_visual_geoms(body, body_name, mass=0.1)
            else:
                body.add_geom(
                    name=f"{body_name}_visual",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=f"{asset_name}_visual_mesh",
                    rgba=[0.6, 0.5, 0.4, 1],
                    mass=0.1,
                    contype=0, conaffinity=0, group=0,
                )
            asset.add_collision_geoms(body, body_name, **col_kwargs)

    def setup_extra(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        self._distractor_jnt_adrs = []
        for i in range(self.n_distractors):
            jnt_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, f"_distractor_{i}_jnt")
            self._distractor_jnt_adrs.append(mjm.jnt_qposadr[jnt_id])

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        configs = self._task.generate_eval_configs(n, base_seed)
        rng = np.random.default_rng(base_seed)

        spawn_polys = self._task.get_spawn_polygons() or {}
        exclusion_paths = [MplPath(np.array(verts)) for verts in spawn_polys.values()]

        x_lo, x_hi = _DISTRACTOR_POLY_X
        y_lo, y_hi = _DISTRACTOR_POLY_Y

        for cfg in configs:
            distractors = []
            for _ in range(self.n_distractors):
                xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
                for _attempt in range(200):
                    if not any(p.contains_point(xy) for p in exclusion_paths):
                        break
                    xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
                yaw = float(rng.uniform(0, 2 * np.pi))
                c, s = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
                distractors.append({"pos": [float(xy[0]), float(xy[1]), 0.03], "quat": [c, 0.0, 0.0, s]})
            cfg["distractors"] = distractors
        return configs

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        self._task.apply_eval_config(mjm, mjd, config)
        for i, d_cfg in enumerate(config.get("distractors", [])):
            if i >= len(self._distractor_jnt_adrs):
                break
            a = self._distractor_jnt_adrs[i]
            mjd.qpos[a : a + 3] = d_cfg["pos"]
            mjd.qpos[a + 3 : a + 7] = d_cfg["quat"]
