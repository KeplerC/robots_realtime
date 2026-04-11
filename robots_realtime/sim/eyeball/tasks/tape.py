"""TapeHandoverTask and TapeHandoverOODTask."""

from pathlib import Path

import mujoco
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ..decompose_mesh import MeshAsset
from ..poisson_utils import PoissonPool, generate_poisson_configs
from ..polygon_utils import (
    rpy_jitter_quat,
    validate_spawn_config,
    resolve_spawn_collision,
)
from .base import Task


class TapeHandoverTask(Task):
    """Pick up yellow tape from left side, hand to right gripper (or vice versa).

    Adds yellow_tape and grey_tape objects to the scene via MeshAsset +
    MjSpec. CLIP embedding is computed lazily and cached.

    Spawn regions are defined as 2D polygons on the table surface.  The
    drawn polygon is the *boundary* for the entire object — sampling
    automatically insets by each object's bounding radius so the full
    object stays inside.
    """

    ASSETS_DIR = Path(__file__).parent.parent / "assets"
    TAPE_CYLINDER_RADIUS = 0.04  # 40mm — between inner hole (~38mm) and outer wall (~49mm)
    SETTLE_VEL_THRESHOLD = 0.5  # max qvel magnitude for tape to count as settled
    # Fingertip offset in left_lf_down body frame (from outermost sphere_collision geoms)
    LEFT_LF_TIP_OFFSET = np.array([0.0, -0.0004, 0.078])

    OBJECTS = {
        "yellow_tape": {
            "asset_dir": "yellow_tape",
            "rgba": [0.95, 0.85, 0.2, 1],
            "mass": 0.025,
        },
        "grey_tape": {
            "asset_dir": "grey_tape",
            "rgba": [0.55, 0.55, 0.55, 1],
            "mass": 0.08,
        },
    }

    # Default spawn regions matching the original hardcoded rectangles.
    SPAWN_POLYGONS_XY: dict[str, tuple[tuple[float, float], ...]] = {
        "yellow_tape": ((0.1, 0.005), (0.65, 0.005), (0.65, 0.35), (0.25, 0.35),(0.25, 0.225),(0.1, 0.225)),
        "grey_tape": ((0.1, -0.005), (0.65, -0.005), (0.65, -0.35), (0.25, -0.35),(0.25, -0.225),(0.1, -0.225)),
    }
    SPAWN_Z: float = 0.03
    RPY_JITTER_DEG: dict[str, tuple[float, float, float]] = {
        "yellow_tape": (0.0, 0.0, 0.0),
        "grey_tape": (0.0, 0.0, 0.0),
    }
    HARDCODED_POSITIONS: dict[str, list[list[float]]] = {
        "yellow_tape": [
            [0.2375, 0.0875], [0.2375, 0.175],
            [0.375, 0.0875], [0.375, 0.175],
            [0.375, 0.2625], [0.5125, 0.0875],
            [0.5125, 0.175], [0.5125, 0.2625]
        ],
        "grey_tape": [
            [0.2375, -0.0875], [0.2375, -0.175],
            [0.375, -0.0875], [0.375, -0.175],
            [0.375, -0.2625], [0.5125, -0.0875],
            [0.5125, -0.175], [0.5125, -0.2625]
        ],
    }

    _VALID_MODES = {"grid", "poisson"}

    def __init__(
        self,
        placement_mode: str = "grid",
    ):
        if placement_mode not in self._VALID_MODES:
            raise ValueError(
                f"placement_mode must be one of {self._VALID_MODES}, got {placement_mode!r}"
            )
        self._placement_mode = placement_mode
        self._clip_embedding: torch.Tensor | None = None
        self.spawn_polygons_xy = self.SPAWN_POLYGONS_XY
        self.spawn_z = self.SPAWN_Z
        self.rpy_jitter_deg = self.RPY_JITTER_DEG
        self.object_radii_xy = {
            "yellow_tape": 0.044,
            "grey_tape": 0.06,
        }
        validate_spawn_config(
            self.OBJECTS, self.spawn_polygons_xy, self.rpy_jitter_deg,
        )

    @property
    def prompt(self) -> str:
        return "a yellow tape roll"

    def configure_scene(self, spec: mujoco.MjSpec):
        """Add tape objects with convex decomposition collision meshes."""
        for obj_name, cfg in self.OBJECTS.items():
            mesh_asset = MeshAsset.load(self.ASSETS_DIR / cfg["asset_dir"])

            # Add collision mesh assets
            mesh_asset.add_meshes_to_spec(spec, obj_name)

            # Add body with freejoint
            body = spec.worldbody.add_body(name=obj_name, pos=[0.25, 0, 0.03])
            body.add_freejoint(name=f"{obj_name}_jnt")

            # Add visual geom(s) — textured if available, flat-color fallback
            if mesh_asset.has_textures:
                mesh_asset.add_visual_assets_to_spec(spec, obj_name)
                mesh_asset.add_visual_geoms(body, obj_name, mass=cfg["mass"])
            else:
                spec.add_mesh(name=obj_name, file=str(mesh_asset.visual_mesh_path))
                body.add_geom(
                    name=f"{obj_name}_visual",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=obj_name,
                    rgba=cfg["rgba"],
                    mass=cfg["mass"],
                    contype=0,
                    conaffinity=0,
                    group=0,
                )

            col_mesh_kwargs = dict(
                friction=[1.0, 0.05, 0.01],
                solref=[0.002, 1],
                solimp=[0.99, 0.999, 0.001, 0.5, 2],
            )
            mesh_asset.add_collision_geoms(body, obj_name, **col_mesh_kwargs)

    def setup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        """Cache body/joint/site IDs for fast access after compilation."""
        self.yellow_body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "yellow_tape")
        self.grey_body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "grey_tape")
        yellow_jnt_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "yellow_tape_jnt")
        self.yellow_jnt_adr = mjm.jnt_qposadr[yellow_jnt_id]
        self.yellow_dof_adr = mjm.jnt_dofadr[yellow_jnt_id]
        self.grey_jnt_adr = mjm.jnt_qposadr[
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "grey_tape_jnt")
        ]
        self.left_lf_body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "left_lf_down")
        # Explicit mapping for spawn loop (no brittle string manipulation)
        self._spawn_objects = [
            ("yellow_tape", self.yellow_body_id, self.yellow_jnt_adr),
            ("grey_tape", self.grey_body_id, self.grey_jnt_adr),
        ]

        # Collect all geom IDs belonging to yellow_tape and grey_tape bodies
        self.yellow_geom_ids = set()
        self.grey_geom_ids = set()
        for gid in range(mjm.ngeom):
            if mjm.geom_bodyid[gid] == self.yellow_body_id:
                self.yellow_geom_ids.add(gid)
            elif mjm.geom_bodyid[gid] == self.grey_body_id:
                self.grey_geom_ids.add(gid)

        # Collect finger geom IDs split by arm (for handover detection)
        def _finger_geoms(body_names):
            body_ids = set()
            for name in body_names:
                bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    body_ids.add(bid)
            return {gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] in body_ids}

        self.left_finger_geom_ids = _finger_geoms(
            ["left_lf_rot", "left_lf_down", "left_rf_rot", "left_rf_down"]
        )
        # Right gripper split into individual fingers (for handover: both must touch)
        self.right_lf_geom_ids = _finger_geoms(["right_lf_rot", "right_lf_down"])
        self.right_rf_geom_ids = _finger_geoms(["right_rf_rot", "right_rf_down"])
        self.right_finger_geom_ids = self.right_lf_geom_ids | self.right_rf_geom_ids
        self.finger_geom_ids = self.left_finger_geom_ids | self.right_finger_geom_ids

    @property
    def num_grid_configs(self) -> int:
        """Number of unique grid configs (cross-product of HARDCODED_POSITIONS)."""
        import itertools
        return len(list(itertools.product(
            *(self.HARDCODED_POSITIONS[k] for k in self.HARDCODED_POSITIONS)
        )))

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        """Generate n deterministic eval configs for tape placement.

        Dispatches to grid or Poisson mode based on ``self._placement_mode``.
        All configs use the canonical nested schema::

            {
                "seed": int,
                "objects": {
                    "<name>": {"pos": [x, y, z], "quat": [w, x, y, z]},
                    ...
                },
            }
        """
        if self._placement_mode == "grid":
            return self._generate_grid_configs(n, base_seed)
        else:
            return self._generate_poisson_configs(n, base_seed)

    def _generate_grid_configs(self, n: int, base_seed: int) -> list[dict]:
        """Grid configs: cross-product of HARDCODED_POSITIONS, cycling."""
        import itertools

        obj_names = list(self.HARDCODED_POSITIONS.keys())
        position_lists = [self.HARDCODED_POSITIONS[k] for k in obj_names]
        grid = list(itertools.product(*position_lists))

        configs = []
        for i in range(n):
            combo = grid[i % len(grid)]
            rng = np.random.default_rng(base_seed + i)
            objects = {}
            for name, xy in zip(obj_names, combo):
                quat = rpy_jitter_quat(self.rpy_jitter_deg[name], rng)
                objects[name] = {
                    "pos": [float(xy[0]), float(xy[1]), self.spawn_z],
                    "quat": [float(q) for q in quat],
                }
            configs.append({"seed": base_seed + i, "objects": objects})
        return configs

    def _generate_poisson_configs(self, n: int, base_seed: int) -> list[dict]:
        """Poisson-disc configs via ``generate_poisson_configs``."""
        raw = generate_poisson_configs(
            self.spawn_polygons_xy,
            self.object_radii_xy,
            n=n,
            seed=base_seed,
        )
        configs = []
        for i, raw_cfg in enumerate(raw):
            rng = np.random.default_rng(base_seed + i)
            objects = {}
            for name, xy in raw_cfg.items():
                quat = rpy_jitter_quat(self.rpy_jitter_deg[name], rng)
                objects[name] = {
                    "pos": [float(xy[0]), float(xy[1]), self.spawn_z],
                    "quat": [float(q) for q in quat],
                }
            configs.append({"seed": base_seed + i, "objects": objects})
        return configs

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        """Apply a config dict to set tape positions via freejoint qpos."""
        for obj_name, _, jnt_adr in self._spawn_objects:
            obj_cfg = config["objects"][obj_name]
            mjd.qpos[jnt_adr : jnt_adr + 3] = obj_cfg["pos"]
            mjd.qpos[jnt_adr + 3 : jnt_adr + 7] = obj_cfg["quat"]

    @property
    def stages(self) -> tuple[str, ...]:
        return ("pick_yellow", "handover", "place_yellow")

    def check_stages(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, bool]:
        """Check all task stages with a single contact scan."""
        yellow_contacts_grey = False
        yellow_contacts_left = False
        yellow_contacts_right_lf = False
        yellow_contacts_right_rf = False

        for i in range(mjd.ncon):
            c = mjd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            g1_y = g1 in self.yellow_geom_ids
            g2_y = g2 in self.yellow_geom_ids
            if not (g1_y or g2_y):
                continue
            other = g2 if g1_y else g1
            if other in self.grey_geom_ids:
                yellow_contacts_grey = True
            if other in self.left_finger_geom_ids:
                yellow_contacts_left = True
            if other in self.right_lf_geom_ids:
                yellow_contacts_right_lf = True
            if other in self.right_rf_geom_ids:
                yellow_contacts_right_rf = True

        yellow_z = mjd.xpos[self.yellow_body_id][2]
        yellow_xmat = mjd.xmat[self.yellow_body_id].reshape(3, 3)
        z_alignment = abs(yellow_xmat[:, 2][2])

        # Check that left fingertip is outside the tape cylinder (not poking through the hole)
        tape_pos = mjd.xpos[self.yellow_body_id]
        lf_xpos = mjd.xpos[self.left_lf_body_id]
        lf_xmat = mjd.xmat[self.left_lf_body_id].reshape(3, 3)
        fingertip_pos = lf_xpos + lf_xmat @ self.LEFT_LF_TIP_OFFSET
        local = yellow_xmat.T @ (fingertip_pos - tape_pos)
        radial = np.sqrt(local[0] ** 2 + local[1] ** 2)
        left_fingertip_outside = radial > self.TAPE_CYLINDER_RADIUS

        yellow_contacts_right = yellow_contacts_right_lf or yellow_contacts_right_rf

        # Check that the tape has settled (low velocity — not mid-drop)
        yellow_qvel = mjd.qvel[self.yellow_dof_adr : self.yellow_dof_adr + 6]
        yellow_settled = np.linalg.norm(yellow_qvel) < self.SETTLE_VEL_THRESHOLD

        return {
            "pick_yellow": yellow_z > 0.06 + self.z_offset and yellow_contacts_left and left_fingertip_outside,
            "handover": (
                yellow_contacts_right_lf
                and yellow_contacts_right_rf
                and not yellow_contacts_left
            ),
            "place_yellow": (
                yellow_contacts_grey
                and not (yellow_contacts_left or yellow_contacts_right)
                and z_alignment > 0.9
                and yellow_settled
            ),
        }

    def check_success(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> bool:
        return self.check_stages(mjm, mjd)["place_yellow"]

    def get_spawn_polygons(self) -> dict[str, tuple[tuple[float, float], ...]]:
        return self.spawn_polygons_xy

    def get_spawn_z(self) -> float:
        return self.spawn_z

    def get_gaze_targets(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, np.ndarray]:
        return {
            "yellow_tape": mjd.xpos[self.yellow_body_id].copy(),
            "grey_tape": mjd.xpos[self.grey_body_id].copy(),
        }

    def get_clip_embedding(self, device: torch.device) -> torch.Tensor:
        """Compute and cache CLIP embedding for the task prompt."""
        if self._clip_embedding is not None:
            return self._clip_embedding.to(device)

        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device, precision="fp16"
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        with torch.no_grad():
            tokens = tokenizer([self.prompt]).to(device)
            embedding = model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        self._clip_embedding = embedding[0].float()
        return self._clip_embedding


class TapeHandoverOODTask(TapeHandoverTask):
    """OOD evaluation: objects spawn outside their training distribution.

    Two OOD conditions (can be combined):
    - ``yellow_right``: yellow tape spawns in negative-Y (right of training region)
    - ``grey_left``: grey tape spawns in positive-Y (left of training region)
    - ``both``: both objects shifted simultaneously

    OOD spawn regions are thin rectangles (~15cm in Y) starting just past
    the training boundary and extending into OOD territory.  Rejection
    sampling prevents collisions when both objects are near the Y≈0
    midline.
    """

    COLLISION_BUFFER: float = 0.01  # 1cm extra clearance beyond bounding radii

    # Thin OOD rectangles: 15cm in Y, full X extent [0.1, 0.65].
    # Yellow training low-Y boundary = 0.005 → OOD extends into negative Y.
    # Grey training high-Y boundary = -0.005 → OOD extends into positive Y.
    _OOD_POLYGONS: dict[str, tuple[tuple[float, float], ...]] = {
        "yellow_tape": ((0.1, -0.15), (0.65, -0.15), (0.65, 0.0), (0.1, 0.0)),
        "grey_tape": ((0.1, 0.0), (0.65, 0.0), (0.65, 0.15), (0.1, 0.15)),
    }

    _VALID_OOD_MODES = {"yellow_right", "grey_left", "both"}

    def __init__(self, ood_mode: str = "both"):
        if ood_mode not in self._VALID_OOD_MODES:
            raise ValueError(
                f"ood_mode must be one of {self._VALID_OOD_MODES}, got {ood_mode!r}"
            )
        self._ood_mode = ood_mode
        super().__init__(placement_mode="poisson")
        # Override spawn polygons based on OOD mode — non-OOD object keeps
        # its training polygon so we isolate the distribution shift.
        if ood_mode == "yellow_right":
            self.spawn_polygons_xy = {
                "yellow_tape": self._OOD_POLYGONS["yellow_tape"],
                "grey_tape": self.SPAWN_POLYGONS_XY["grey_tape"],
            }
        elif ood_mode == "grey_left":
            self.spawn_polygons_xy = {
                "yellow_tape": self.SPAWN_POLYGONS_XY["yellow_tape"],
                "grey_tape": self._OOD_POLYGONS["grey_tape"],
            }
        else:  # both
            self.spawn_polygons_xy = dict(self._OOD_POLYGONS)

    def _generate_poisson_configs(self, n: int, base_seed: int) -> list[dict]:
        """Poisson configs with rejection sampling to prevent collisions.

        Samples grey first, then rejection-samples yellow until the pair
        satisfies the minimum distance (sum of radii + buffer).
        """
        min_dist = (
            self.object_radii_xy["yellow_tape"]
            + self.object_radii_xy["grey_tape"]
            + self.COLLISION_BUFFER
        )

        rng = np.random.default_rng(base_seed)
        grey_pool = PoissonPool(
            polygon=np.asarray(self.spawn_polygons_xy["grey_tape"], dtype=np.float64),
            n_target=max(n, 50),
            margin=self.object_radii_xy["grey_tape"],
        )
        yellow_pool = PoissonPool(
            polygon=np.asarray(self.spawn_polygons_xy["yellow_tape"], dtype=np.float64),
            n_target=max(n, 50),
            margin=self.object_radii_xy["yellow_tape"],
        )

        configs: list[dict] = []
        for i in range(n):
            ep_rng = np.random.default_rng(base_seed + i)
            grey_xy = grey_pool.pop(rng)
            # Rejection-sample yellow until it clears grey
            for _attempt in range(200):
                yellow_xy = yellow_pool.pop(rng)
                if np.linalg.norm(np.asarray(yellow_xy) - np.asarray(grey_xy)) >= min_dist:
                    break

            objects = {}
            for name, xy in [("grey_tape", grey_xy), ("yellow_tape", yellow_xy)]:
                quat = rpy_jitter_quat(self.rpy_jitter_deg[name], ep_rng)
                objects[name] = {
                    "pos": [float(xy[0]), float(xy[1]), self.spawn_z],
                    "quat": [float(q) for q in quat],
                }
            configs.append({"seed": base_seed + i, "objects": objects})

        return configs
