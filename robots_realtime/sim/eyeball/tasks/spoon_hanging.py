"""HangSpoonOnHookTask."""

from pathlib import Path

import mujoco
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ..decompose_mesh import MeshAsset
from ..poisson_utils import PoissonPool
from .base import Task


class HangSpoonOnHookTask(Task):
    """Hang a wooden spoon on a free-standing hook stand.

    The hook stand is placed at a random position/yaw on the table.
    The spoon spawns ~10-20cm in front of the hook (toward the robot),
    with its long axis pointing away from the robot (+X).
    """

    ASSETS_DIR = Path(__file__).parent.parent / "assets"

    # --- Hook (static) ---
    # Post-conversion: Z is up (1.0 pre-scale), X/Y are lateral (~0.5 each)
    # Scale: [0.342, 0.342, 0.3933] — 90% of original [0.38, 0.38, 0.437]
    # Table surface z = -0.0269  (table_top geom z=-0.0344 + half-height 0.0075)
    # spawn_z = table_z + 0.50 * 0.3933 = -0.0269 + 0.1967 = 0.1698
    HOOK_SPAWN_Z = 0.1698
    HOOK_POLYGON = ((0.55, -0.30), (0.75, -0.30), (0.75, 0.30), (0.55, 0.30))

    # --- Spoon (dynamic) ---
    # Spawns in a 8×15cm grid centered 30cm below (toward robot) the hook in X.
    SPOON_SPAWN_Z = 0.08  # drop from 8cm above table
    SPOON_X_OFFSET = -0.30  # offset toward robot from hook
    SPOON_GRID_HALF_X = 0.04  # ±4cm → 8cm in X
    SPOON_GRID_HALF_Y = 0.075  # ±7.5cm → 15cm in Y

    # Base quat: long axis (mesh Y) pointing along +X (away from robot)
    # Rotate 90° around Z: wxyz = [cos(45°), 0, 0, sin(45°)] = [√2/2, 0, 0, √2/2]
    SPOON_BASE_QUAT = [float(np.sqrt(2) / 2), 0.0, 0.0, float(np.sqrt(2) / 2)]

    SETTLE_VEL_THRESHOLD = 0.1  # max qvel magnitude for spoon to count as settled

    @property
    def prompt(self) -> str:
        return "a wooden spoon"

    def configure_scene(self, spec: mujoco.MjSpec):
        col_kwargs = dict(
            friction=[0.3, 0.02, 0.005],
            solref=[0.002, 1],
            solimp=[0.99, 0.999, 0.001, 0.5, 2],
        )
        hook_col_kwargs = dict(
            friction=[0.05, 0.005, 0.001],
            solref=[0.002, 1],
            solimp=[0.99, 0.999, 0.001, 0.5, 2],
        )

        # --- Hook stand (static) ---
        hook_asset = MeshAsset.load(self.ASSETS_DIR / "hanger_hook")
        hook_asset.add_meshes_to_spec(spec, "hook")

        hook_body = spec.worldbody.add_body(
            name="hook",
            pos=[0.30, 0.0, self.HOOK_SPAWN_Z],
            quat=[1, 0, 0, 0],
        )
        if hook_asset.has_textures:
            hook_asset.add_visual_assets_to_spec(spec, "hook")
            hook_asset.add_visual_geoms(hook_body, "hook", mass=1000.0)
        else:
            m = spec.add_mesh(name="hook_visual_mesh", file=str(hook_asset.visual_mesh_path))
            if hook_asset._scale_vec:
                m.scale = hook_asset._scale_vec
            hook_body.add_geom(
                name="hook_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="hook_visual_mesh",
                rgba=[0.6, 0.45, 0.3, 1],
                mass=1000.0,
                contype=0, conaffinity=0, group=0,
            )
        hook_asset.add_collision_geoms(hook_body, "hook", **hook_col_kwargs)

        # --- Wooden spoon (dynamic) ---
        spoon_asset = MeshAsset.load(self.ASSETS_DIR / "wooden_spoon")
        spoon_asset.add_meshes_to_spec(spec, "spoon")

        spoon_body = spec.worldbody.add_body(
            name="spoon",
            pos=[0.20, 0.0, self.SPOON_SPAWN_Z],
            quat=self.SPOON_BASE_QUAT,
        )
        spoon_body.add_freejoint(name="spoon_jnt")
        if spoon_asset.has_textures:
            spoon_asset.add_visual_assets_to_spec(spec, "spoon")
            spoon_asset.add_visual_geoms(spoon_body, "spoon", mass=0.15)
        else:
            m = spec.add_mesh(name="spoon_visual_mesh", file=str(spoon_asset.visual_mesh_path))
            if spoon_asset._scale_vec:
                m.scale = spoon_asset._scale_vec
            spoon_body.add_geom(
                name="spoon_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="spoon_visual_mesh",
                rgba=[0.6, 0.45, 0.3, 1],
                mass=0.15,
                contype=0, conaffinity=0, group=0,
            )
        spoon_asset.add_collision_geoms(spoon_body, "spoon", **col_kwargs)

    def setup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        self.hook_body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "hook")
        self.spoon_body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "spoon")
        spoon_jnt_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "spoon_jnt")
        self.spoon_jnt_adr = mjm.jnt_qposadr[spoon_jnt_id]
        self.spoon_dof_adr = mjm.jnt_dofadr[spoon_jnt_id]
        self._hook_pool: PoissonPool | None = None

        # Collect geom IDs for contact checks
        self.spoon_geom_ids: set[int] = set()
        self.hook_geom_ids: set[int] = set()
        for gid in range(mjm.ngeom):
            if mjm.geom_bodyid[gid] == self.spoon_body_id:
                self.spoon_geom_ids.add(gid)
            elif mjm.geom_bodyid[gid] == self.hook_body_id:
                self.hook_geom_ids.add(gid)

        # Finger geom IDs (either gripper can pick)
        def _finger_geoms(body_names: list[str]) -> set[int]:
            body_ids = set()
            for name in body_names:
                bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    body_ids.add(bid)
            return {gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] in body_ids}

        self.left_finger_geom_ids = _finger_geoms(
            ["left_lf_rot", "left_lf_down", "left_rf_rot", "left_rf_down"]
        )
        self.right_finger_geom_ids = _finger_geoms(
            ["right_lf_rot", "right_lf_down", "right_rf_rot", "right_rf_down"]
        )
        self.finger_geom_ids = self.left_finger_geom_ids | self.right_finger_geom_ids

    @staticmethod
    def _yaw_quat(yaw: float) -> list[float]:
        """Pure Z rotation quat (wxyz)."""
        c, s = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
        return [c, 0.0, 0.0, s]

    @staticmethod
    def _quat_mul(q1: list[float], q2: list[float]) -> list[float]:
        """Hamilton product of two wxyz quats."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ]

    def _spoon_quat_with_yaw(self, yaw: float) -> list[float]:
        """Compose a Z rotation on top of the spoon base orientation."""
        return self._quat_mul(self._yaw_quat(yaw), self.SPOON_BASE_QUAT)

    def _set_hook(self, mjm: mujoco.MjModel, mjd: mujoco.MjData,
                  xy: np.ndarray, yaw: float):
        """Move the hook body (static — edit mjm.body_pos/quat directly)."""
        self._hook_pos = [float(xy[0]), float(xy[1]), self.HOOK_SPAWN_Z]
        mjm.body_pos[self.hook_body_id] = self._hook_pos
        mjm.body_quat[self.hook_body_id] = self._yaw_quat(yaw)

    def _set_spoon(self, mjd: mujoco.MjData, pos: list[float], quat: list[float]):
        a = self.spoon_jnt_adr
        mjd.qpos[a:a+3] = pos
        mjd.qpos[a+3:a+7] = quat
        mjd.qvel[self.spoon_dof_adr:self.spoon_dof_adr+6] = 0.0

    def _sample_spoon_xy(self, hook_xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Uniform 8×15cm grid centered 30cm in front of hook (toward robot)."""
        dx = float(rng.uniform(-self.SPOON_GRID_HALF_X, self.SPOON_GRID_HALF_X))
        dy = float(rng.uniform(-self.SPOON_GRID_HALF_Y, self.SPOON_GRID_HALF_Y))
        return np.array([hook_xy[0] + self.SPOON_X_OFFSET + dx, hook_xy[1] + dy])

    def randomize(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, rng: np.random.Generator):
        # Place hook
        if self._hook_pool is None:
            self._hook_pool = PoissonPool(np.array(self.HOOK_POLYGON), n_target=50, margin=0.0)
        hook_xy = self._hook_pool.pop(rng)
        hook_yaw = float(rng.uniform(0, 2 * np.pi))
        self._set_hook(mjm, mjd, hook_xy, hook_yaw)

        # Place spoon on 15×15cm grid offset from hook
        spoon_xy = self._sample_spoon_xy(hook_xy, rng)
        spoon_yaw = float(rng.uniform(-0.3, 0.3))
        self._set_spoon(
            mjd,
            [float(spoon_xy[0]), float(spoon_xy[1]), self.SPOON_SPAWN_Z],
            self._spoon_quat_with_yaw(spoon_yaw),
        )

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        rng = np.random.default_rng(base_seed)
        hook_pool = PoissonPool(np.array(self.HOOK_POLYGON), n_target=n, margin=0.0)
        configs = []
        for i in range(n):
            hook_xy = hook_pool.pop(rng)
            hook_yaw = float(rng.uniform(0, 2 * np.pi))
            spoon_xy = self._sample_spoon_xy(hook_xy, rng)
            spoon_yaw = float(rng.uniform(-0.3, 0.3))
            configs.append({
                "seed": base_seed + i,
                "hook": {
                    "pos": [float(hook_xy[0]), float(hook_xy[1])],
                    "yaw": hook_yaw,
                },
                "spoon": {
                    "pos": [float(spoon_xy[0]), float(spoon_xy[1]), self.SPOON_SPAWN_Z],
                    "quat": self._spoon_quat_with_yaw(spoon_yaw),
                },
            })
        return configs

    def get_spawn_polygons(self) -> dict[str, tuple[tuple[float, float], ...]]:
        return {"hook": self.HOOK_POLYGON}

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        hook_cfg = config["hook"]
        self._set_hook(mjm, mjd, np.array(hook_cfg["pos"]), hook_cfg["yaw"])
        spoon_cfg = config["spoon"]
        self._set_spoon(mjd, spoon_cfg["pos"], spoon_cfg["quat"])

    @property
    def stages(self) -> tuple[str, ...]:
        return ("pick_spoon", "hang_spoon")

    def check_stages(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, bool]:
        """Check all task stages with a single contact scan.

        pick_spoon: spoon z > 5cm above table AND in contact with any gripper finger.
        hang_spoon: spoon z > 10cm above table AND contacting hook AND settled
                    AND NOT contacting any gripper finger.
        """
        spoon_contacts_finger = False
        spoon_contacts_hook = False

        for i in range(mjd.ncon):
            c = mjd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            g1_spoon = g1 in self.spoon_geom_ids
            g2_spoon = g2 in self.spoon_geom_ids
            if not (g1_spoon or g2_spoon):
                continue
            other = g2 if g1_spoon else g1
            if other in self.finger_geom_ids:
                spoon_contacts_finger = True
            if other in self.hook_geom_ids:
                spoon_contacts_hook = True

        spoon_z = mjd.xpos[self.spoon_body_id][2]
        spoon_qvel = mjd.qvel[self.spoon_dof_adr:self.spoon_dof_adr + 6]
        spoon_settled = np.linalg.norm(spoon_qvel) < self.SETTLE_VEL_THRESHOLD

        return {
            "pick_spoon": bool(
                spoon_z > 0.05 + self.z_offset
                and spoon_contacts_finger
            ),
            "hang_spoon": bool(
                spoon_z > 0.10 + self.z_offset
                and spoon_contacts_hook
                and spoon_settled
                and not spoon_contacts_finger
            ),
        }

    def check_success(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> bool:
        return self.check_stages(mjm, mjd)["hang_spoon"]

    def get_gaze_targets(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, np.ndarray]:
        return {
            "spoon": mjd.xpos[self.spoon_body_id].copy(),
            "hook": mjd.xpos[self.hook_body_id].copy(),
        }

    def get_clip_embedding(self, device: torch.device) -> torch.Tensor:
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
        return embedding[0].float()

    def get_spawn_z(self) -> float:
        return self.HOOK_SPAWN_Z
