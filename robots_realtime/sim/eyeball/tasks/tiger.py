"""PickUpTigerTask."""

from functools import partial
from pathlib import Path

import mujoco
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ..decompose_mesh import MeshAsset
from ..poisson_utils import PoissonPool
from ..polygon_utils import resolve_spawn_collision
from .base import Task


class PickUpTigerTask(Task):
    """Pick up a tiger figurine from the table.

    Scene: 1 tiger object on the table. Robot must grasp and lift it.

    Args:
        spawn_region: Controls where the tiger spawns.
            - ``"full"`` — full reachable workspace of the right arm (eval distribution)
            - ``"right_half"`` — right half of the workspace (y ∈ [-0.30, -0.05])
            - ``"middle_strip"`` — thin strip down the middle (y ∈ [-0.08, -0.02])
    """

    ASSETS_DIR = Path(__file__).parent.parent / "assets"
    # Lay long axis (Z in mesh space) flat along +X, belly facing down
    TIGER_BASE_QUAT = np.array([0.5, 0.5, 0.5, 0.5])  # wxyz

    _SPAWN_POLYGONS: dict[str, tuple[tuple[float, float], ...]] = {
        "full":         ((0.20, -0.30), (0.50, -0.30), (0.50, 0.20), (0.20, 0.20)),
        "right_half":   ((0.20, -0.30), (0.50, -0.30), (0.50, -0.05), (0.20, -0.05)),
        "middle_strip": ((0.20, -0.08), (0.50, -0.08), (0.50, -0.02), (0.20, -0.02)),
    }
    _VALID_SPAWN_REGIONS = {"full", "right_half", "middle_strip"}

    SPAWN_Z = 0.03
    OBJECT_RADIUS_XY = 0.08  # half-diagonal of tiger bounding box
    THETA_RANGE = np.pi  # ±180°

    def __init__(self, spawn_region: str = "full"):
        if spawn_region not in self._VALID_SPAWN_REGIONS:
            raise ValueError(
                f"spawn_region must be one of {self._VALID_SPAWN_REGIONS}, got {spawn_region!r}"
            )
        self._spawn_region = spawn_region
        self.SPAWN_POLYGON_XY = self._SPAWN_POLYGONS[spawn_region]
        self._clip_embedding: torch.Tensor | None = None
        self._poisson_pool: PoissonPool | None = None

    @property
    def prompt(self) -> str:
        return "a stuffed tiger"

    def configure_scene(self, spec: mujoco.MjSpec):
        """Add tiger object to the scene."""
        col_kwargs = dict(
            friction=[1.0, 0.05, 0.01],
            solref=[0.002, 1],
            solimp=[0.99, 0.999, 0.001, 0.5, 2],
        )

        tiger_asset = MeshAsset.load(self.ASSETS_DIR / "tiger")
        tiger_asset.add_meshes_to_spec(spec, "tiger")

        body = spec.worldbody.add_body(name="tiger", pos=[0.35, 0.0, 0.03])
        body.add_freejoint(name="tiger_jnt")

        if tiger_asset.has_textures:
            tiger_asset.add_visual_assets_to_spec(spec, "tiger")
            tiger_asset.add_visual_geoms(body, "tiger", mass=0.1)
        else:
            spec.add_mesh(name="tiger", file=str(tiger_asset.visual_mesh_path))
            body.add_geom(
                name="tiger_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="tiger",
                rgba=[0.9, 0.5, 0.1, 1],
                mass=0.1,
                contype=0, conaffinity=0, group=0,
            )

        tiger_asset.add_collision_geoms(body, "tiger", **col_kwargs)

    def setup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        """Cache body/joint/geom IDs after compilation."""
        self.tiger_body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "tiger")
        tiger_jnt_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "tiger_jnt")
        self.tiger_jnt_adr = mjm.jnt_qposadr[tiger_jnt_id]
        self.tiger_dof_adr = mjm.jnt_dofadr[tiger_jnt_id]
        self.tiger_geom_ids = {
            gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] == self.tiger_body_id
        }

        def _finger_geoms(body_names):
            body_ids = {
                mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in body_names
            }
            return {gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] in body_ids}

        self.right_finger_geom_ids = _finger_geoms(
            ["right_lf_rot", "right_lf_down", "right_rf_rot", "right_rf_down"]
        )

        self._spawn_object = ("tiger", self.tiger_body_id, self.tiger_jnt_adr)

    def _theta_to_quat(self, theta: float) -> list[float]:
        """Compose a Z rotation on top of the flat base orientation (wxyz)."""
        cz, sz = np.cos(theta / 2), np.sin(theta / 2)
        zq = np.array([cz, 0.0, 0.0, sz])
        bq = self.TIGER_BASE_QUAT
        w = zq[0]*bq[0] - zq[1]*bq[1] - zq[2]*bq[2] - zq[3]*bq[3]
        x = zq[0]*bq[1] + zq[1]*bq[0] + zq[2]*bq[3] - zq[3]*bq[2]
        y = zq[0]*bq[2] - zq[1]*bq[3] + zq[2]*bq[0] + zq[3]*bq[1]
        z = zq[0]*bq[3] + zq[1]*bq[2] - zq[2]*bq[1] + zq[3]*bq[0]
        return [float(w), float(x), float(y), float(z)]

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        """Generate n deterministic Poisson-disc eval configs.

        Uses the active spawn polygon (respects ``spawn_region``).
        Tiger gets a random Z rotation up to ±45°.
        """
        rng = np.random.default_rng(base_seed)
        pool = PoissonPool(
            polygon=np.array(self.SPAWN_POLYGON_XY),
            n_target=n,
            margin=0.0,
        )

        configs = []
        for i in range(n):
            xy = pool.pop(rng)
            theta = float(rng.uniform(-self.THETA_RANGE, self.THETA_RANGE))
            configs.append({
                "seed": base_seed + i,
                "objects": {
                    "tiger": {
                        "pos": [float(xy[0]), float(xy[1]), self.SPAWN_Z],
                        "quat": self._theta_to_quat(theta),
                    },
                },
            })
        return configs

    def randomize(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, rng: np.random.Generator):
        """Place tiger using Poisson-disc sampling."""
        if self._poisson_pool is None:
            self._poisson_pool = PoissonPool(
                polygon=np.array(self.SPAWN_POLYGON_XY),
                n_target=50,
                margin=0.0,
            )

        xy = self._poisson_pool.pop(rng)
        theta = float(rng.uniform(-self.THETA_RANGE, self.THETA_RANGE))
        quat = self._theta_to_quat(theta)

        _, body_id, jnt_adr = self._spawn_object
        mjd.qpos[jnt_adr : jnt_adr + 3] = [xy[0], xy[1], self.SPAWN_Z]
        mjd.qpos[jnt_adr + 3 : jnt_adr + 7] = quat
        resolve_spawn_collision(
            mjm, mjd, body_id, jnt_adr, self.SPAWN_Z,
            np.array(self.SPAWN_POLYGON_XY),
            self.OBJECT_RADIUS_XY,
            set(), rng,
            label="tiger",
        )

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        """Apply a config dict to set tiger position via freejoint qpos."""
        obj = config["objects"]["tiger"]
        a = self.tiger_jnt_adr
        mjd.qpos[a : a + 3] = obj["pos"]
        mjd.qpos[a + 3 : a + 7] = obj["quat"]

    @property
    def stages(self) -> tuple[str, ...]:
        return ("pick_tiger",)

    def check_stages(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, bool]:
        tiger_z = mjd.xpos[self.tiger_body_id][2]
        tiger_contacts_right = False
        for i in range(mjd.ncon):
            c = mjd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.tiger_geom_ids or g2 in self.tiger_geom_ids:
                other = g2 if g1 in self.tiger_geom_ids else g1
                if other in self.right_finger_geom_ids:
                    tiger_contacts_right = True
                    break
        return {
            "pick_tiger": tiger_z > 0.05 + self.z_offset and tiger_contacts_right,
        }

    def check_success(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> bool:
        return self.check_stages(mjm, mjd)["pick_tiger"]

    def get_clip_embedding(self, device: torch.device) -> torch.Tensor:
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

    def get_spawn_polygons(self) -> dict[str, tuple[tuple[float, float], ...]]:
        return {"tiger": self.SPAWN_POLYGON_XY}

    def get_spawn_z(self) -> float:
        return self.SPAWN_Z


# Aliases used by scripts/draw_spawn_regions.py
PickUpTigerTaskRightHalf = partial(PickUpTigerTask, spawn_region="right_half")
PickUpTigerTaskMiddleStrip = partial(PickUpTigerTask, spawn_region="middle_strip")
