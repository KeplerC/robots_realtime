"""DoubleMarkerInCupTask."""

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


class DoubleMarkerInCupTask(Task):
    """Place two green markers into a yellow cup.

    Scene: 1 yellow cup + 2 green markers on the table. Markers initialized
    lying on their side with long axis along +X (caps away from robot).
    Cup placed near center, markers randomly on left, right, or split.
    """

    ASSETS_DIR = Path(__file__).parent.parent / "assets"

    # Marker lying along +X: default long axis is Y after GLB→MuJoCo rotation,
    # so rotate -90° around Z to align with +X. May need 180° flip if cap end
    # is backwards — verify visually.
    MARKER_QUAT = [0.5, 0.5, 0.5, 0.5]  # wxyz — laying on side, long axis along world X
    MARKER_YAW_JITTER_DEG = 45.0  # +/- random yaw around world Z

    # Spawn polygons for Poisson-disc sampling
    CUP_POLYGON = ((0.15, -0.05), (0.5, -0.05), (0.5, 0.05), (0.15, 0.05))
    MARKER_LEFT_POLYGON = ((0.15, 0.05), (0.5, 0.05), (0.5, 0.30), (0.15, 0.30))
    MARKER_RIGHT_POLYGON = ((0.15, -0.30), (0.5, -0.30), (0.5, -0.05), (0.15, -0.05))
    # Rest heights computed from collision mesh vertex Z extents + 2mm margin
    CUP_SPAWN_Z = 0.043
    MARKER_SPAWN_Z = 0.011
    CUP_RADIUS_XY = 0.04
    MARKER_RADIUS_XY = 0.03
    MARKER_MIN_SEP = 0.13  # min distance between two markers on same side
    MARKER_LIFT_Z = 0.05        # Z threshold for "picked up"
    PLACE_XY_RADIUS = 0.035     # max XY distance from cup center for "placed in cup" (~cup outer radius)

    def __init__(self):
        self._clip_embedding: torch.Tensor | None = None

    @property
    def prompt(self) -> str:
        return "a green marker"

    def get_spawn_polygons(self) -> dict[str, tuple[tuple[float, float], ...]]:
        return {
            "yellow_cup": self.CUP_POLYGON,
            "green_marker_left": self.MARKER_LEFT_POLYGON,
            "green_marker_right": self.MARKER_RIGHT_POLYGON,
        }

    def get_spawn_z(self) -> float:
        return self.MARKER_SPAWN_Z

    def configure_scene(self, spec: mujoco.MjSpec):
        """Add yellow cup + 2 green markers to the scene."""
        col_kwargs = dict(
            friction=[1.0, 0.05, 0.01],
            solref=[0.008, 1.5],
            solimp=[0.95, 0.99, 0.001, 0.5, 2],
        )

        # --- Yellow cup ---
        cup_asset = MeshAsset.load(self.ASSETS_DIR / "yellow_cup")
        cup_asset.add_meshes_to_spec(spec, "yellow_cup")

        cup_body = spec.worldbody.add_body(name="yellow_cup", pos=[0.3, 0, 0.043])
        cup_body.add_freejoint(name="yellow_cup_jnt")

        if cup_asset.has_textures:
            cup_asset.add_visual_assets_to_spec(spec, "yellow_cup")
            cup_asset.add_visual_geoms(cup_body, "yellow_cup", mass=0.2)
        else:
            spec.add_mesh(name="yellow_cup", file=str(cup_asset.visual_mesh_path))
            cup_body.add_geom(
                name="yellow_cup_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="yellow_cup",
                rgba=[0.95, 0.85, 0.2, 1],
                mass=0.2,
                contype=0, conaffinity=0, group=0,
            )

        cup_asset.add_collision_geoms(cup_body, "yellow_cup", **col_kwargs)

        # --- Green markers (shared mesh assets, two body instances) ---
        marker_asset = MeshAsset.load(self.ASSETS_DIR / "green_marker")
        marker_asset.add_meshes_to_spec(spec, "green_marker")
        if marker_asset.has_textures:
            marker_asset.add_visual_assets_to_spec(spec, "green_marker")

        for idx in (1, 2):
            name = f"green_marker_{idx}"
            body = spec.worldbody.add_body(name=name, pos=[0.3, 0.2 * (2 * idx - 3), 0.011])
            jnt = body.add_freejoint(name=f"{name}_jnt")
            jnt.damping = 0.0001

            if marker_asset.has_textures:
                marker_asset.add_visual_geoms(body, name, mass=0.03)
            else:
                # Flat-color fallback (shared visual mesh registered once)
                if idx == 1:
                    spec.add_mesh(
                        name="green_marker_flat",
                        file=str(marker_asset.visual_mesh_path),
                    )
                body.add_geom(
                    name=f"{name}_visual",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname="green_marker_flat",
                    rgba=[0.2, 0.7, 0.2, 1],
                    mass=0.03,
                    contype=0, conaffinity=0, group=0,
                )

            marker_col_kwargs = dict(
                friction=[5.0, 5.0, 0.5],
                solref=[0.002, 1],
                solimp=[0.99, 0.999, 0.001, 0.5, 2],
            )
            marker_asset.add_collision_geoms(body, name, **marker_col_kwargs)

    def setup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        """Cache body/joint/geom IDs after compilation."""
        self.cup_body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "yellow_cup")
        self.cup_jnt_adr = mjm.jnt_qposadr[
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "yellow_cup_jnt")
        ]

        self.marker_body_ids = []
        self.marker_jnt_adrs = []
        self.marker_geom_ids = []
        for idx in (1, 2):
            name = f"green_marker_{idx}"
            bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)
            self.marker_body_ids.append(bid)
            self.marker_jnt_adrs.append(
                mjm.jnt_qposadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_jnt")]
            )
            self.marker_geom_ids.append(
                {gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] == bid}
            )

        self.cup_geom_ids = {
            gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] == self.cup_body_id
        }

        def _finger_geoms(body_names):
            bids = {
                mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in body_names
                if mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n) >= 0
            }
            return {gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] in bids}

        self.finger_geom_ids = _finger_geoms([
            "left_lf_rot", "left_lf_down", "left_rf_rot", "left_rf_down",
            "right_lf_rot", "right_lf_down", "right_rf_rot", "right_rf_down",
        ])

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        """Generate n deterministic eval configs using Poisson-disc sampling.

        Cup sampled from center polygon. Markers sampled from left/right
        polygons with random layout (both-left, both-right, or split).
        """
        rng = np.random.default_rng(base_seed)

        cup_pool = PoissonPool(
            polygon=np.asarray(self.CUP_POLYGON, dtype=np.float64),
            n_target=n,
            margin=self.CUP_RADIUS_XY,
        )
        left_pool = PoissonPool(
            polygon=np.asarray(self.MARKER_LEFT_POLYGON, dtype=np.float64),
            n_target=n,
            margin=self.MARKER_RADIUS_XY,
        )
        right_pool = PoissonPool(
            polygon=np.asarray(self.MARKER_RIGHT_POLYGON, dtype=np.float64),
            n_target=n,
            margin=self.MARKER_RADIUS_XY,
        )

        configs = []
        for i in range(n):
            cup_xy = cup_pool.pop(rng)
            cup_pos = [float(cup_xy[0]), float(cup_xy[1]), self.CUP_SPAWN_Z]

            # 0 = both left, 1 = both right, 2 = one left + one right
            layout = int(rng.integers(0, 3))

            if layout == 0:
                m1_xy = left_pool.pop(rng)
                m2_xy = left_pool.pop(rng)
                while np.linalg.norm(m1_xy - m2_xy) < self.MARKER_MIN_SEP:
                    m2_xy = left_pool.pop(rng)
            elif layout == 1:
                m1_xy = right_pool.pop(rng)
                m2_xy = right_pool.pop(rng)
                while np.linalg.norm(m1_xy - m2_xy) < self.MARKER_MIN_SEP:
                    m2_xy = right_pool.pop(rng)
            else:
                m1_xy = left_pool.pop(rng)
                m2_xy = right_pool.pop(rng)

            # Random yaw jitter for each marker
            def _jittered_marker_quat():
                yaw = np.deg2rad(rng.uniform(-self.MARKER_YAW_JITTER_DEG, self.MARKER_YAW_JITTER_DEG))
                c, s = np.cos(yaw / 2), np.sin(yaw / 2)
                q_yaw = np.array([c, 0.0, 0.0, s])  # wxyz
                q_base = np.array(self.MARKER_QUAT)
                # Hamilton product: q_yaw * q_base (wxyz)
                w1, x1, y1, z1 = q_yaw
                w2, x2, y2, z2 = q_base
                q = np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ])
                q /= np.linalg.norm(q)
                return [float(v) for v in q]

            configs.append({
                "seed": base_seed + i,
                "layout": layout,
                "objects": {
                    "yellow_cup": {
                        "pos": cup_pos,
                        "quat": [1.0, 0.0, 0.0, 0.0],
                    },
                    "green_marker_1": {
                        "pos": [float(m1_xy[0]), float(m1_xy[1]), self.MARKER_SPAWN_Z],
                        "quat": _jittered_marker_quat(),
                    },
                    "green_marker_2": {
                        "pos": [float(m2_xy[0]), float(m2_xy[1]), self.MARKER_SPAWN_Z],
                        "quat": _jittered_marker_quat(),
                    },
                },
            })
        return configs

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        """Set freejoint qpos for cup and both markers."""
        objects = config["objects"]

        # Cup
        a = self.cup_jnt_adr
        mjd.qpos[a : a + 3] = objects["yellow_cup"]["pos"]
        mjd.qpos[a + 3 : a + 7] = objects["yellow_cup"]["quat"]

        # Markers
        for idx in range(2):
            name = f"green_marker_{idx + 1}"
            a = self.marker_jnt_adrs[idx]
            mjd.qpos[a : a + 3] = objects[name]["pos"]
            mjd.qpos[a + 3 : a + 7] = objects[name]["quat"]

    @property
    def stages(self) -> tuple[str, ...]:
        return ("pick_marker_1", "place_marker_1", "pick_marker_2", "place_marker_2")

    def check_stages(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, bool]:
        marker_contacts_finger = [False, False]
        marker_contacts_cup = [False, False]

        for i in range(mjd.ncon):
            c = mjd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for idx in range(2):
                m_geoms = self.marker_geom_ids[idx]
                if g1 in m_geoms or g2 in m_geoms:
                    other = g2 if g1 in m_geoms else g1
                    if other in self.finger_geom_ids:
                        marker_contacts_finger[idx] = True
                    if other in self.cup_geom_ids:
                        marker_contacts_cup[idx] = True

        cup_pos = mjd.xpos[self.cup_body_id]
        cup_xy = cup_pos[:2]

        def _picked(idx):
            z = mjd.xpos[self.marker_body_ids[idx]][2]
            return z > self.MARKER_LIFT_Z + self.z_offset and marker_contacts_finger[idx]

        def _placed(idx):
            marker_pos = mjd.xpos[self.marker_body_ids[idx]]
            dist_xy = np.linalg.norm(marker_pos[:2] - cup_xy)
            return (
                marker_contacts_cup[idx]
                and not marker_contacts_finger[idx]
                and marker_pos[2] > cup_pos[2]
                and dist_xy < self.PLACE_XY_RADIUS
            )

        return {
            "pick_marker_1": _picked(0),
            "place_marker_1": _placed(0),
            "pick_marker_2": _picked(1),
            "place_marker_2": _placed(1),
        }

    def check_success(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> bool:
        return self.check_stages(mjm, mjd)["place_marker_2"]

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
