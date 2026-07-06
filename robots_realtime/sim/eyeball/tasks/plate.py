"""PlacePlateInRackTask."""

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


class PlacePlateInRackTask(Task):
    """Bimanual task: right arm picks up plate, left arm guides, plate placed upright in dish rack.

    Scene: 1 dish rack (dynamic, randomized position) + 1 plate (dynamic, randomized position).
    The plate starts lying flat near the arms. Both arms grip simultaneously (right arm picks
    up, left arm provides resistance), then plate is placed upright into the rack slots.

    Coordinate notes:
    - Rack GLB: Z-up in model space — identity quaternion, z offset lifts base to table.
      Effective size after 0.22 scale: 0.22m wide (worldY), 0.08m deep (worldX), 0.07m tall (Z).
    - Plate GLB: Y is thickness axis, XZ is the face plane. Lying flat: quat rotates Y→Z.
      After mesh_scale=0.6: radius 0.076m (~15cm diameter).

    Stages:
        pick_plate  — plate lifted off table (at least one arm in contact)
        place_plate — plate upright in rack, settled, no fingers touching
    """

    ASSETS_DIR = Path(__file__).parent.parent / "assets"

    # Rack: Z-up model. Base yaw = 90° around Z (width along world Y = "horizontal").
    # Per-episode yaw jitter of ±30° is applied on top, stored in config/qpos.
    RACK_BASE_YAW = np.pi / 2          # 90° base
    RACK_YAW_RANGE = np.pi / 6         # ±30° jitter
    RACK_SPAWN_Z   = 0.037             # body z so rack bottom sits at table (z=0): 0.168 * 0.22
    # In-rack check uses rack local frame (model space, after 0.22 scale):
    RACK_HALF_LOCAL_X = 0.160          # half of width — relaxed for plate center offset in slots
    RACK_HALF_LOCAL_Y = 0.080          # half of depth — relaxed to allow plate center offset
    # Rack spawn polygon: x: 0.49-0.56.
    # No-collision guarantee: rack_min(0.49) - rack_max_extent(0.090) - plate_max(0.33) - plate_radius(0.076) = 0.004m > 0
    RACK_POLYGON = ((0.49, -0.18), (0.56, -0.18), (0.56, 0.18), (0.49, 0.18))

    # Plate: Y is thickness axis. Flat quat: Y→Z (90° around X). mesh_scale=0.75 → radius 0.095m (~19cm diameter)
    PLATE_FLAT_QUAT = [float(np.sqrt(2) / 2), float(np.sqrt(2) / 2), 0.0, 0.0]  # wxyz
    PLATE_SPAWN_Z   = 0.001
    PLATE_RADIUS    = 0.095            # after 0.42 bake * 0.75 mesh_scale (~19cm diameter)
    # Plate spawn polygon: x: 0.15-0.33, y: ±0.20 (arms at y=±0.30, keep plate between them).
    # Clearance: rack_min(0.49) - rack_extent(0.090) - plate_max(0.33) - plate_radius(0.076) = 0.004m
    PLATE_POLYGON = ((0.15, -0.20), (0.33, -0.20), (0.33, 0.20), (0.15, 0.20))

    # Success thresholds
    PICK_Z_THRESH     = 0.05  # 5cm above table surface
    SETTLE_VEL_THRESH = 0.1

    def __init__(self):
        self._clip_embedding: torch.Tensor | None = None
        self._plate_pool: PoissonPool | None = None
        self._rack_pool:  PoissonPool | None = None

    @property
    def prompt(self) -> str:
        return "a white plate"

    def configure_scene(self, spec: mujoco.MjSpec):
        """Add dish rack (dynamic) and plate (dynamic) as mesh assets."""
        col_kwargs = dict(
            friction=[1.0, 0.05, 0.01],
            solref=[0.002, 1],
            solimp=[0.99, 0.999, 0.001, 0.5, 2],
        )

        # --- Dish rack (dynamic — freejoint so position can be randomized) ---
        rack_asset = MeshAsset.load(self.ASSETS_DIR / "dish_rack")
        rack_asset.add_meshes_to_spec(spec, "dish_rack")

        rack_body = spec.worldbody.add_body(
            name="dish_rack",
            pos=[0.52, 0.0, self.RACK_SPAWN_Z],
            quat=self._rack_quat(self.RACK_BASE_YAW),
        )
        rack_body.add_freejoint(name="dish_rack_jnt")

        if rack_asset.has_textures:
            rack_asset.add_visual_assets_to_spec(spec, "dish_rack")
            rack_asset.add_visual_geoms(rack_body, "dish_rack", mass=3.0)
        else:
            spec.add_mesh(name="dish_rack_visual_mesh", file=str(rack_asset.visual_mesh_path))
            rack_body.add_geom(
                name="dish_rack_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="dish_rack_visual_mesh",
                rgba=[0.6, 0.6, 0.6, 1],
                mass=3.0,
                contype=0, conaffinity=0, group=0,
            )
        rack_asset.add_collision_geoms(rack_body, "dish_rack", **col_kwargs)

        # --- Plate (dynamic) ---
        plate_asset = MeshAsset.load(self.ASSETS_DIR / "plate")
        plate_asset.add_meshes_to_spec(spec, "plate")

        plate_body = spec.worldbody.add_body(
            name="plate",
            pos=[0.25, 0.0, self.PLATE_SPAWN_Z],
            quat=self.PLATE_FLAT_QUAT,
        )
        plate_body.add_freejoint(name="plate_jnt")

        if plate_asset.has_textures:
            plate_asset.add_visual_assets_to_spec(spec, "plate")
            plate_asset.add_visual_geoms(plate_body, "plate", mass=0.3)
        else:
            m = spec.add_mesh(name="plate_visual_mesh", file=str(plate_asset.visual_mesh_path))
            if plate_asset._scale_vec:
                m.scale = plate_asset._scale_vec
            plate_body.add_geom(
                name="plate_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="plate_visual_mesh",
                rgba=[0.95, 0.95, 0.95, 1],
                mass=0.3,
                contype=0, conaffinity=0, group=0,
            )
        plate_asset.add_collision_geoms(plate_body, "plate", **col_kwargs)

    def setup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        self.rack_body_id  = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "dish_rack")
        self.plate_body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "plate")

        rack_jnt_id        = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "dish_rack_jnt")
        self.rack_jnt_adr  = mjm.jnt_qposadr[rack_jnt_id]

        plate_jnt_id       = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "plate_jnt")
        self.plate_jnt_adr = mjm.jnt_qposadr[plate_jnt_id]
        self.plate_dof_adr = mjm.jnt_dofadr[plate_jnt_id]

        self.plate_geom_ids = {gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] == self.plate_body_id}
        self.rack_geom_ids  = {gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] == self.rack_body_id}

        def _finger_geoms(names):
            bids = {mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
            return {gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] in bids}

        self.left_finger_geom_ids  = _finger_geoms(["left_lf_rot",  "left_lf_down",  "left_rf_rot",  "left_rf_down"])
        self.right_finger_geom_ids = _finger_geoms(["right_lf_rot", "right_lf_down", "right_rf_rot", "right_rf_down"])
        self.table_geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "table_top")

    @staticmethod
    def _rack_quat(yaw: float) -> list[float]:
        """Quat for given yaw angle (rotation around world Z), wxyz."""
        c, s = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
        return [c, 0.0, 0.0, s]

    def _set_rack(self, mjd: mujoco.MjData, xy: tuple, yaw: float):
        a = self.rack_jnt_adr
        mjd.qpos[a:a+3]   = [float(xy[0]), float(xy[1]), self.RACK_SPAWN_Z]
        mjd.qpos[a+3:a+7] = self._rack_quat(yaw)

    def _set_plate(self, mjd: mujoco.MjData, xy: tuple):
        a = self.plate_jnt_adr
        mjd.qpos[a:a+3]   = [float(xy[0]), float(xy[1]), self.PLATE_SPAWN_Z]
        mjd.qpos[a+3:a+7] = self.PLATE_FLAT_QUAT

    def randomize(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, rng: np.random.Generator):
        """Randomize plate and rack positions + rack yaw (called per teleop episode reset)."""
        if self._rack_pool is None:
            self._rack_pool  = PoissonPool(polygon=np.array(self.RACK_POLYGON),  n_target=50, margin=0.0)
        if self._plate_pool is None:
            self._plate_pool = PoissonPool(polygon=np.array(self.PLATE_POLYGON), n_target=50, margin=0.0)
        yaw = float(rng.uniform(
            self.RACK_BASE_YAW - self.RACK_YAW_RANGE,
            self.RACK_BASE_YAW + self.RACK_YAW_RANGE,
        ))
        self._set_rack(mjd,  self._rack_pool.pop(rng),  yaw)
        self._set_plate(mjd, self._plate_pool.pop(rng))

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        rng        = np.random.default_rng(base_seed)
        rack_pool  = PoissonPool(polygon=np.array(self.RACK_POLYGON),  n_target=n, margin=0.0)
        plate_pool = PoissonPool(polygon=np.array(self.PLATE_POLYGON), n_target=n, margin=0.0)
        configs = []
        for i in range(n):
            rack_xy  = rack_pool.pop(rng)
            plate_xy = plate_pool.pop(rng)
            yaw = float(rng.uniform(
                self.RACK_BASE_YAW - self.RACK_YAW_RANGE,
                self.RACK_BASE_YAW + self.RACK_YAW_RANGE,
            ))
            configs.append({
                "seed": base_seed + i,
                "objects": {
                    "dish_rack": {
                        "pos":  [float(rack_xy[0]),  float(rack_xy[1]),  self.RACK_SPAWN_Z],
                        "quat": self._rack_quat(yaw),
                    },
                    "plate": {
                        "pos":  [float(plate_xy[0]), float(plate_xy[1]), self.PLATE_SPAWN_Z],
                        "quat": self.PLATE_FLAT_QUAT,
                    },
                },
            })
        return configs

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        objs = config["objects"]
        rack_cfg = objs["dish_rack"]
        a = self.rack_jnt_adr
        mjd.qpos[a:a+3]   = rack_cfg["pos"]
        mjd.qpos[a+3:a+7] = rack_cfg["quat"]
        self._set_plate(mjd, objs["plate"]["pos"][:2])

    @property
    def stages(self) -> tuple[str, ...]:
        return ("pick_plate", "place_plate")

    def check_stages(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, bool]:
        plate_contacts_left  = False
        plate_contacts_right = False
        plate_contacts_rack  = False
        plate_contacts_table = False

        for i in range(mjd.ncon):
            c = mjd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if not (g1 in self.plate_geom_ids or g2 in self.plate_geom_ids):
                continue
            other = g2 if g1 in self.plate_geom_ids else g1
            if other in self.left_finger_geom_ids:
                plate_contacts_left = True
            if other in self.right_finger_geom_ids:
                plate_contacts_right = True
            if other in self.rack_geom_ids:
                plate_contacts_rack = True
            if other == self.table_geom_id:
                plate_contacts_table = True

        plate_pos  = mjd.xpos[self.plate_body_id]
        plate_xmat = mjd.xmat[self.plate_body_id].reshape(3, 3)

        # Upright = plate thickness axis (local Y) is horizontal (low world-Z component)
        plate_upright = abs(plate_xmat[:, 1][2]) < 0.65

        # Within rack footprint — transform plate into rack local frame for any yaw
        rack_pos  = mjd.xpos[self.rack_body_id]
        rack_xmat = mjd.xmat[self.rack_body_id].reshape(3, 3)
        plate_local = rack_xmat.T @ (plate_pos - rack_pos)
        in_rack_xy = (
            abs(plate_local[0]) < self.RACK_HALF_LOCAL_X and
            abs(plate_local[1]) < self.RACK_HALF_LOCAL_Y
        )

        plate_qvel    = mjd.qvel[self.plate_dof_adr:self.plate_dof_adr+6]
        plate_settled = np.linalg.norm(plate_qvel) < self.SETTLE_VEL_THRESH

        place_ok = (
            plate_contacts_rack
            and in_rack_xy
            and plate_upright
            and not plate_contacts_left
            and not plate_contacts_right
            and plate_settled
        )

        if not place_ok and plate_contacts_rack:
            vel_norm = float(np.linalg.norm(plate_qvel))
            upright_val = float(abs(plate_xmat[:, 1][2]))
            print(f"[place_plate] rack={plate_contacts_rack} in_rack_xy={in_rack_xy}"
                  f"(local={plate_local[0]:.3f},{plate_local[1]:.3f})"
                  f" upright={plate_upright}({upright_val:.2f})"
                  f" no_L={not plate_contacts_left} no_R={not plate_contacts_right}"
                  f" settled={plate_settled}(vel={vel_norm:.2f})")

        pick_ok = (
            plate_pos[2] > self.PICK_Z_THRESH
            and not plate_contacts_table
            and (plate_contacts_right or plate_contacts_left)
        )

        return {
            "pick_plate": pick_ok or place_ok,
            "place_plate": place_ok,
        }

    def check_success(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> bool:
        return self.check_stages(mjm, mjd)["place_plate"]

    def get_gaze_targets(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, np.ndarray]:
        return {
            "plate":     mjd.xpos[self.plate_body_id].copy(),
            "dish_rack": mjd.xpos[self.rack_body_id].copy(),
        }

    def get_spawn_polygons(self) -> dict[str, tuple[tuple[float, float], ...]]:
        return {"plate": self.PLATE_POLYGON, "dish_rack": self.RACK_POLYGON}

    def get_spawn_z(self) -> float:
        return self.PLATE_SPAWN_Z

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
            tokens    = tokenizer([self.prompt]).to(device)
            embedding = model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        self._clip_embedding = embedding[0].float()
        return self._clip_embedding
