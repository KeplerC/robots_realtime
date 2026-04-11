"""MedicalTrayTask — pick tray from shelf, place on table, pick blue-cap bottle."""

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


class MedicalTrayTask(Task):
    """Bimanual: tray from rack shelf → table → pick blue-cap pill bottle.

    Scene:
        - 2-shelf storage rack (dynamic, randomized position + yaw)
        - Medical tray (dynamic, starts on random shelf — either of the 2)
        - Blue-cap pill bottle (target) among distractors on the tray
        - Distractors: 1 plain pill bottle, 1 thermometer, 1 tissue box

    Stages:
        pick_tray   — tray lifted off shelf, both arms contacting tray
        place_tray  — tray settled on table, no arm contact
        pick_bottle — blue-cap bottle lifted, arm contact
    """

    ASSETS_DIR = Path(__file__).parent.parent / "assets"

    # ── Y-up → Z-up rotation (90° around X) ─────────────────────────────
    _Q_Y2Z = [float(np.sqrt(2) / 2), float(np.sqrt(2) / 2), 0.0, 0.0]  # wxyz

    # ── Rack (static body) ─────────────────────────────────────────────
    # new_rack mesh is Z-up; identity quat keeps it upright with no rotation needed.
    RACK_SCALE = 0.5653
    # Matched so bottom of new_rack mesh sits at same table-relative Z as old shelf.
    # new_rack raw Z_min = -0.3513; target bottom world z = -0.0409 (old shelf bottom).
    # RACK_BODY_Z = -0.0409 - (-0.3513 * 0.5653) = 0.1579
    RACK_BODY_Z = 0.1579
    RACK_MASS = 1000.0  # effectively static (no freejoint)
    RACK_SPAWN_POLYGON = ((0.75, -0.05), (0.77, -0.05), (0.77, 0.05), (0.75, 0.05))
    RACK_BASE_YAW = 3 * np.pi / 2  # open face toward YAM arms (confirmed correct)
    RACK_YAW_RANGE = np.pi / 24   # ±10° jitter

    # 2-shelf rack: bottom floor + 1 shelf board (raw mesh center Z).
    # Board top surfaces measured from collision geometry: shelf0=0.027m, shelf1=0.241m world.
    # Values are set ~5mm above the actual board top so the tray spawns with clearance
    # (prevents the tray bottom from being inside the board and getting ejected).
    _SHELF_MESH_Z = (-0.232, 0.146)
    _SPAWNABLE_SHELVES = (0, 1)
    _SHELF_BOARD_HALF_THICK = 0.01  # raw mesh units; top surface ≈ center + this

    # ── Tray (dynamic, freejoint) ────────────────────────────────────────
    # Tray mesh: thin in Z (Z=0.104), flat in XY. Identity quat → lies flat.
    TRAY_SCALE = 0.3023
    TRAY_Z_OFFSET = 0.1324 * 0.3023    # body origin to bottom (|z_min| from mesh)
    # Inner floor is ~7mm above exterior bottom (upward-facing faces at z≈-0.112 raw)
    TRAY_SURFACE_Z_LOCAL = -0.112 * 0.3023    # body origin to inside floor
    TRAY_MASS = 0.060
    TRAY_Y_JITTER = 0.0            # tray centered on shelf (no jitter)
    TRAY_PROTRUDE = 0.1275          # protrusion from rack opening (local -Y)

    # ── Blue-cap bottle (target) ─────────────────────────────────────────
    BOTTLE_SCALE = [0.075, 0.075, 0.045]   # xy enlarged, z flattened
    BOTTLE_Z_OFFSET = 0.4998 * 0.045  # ≈ 0.022
    BOTTLE_MASS = 0.09

    # ── Plain bottle (distractor, same mesh loaded twice) ────────────────
    PLAIN_SCALE = [0.075, 0.075, 0.045]
    PLAIN_Z_OFFSET = 0.4999 * 0.045   # ≈ 0.022
    PLAIN_MASS = 0.09

    # ── Thermometer (identity quat — lies flat, long axis = mesh Y) ──────
    THERMO_SCALE = 0.102
    THERMO_Z_OFFSET = 0.0689 * 0.102   # mesh Z_min * scale ≈ 0.0070
    THERMO_MASS = 0.02
    THERMO_QUAT = [1.0, 0.0, 0.0, 0.0]

    # ── Tissue box ───────────────────────────────────────────────────────
    TISSUE_SCALE = 0.1105
    TISSUE_Z_OFFSET = 0.457 * 0.1105    # ≈ 0.050
    TISSUE_MASS = 0.06

    # ── Object slots on tray (relative to tray center XY) ───────────────
    # Tray interior is ±15cm × ±14cm at TRAY_SCALE=0.3023.
    _SLOTS = [
        (-0.068, -0.068),   # back-left
        (-0.068,  0.068),   # back-right
        ( 0.068, -0.068),   # front-left
        ( 0.068,  0.068),   # front-right
    ]
    _SLOT_JITTER = 0.017   # ±1.7cm jitter
    _SPAWN_Z_MARGIN = 0.005

    # ── Stage thresholds ─────────────────────────────────────────────────
    PICK_TRAY_LIFT = 0.05        # tray must rise 5cm above shelf rest z
    PICK_BOTTLE_Z = 0.10         # bottle center must be above 10cm world z
    SETTLE_VEL_THRESH = 0.25

    # ── Collision parameters ─────────────────────────────────────────────
    _COL_KW = dict(
        friction=[1.0, 0.05, 0.01],
        solref=[0.002, 1],
        solimp=[0.99, 0.999, 0.001, 0.5, 2],
    )

    # Object names that sit on the tray (must match _SLOTS length)
    _TRAY_OBJECTS = [
        "blue_cap_bottle",
        "plain_bottle_0",
        "thermometer",
        "tissue_box",
    ]

    def __init__(self):
        self._clip_embedding: torch.Tensor | None = None
        self._current_shelf_z: float = 0.0  # set per-episode in apply_eval_config

    @property
    def prompt(self) -> str:
        return "a pill bottle with a blue cap"

    # ── Scene construction ───────────────────────────────────────────────

    def _add_mesh_object(
        self,
        spec: mujoco.MjSpec,
        asset_dir: str,
        body_name: str,
        mesh_prefix: str,
        init_pos: list[float],
        init_quat: list[float],
        mass: float,
    ):
        """Add a mesh-based dynamic object (freejoint) to the spec."""
        ck = self._COL_KW
        asset = MeshAsset.load(self.ASSETS_DIR / asset_dir)
        asset.add_meshes_to_spec(spec, mesh_prefix)

        body = spec.worldbody.add_body(name=body_name, pos=init_pos, quat=init_quat)
        body.add_freejoint(name=f"{body_name}_jnt")

        if asset.has_textures:
            asset.add_visual_assets_to_spec(spec, mesh_prefix)
            asset.add_visual_geoms(body, mesh_prefix, mass=mass)
        else:
            m = spec.add_mesh(
                name=f"{mesh_prefix}_vis_mesh",
                file=str(asset.visual_mesh_path),
            )
            if asset._scale_vec:
                m.scale = asset._scale_vec
            body.add_geom(
                name=f"{body_name}_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"{mesh_prefix}_vis_mesh",
                rgba=[0.8, 0.8, 0.8, 1],
                mass=mass,
                contype=0, conaffinity=0, group=0,
            )
        asset.add_collision_geoms(body, mesh_prefix, **ck)

    def configure_scene(self, spec: mujoco.MjSpec):
        ck = self._COL_KW

        # ── Storage rack ─────────────────────────────────────────────
        rack_asset = MeshAsset.load(self.ASSETS_DIR / "new_rack")
        rack_asset.add_meshes_to_spec(spec, "new_rack")

        rack_body = spec.worldbody.add_body(
            name="storage_rack",
            pos=[0.76, 0.0, self.RACK_BODY_Z],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
        # No freejoint — rack is static to prevent solver jitter in the
        # stacked contact chain (objects → tray → shelf → table).

        if rack_asset.has_textures:
            rack_asset.add_visual_assets_to_spec(spec, "new_rack")
            rack_asset.add_visual_geoms(rack_body, "new_rack", mass=self.RACK_MASS)
        else:
            m = spec.add_mesh(
                name="new_rack_vis_mesh",
                file=str(rack_asset.visual_mesh_path),
            )
            if rack_asset._scale_vec:
                m.scale = rack_asset._scale_vec
            rack_body.add_geom(
                name="storage_rack_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="new_rack_vis_mesh",
                rgba=[0.6, 0.5, 0.4, 1],
                mass=self.RACK_MASS,
                contype=0, conaffinity=0, group=0,
            )
        rack_asset.add_collision_geoms(rack_body, "new_rack", **ck)

        # ── Tray ─────────────────────────────────────────────────────
        self._add_mesh_object(
            spec,
            asset_dir="tray",
            body_name="tray",
            mesh_prefix="tray",
            init_pos=[0.77, 0.0, 0.30],
            init_quat=[1.0, 0.0, 0.0, 0.0],
            mass=self.TRAY_MASS,
        )

        # ── Blue-cap bottle (target) ─────────────────────────────────
        self._add_mesh_object(
            spec,
            asset_dir="special_bottle",
            body_name="blue_cap_bottle",
            mesh_prefix="special_bottle",
            init_pos=[0.77, 0.0, 0.35],
            init_quat=[1.0, 0.0, 0.0, 0.0],
            mass=self.BOTTLE_MASS,
        )

        # ── Plain bottle (distractor, 1 copy) ───────────────────────
        plain_asset = MeshAsset.load(self.ASSETS_DIR / "normal_bottle")
        plain_asset.add_meshes_to_spec(spec, "normal_bottle")
        if plain_asset.has_textures:
            plain_asset.add_visual_assets_to_spec(spec, "normal_bottle")

        for i in range(1):
            body = spec.worldbody.add_body(
                name=f"plain_bottle_{i}",
                pos=[0.77, 0.05 * (i + 1), 0.35],
                quat=[1.0, 0.0, 0.0, 0.0],
            )
            body.add_freejoint(name=f"plain_bottle_{i}_jnt")

            # Visual geoms: unique geom names, shared mesh/material assets
            if plain_asset.has_textures:
                plain_asset.add_visual_geoms(
                    body, f"plain_bottle_{i}", mass=self.PLAIN_MASS,
                )
            else:
                if i == 0:
                    m = spec.add_mesh(
                        name="normal_bottle_vis_mesh",
                        file=str(plain_asset.visual_mesh_path),
                    )
                    if plain_asset._scale_vec:
                        m.scale = plain_asset._scale_vec
                body.add_geom(
                    name=f"plain_bottle_{i}_visual",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname="normal_bottle_vis_mesh",
                    rgba=[0.8, 0.8, 0.8, 1],
                    mass=self.PLAIN_MASS,
                    contype=0, conaffinity=0, group=0,
                )
            # Collision geoms: unique geom names, shared collision mesh assets
            plain_asset.add_collision_geoms(body, f"plain_bottle_{i}", **ck)

        # ── Thermometer (identity quat — Z stays Z) ─────────────────
        self._add_mesh_object(
            spec,
            asset_dir="thermometer",
            body_name="thermometer",
            mesh_prefix="thermometer",
            init_pos=[0.77, -0.05, 0.35],
            init_quat=self.THERMO_QUAT,
            mass=self.THERMO_MASS,
        )

        # ── Tissue box ───────────────────────────────────────────────
        self._add_mesh_object(
            spec,
            asset_dir="tissues",
            body_name="tissue_box",
            mesh_prefix="tissues",
            init_pos=[0.77, -0.08, 0.35],
            init_quat=[1.0, 0.0, 0.0, 0.0],
            mass=self.TISSUE_MASS,
        )

    # ── Post-compile setup ───────────────────────────────────────────────

    def setup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        _id = lambda kind, name: mujoco.mj_name2id(mjm, kind, name)
        B = mujoco.mjtObj.mjOBJ_BODY
        J = mujoco.mjtObj.mjOBJ_JOINT
        G = mujoco.mjtObj.mjOBJ_GEOM

        # Body IDs
        self.rack_body_id = _id(B, "storage_rack")
        self.tray_body_id = _id(B, "tray")
        self.blue_body_id = _id(B, "blue_cap_bottle")

        # Joint addresses (rack is static — no freejoint)
        def _jnt(name):
            jid = _id(J, name)
            return mjm.jnt_qposadr[jid], mjm.jnt_dofadr[jid]

        self.tray_qadr, self.tray_dadr = _jnt("tray_jnt")
        self.blue_qadr, self.blue_dadr = _jnt("blue_cap_bottle_jnt")

        # All freejoint qpos addresses (for apply_eval_config)
        self._obj_qadrs: dict[str, int] = {
            "tray": self.tray_qadr,
            "blue_cap_bottle": self.blue_qadr,
        }
        self._obj_qadrs["plain_bottle_0"] = _jnt("plain_bottle_0_jnt")[0]
        self._obj_qadrs["thermometer"] = _jnt("thermometer_jnt")[0]
        self._obj_qadrs["tissue_box"] = _jnt("tissue_box_jnt")[0]

        # Geom sets for contact detection
        def _body_geoms(bid):
            return {gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] == bid}

        self.tray_geom_ids = _body_geoms(self.tray_body_id)
        self.blue_geom_ids = _body_geoms(self.blue_body_id)
        self.rack_geom_ids = _body_geoms(self.rack_body_id)
        self.table_geom_id = _id(G, "table_top")

        def _finger_geoms(names):
            bids = {_id(B, n) for n in names}
            return {gid for gid in range(mjm.ngeom) if mjm.geom_bodyid[gid] in bids}

        self.left_finger_geom_ids = _finger_geoms(
            ["left_lf_rot", "left_lf_down", "left_rf_rot", "left_rf_down"]
        )
        self.right_finger_geom_ids = _finger_geoms(
            ["right_lf_rot", "right_lf_down", "right_rf_rot", "right_rf_down"]
        )
        self.finger_geom_ids = self.left_finger_geom_ids | self.right_finger_geom_ids

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _yaw_quat(yaw: float) -> list[float]:
        """Quaternion for Z-axis rotation (wxyz)."""
        c, s = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
        return [c, 0.0, 0.0, s]

    @staticmethod
    def _compose_quat(q1: list[float], q2: list[float]) -> list[float]:
        """Hamilton product q1 * q2 (both wxyz)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]

    def _shelf_world_z(self, shelf_idx: int) -> float:
        """World Z of the top surface of shelf_idx when rack is at RACK_BODY_Z."""
        mesh_z = self._SHELF_MESH_Z[shelf_idx]
        return self.RACK_BODY_Z + (mesh_z + self._SHELF_BOARD_HALF_THICK) * self.RACK_SCALE

    def _local_to_world(
        self,
        local_xy: tuple[float, float],
        local_z: float,
        rack_x: float,
        rack_y: float,
        rack_yaw: float,
    ) -> list[float]:
        """Transform a rack-local (x, y, z) to world coordinates via rack yaw."""
        cos_y = np.cos(rack_yaw)
        sin_y = np.sin(rack_yaw)
        lx, ly = local_xy
        wx = rack_x + cos_y * lx - sin_y * ly
        wy = rack_y + sin_y * lx + cos_y * ly
        wz = self.RACK_BODY_Z + local_z
        return [float(wx), float(wy), float(wz)]

    # ── Config generation ────────────────────────────────────────────────

    # Approximate XY collision radii per object (half-width + margin)
    def _place_objects_on_tray(
        self,
        tray_world_pos: list[float],
        tray_yaw: float,
        tray_surface_z: float,
        rng: np.random.Generator,
    ) -> dict[str, dict]:
        """Assign objects to slots (shuffled) with jitter."""
        z_offsets = {
            "blue_cap_bottle": self.BOTTLE_Z_OFFSET,
            "plain_bottle_0": self.PLAIN_Z_OFFSET,
            "thermometer": self.THERMO_Z_OFFSET,
            "tissue_box": self.TISSUE_Z_OFFSET,
        }
        _I = [1.0, 0.0, 0.0, 0.0]

        cos_y = np.cos(tray_yaw)
        sin_y = np.sin(tray_yaw)

        slots = list(self._SLOTS)
        rng.shuffle(slots)

        out: dict[str, dict] = {}
        for name, (dx, dy) in zip(self._TRAY_OBJECTS, slots):
            jx = float(rng.uniform(-self._SLOT_JITTER, self._SLOT_JITTER))
            jy = float(rng.uniform(-self._SLOT_JITTER, self._SLOT_JITTER))
            lx, ly = dx + jx, dy + jy
            wx = tray_world_pos[0] + cos_y * lx - sin_y * ly
            wy = tray_world_pos[1] + sin_y * lx + cos_y * ly
            wz = tray_surface_z + z_offsets[name] + self._SPAWN_Z_MARGIN
            out[name] = {"pos": [float(wx), float(wy), float(wz)], "quat": _I}
        return out

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        rng = np.random.default_rng(base_seed)
        rack_pool = PoissonPool(
            polygon=np.array(self.RACK_SPAWN_POLYGON), n_target=max(n, 50), margin=0.0,
        )
        configs: list[dict] = []

        for i in range(n):
            ep_rng = np.random.default_rng(base_seed + i)
            shelf_idx = int(rng.choice(self._SPAWNABLE_SHELVES))

            # Rack position + yaw (identity base quat, apply yaw around Z only)
            rack_xy = rack_pool.pop(rng)
            rack_x, rack_y = float(rack_xy[0]), float(rack_xy[1])
            rack_yaw = float(ep_rng.uniform(
                self.RACK_BASE_YAW - self.RACK_YAW_RANGE,
                self.RACK_BASE_YAW + self.RACK_YAW_RANGE,
            ))
            rack_quat = self._yaw_quat(rack_yaw)

            # Tray on shelf — local z relative to rack body z
            shelf_mesh_z = self._SHELF_MESH_Z[shelf_idx]
            shelf_top_z = (shelf_mesh_z + self._SHELF_BOARD_HALF_THICK) * self.RACK_SCALE
            tray_local_z = shelf_top_z + self.TRAY_Z_OFFSET
            tray_y_jitter = float(ep_rng.uniform(-self.TRAY_Y_JITTER, self.TRAY_Y_JITTER))
            # Shift tray in local -Y so it protrudes from the rack opening
            tray_local_y = -self.TRAY_PROTRUDE + tray_y_jitter
            tray_world = self._local_to_world(
                (0.0, tray_local_y), tray_local_z, rack_x, rack_y, rack_yaw,
            )
            tray_quat = self._yaw_quat(rack_yaw)

            # Tray top surface in world z
            tray_surface_z = tray_world[2] + self.TRAY_SURFACE_Z_LOCAL

            # Place objects on tray
            obj_cfgs = self._place_objects_on_tray(
                tray_world, rack_yaw, tray_surface_z, ep_rng,
            )

            objects: dict[str, dict] = {
                "storage_rack": {
                    "pos": [rack_x, rack_y, self.RACK_BODY_Z],
                    "quat": [float(q) for q in rack_quat],
                },
                "tray": {
                    "pos": tray_world,
                    "quat": [float(q) for q in tray_quat],
                },
            }
            objects.update(obj_cfgs)

            configs.append({
                "seed": base_seed + i,
                "shelf_idx": shelf_idx,
                "rack_yaw": rack_yaw,
                "objects": objects,
            })
        return configs

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        shelf_idx = config["shelf_idx"]
        self._current_shelf_z = self._shelf_world_z(shelf_idx)

        # Rack is static — set body_pos/body_quat directly on mjm
        rack_cfg = config["objects"]["storage_rack"]
        mjm.body_pos[self.rack_body_id] = rack_cfg["pos"]
        mjm.body_quat[self.rack_body_id] = rack_cfg["quat"]

        # Dynamic objects via freejoint qpos
        for obj_name, qadr in self._obj_qadrs.items():
            cfg = config["objects"][obj_name]
            mjd.qpos[qadr: qadr + 3] = cfg["pos"]
            mjd.qpos[qadr + 3: qadr + 7] = cfg["quat"]

    def _zero_obj_vel(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, obj_name: str):
        jnt_name = f"{obj_name}_jnt"
        jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        dadr = mjm.jnt_dofadr[jid]
        mjd.qvel[dadr: dadr + 6] = 0.0

    def post_warmup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        """Re-place objects in order: tray first, then objects on tray.

        Each stage gets an mj_forward so that subsequent placements see
        correct world positions and collision geometry.
        """
        # 1. Place tray, zero its velocity, forward-propagate
        tray_cfg = config["objects"]["tray"]
        tray_qadr = self._obj_qadrs["tray"]
        mjd.qpos[tray_qadr: tray_qadr + 3] = tray_cfg["pos"]
        mjd.qpos[tray_qadr + 3: tray_qadr + 7] = tray_cfg["quat"]
        self._zero_obj_vel(mjm, mjd, "tray")
        mujoco.mj_forward(mjm, mjd)

        # 2. Place objects on tray, zero their velocities
        for obj_name, qadr in self._obj_qadrs.items():
            if obj_name == "tray":
                continue
            cfg = config["objects"][obj_name]
            mjd.qpos[qadr: qadr + 3] = cfg["pos"]
            mjd.qpos[qadr + 3: qadr + 7] = cfg["quat"]
            self._zero_obj_vel(mjm, mjd, obj_name)

        # 3. Short settle so objects land on tray surface, then freeze
        for _ in range(30):
            mujoco.mj_step(mjm, mjd)
        for obj_name in self._obj_qadrs:
            self._zero_obj_vel(mjm, mjd, obj_name)

    # ── Stage evaluation ─────────────────────────────────────────────────

    @property
    def stages(self) -> tuple[str, ...]:
        return ("pick_tray", "place_tray", "pick_bottle")

    def check_stages(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, bool]:
        tray_left = False
        tray_right = False
        tray_table = False
        bottle_finger = False

        for i in range(mjd.ncon):
            c = mjd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            # Tray contacts
            if g1 in self.tray_geom_ids or g2 in self.tray_geom_ids:
                other = g2 if g1 in self.tray_geom_ids else g1
                if other in self.left_finger_geom_ids:
                    tray_left = True
                if other in self.right_finger_geom_ids:
                    tray_right = True
                if other == self.table_geom_id:
                    tray_table = True

            # Blue-cap bottle contacts
            if g1 in self.blue_geom_ids or g2 in self.blue_geom_ids:
                other = g2 if g1 in self.blue_geom_ids else g1
                if other in self.finger_geom_ids:
                    bottle_finger = True

        tray_z = float(mjd.xpos[self.tray_body_id][2])
        bottle_z = float(mjd.xpos[self.blue_body_id][2])

        # Tray rest position: shelf surface + tray body offset
        tray_rest_z = self._current_shelf_z + self.TRAY_Z_OFFSET
        tray_lifted = tray_z > tray_rest_z + self.PICK_TRAY_LIFT

        tray_qvel = mjd.qvel[self.tray_dadr: self.tray_dadr + 6]
        tray_settled = float(np.linalg.norm(tray_qvel)) < self.SETTLE_VEL_THRESH

        return {
            "pick_tray": tray_lifted and tray_left and tray_right,
            "place_tray": (
                tray_table
                and tray_z < 0.04
                and not tray_left
                and not tray_right
                and tray_settled
            ),
            "pick_bottle": bottle_z > self.PICK_BOTTLE_Z and bottle_finger,
        }

    def check_success(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> bool:
        return self.check_stages(mjm, mjd)["pick_bottle"]

    # ── Gaze & CLIP ──────────────────────────────────────────────────────

    def get_gaze_targets(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, np.ndarray]:
        return {
            "tray": mjd.xpos[self.tray_body_id].copy(),
            "blue_cap_bottle": mjd.xpos[self.blue_body_id].copy(),
        }

    def get_clip_embedding(self, device: torch.device) -> torch.Tensor:
        if self._clip_embedding is not None:
            return self._clip_embedding.to(device)
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device, precision="fp16",
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        with torch.no_grad():
            tokens = tokenizer([self.prompt]).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        self._clip_embedding = emb[0].float()
        return self._clip_embedding
