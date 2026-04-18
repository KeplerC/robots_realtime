"""HangToolOnPegboardTask."""

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


class HangToolOnPegboardTask(Task):
    """Hang scissors and/or wrench on a pegboard.

    Each episode randomly has one of three variants:
      - "scissors": only scissors on table
      - "wrench":   only wrench on table
      - "both":     scissors + wrench, one after the other

    Absent tools are sunk below the table (z = -10).
    """

    ASSETS_DIR = Path(__file__).parent.parent / "assets"

    # -90° around Z: wxyz = [√2/2, 0, 0, -√2/2] — upright, pegs facing front
    PEGBOARD_QUAT = [float(np.sqrt(2) / 2), 0.0, 0.0, -float(np.sqrt(2) / 2)]
    PEGBOARD_POS  = [0.45, 0.0, 0.243]  # z so bottom sits on table

    # Both tools: OBJ thin axis = Z, so identity quat lays them flat
    TOOL_FLAT_QUAT  = [1.0, 0.0, 0.0, 0.0]
    SCISSORS_SPAWN_Z = 0.007  # half of scaled thickness: 0.0075 * 1.82 / 2
    WRENCH_SPAWN_Z   = 0.004  # half of scaled thickness: 0.045 * 0.15 / 2

    # Spawn polygon: x 0.15–0.30, y ±0.20
    # x_max=0.30 keeps tool tips (~13.5 cm radius at 1.75×) clear of pegboard at x=0.45
    TOOL_POLYGON = ((0.15, -0.20), (0.30, -0.20), (0.30, 0.20), (0.15, 0.20))

    VARIANTS = ("scissors", "wrench", "both")

    @property
    def prompt(self) -> str:
        if getattr(self, "_variant", "scissors") == "wrench":
            return "a wrench"
        return "scissors"

    def configure_scene(self, spec: mujoco.MjSpec):
        col_kwargs = dict(
            friction=[1.0, 0.05, 0.01],
            solref=[0.002, 1],
            solimp=[0.99, 0.999, 0.001, 0.5, 2],
        )

        # --- Pegboard (static) ---
        pegboard_asset = MeshAsset.load(self.ASSETS_DIR / "pegboard")
        pegboard_asset.add_meshes_to_spec(spec, "pegboard")

        pegboard_body = spec.worldbody.add_body(
            name="pegboard",
            pos=self.PEGBOARD_POS,
            quat=self.PEGBOARD_QUAT,
        )
        if pegboard_asset.has_textures:
            pegboard_asset.add_visual_assets_to_spec(spec, "pegboard")
            pegboard_asset.add_visual_geoms(pegboard_body, "pegboard", mass=1000.0)
        else:
            m = spec.add_mesh(name="pegboard_visual_mesh", file=str(pegboard_asset.visual_mesh_path))
            if pegboard_asset._scale_vec:
                m.scale = pegboard_asset._scale_vec
            pegboard_body.add_geom(
                name="pegboard_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="pegboard_visual_mesh",
                rgba=[0.7, 0.5, 0.3, 1],
                mass=1000.0,
                contype=0, conaffinity=0, group=0,
            )
        pegboard_asset.add_collision_geoms(pegboard_body, "pegboard", **col_kwargs)

        # Pegboard legs (visual only — from below the pegboard mesh down to table)
        # Pegboard mesh is centered at body origin; leave clearance for mesh thickness
        mesh_clearance = 0.05  # gap so legs don't poke into the mesh
        leg_top_z = -mesh_clearance
        leg_bottom_z = -self.PEGBOARD_POS[2]  # table surface in local frame
        leg_half_height = (leg_top_z - leg_bottom_z) / 2
        leg_center_z = (leg_top_z + leg_bottom_z) / 2
        for name, local_x in [("pegboard_leg_l", 0.10), ("pegboard_leg_r", -0.10)]:
            pegboard_body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.01, 0.012, leg_half_height],
                pos=[local_x, 0.03, leg_center_z],
                rgba=[0.55, 0.4, 0.25, 1],
                contype=0, conaffinity=0, group=0,
                mass=0.0,
            )

        # --- Scissors (dynamic) ---
        scissors_asset = MeshAsset.load(self.ASSETS_DIR / "scissors")
        scissors_asset.add_meshes_to_spec(spec, "scissors")
        scissors_body = spec.worldbody.add_body(
            name="scissors",
            pos=[0.25, -0.10, self.SCISSORS_SPAWN_Z],
            quat=self.TOOL_FLAT_QUAT,
        )
        scissors_body.add_freejoint(name="scissors_jnt")
        if scissors_asset.has_textures:
            scissors_asset.add_visual_assets_to_spec(spec, "scissors")
            scissors_asset.add_visual_geoms(scissors_body, "scissors", mass=0.3)
        else:
            m = spec.add_mesh(name="scissors_visual_mesh", file=str(scissors_asset.visual_mesh_path))
            if scissors_asset._scale_vec:
                m.scale = scissors_asset._scale_vec
            scissors_body.add_geom(
                name="scissors_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="scissors_visual_mesh",
                rgba=[0.7, 0.7, 0.7, 1],
                mass=0.3,
                contype=0, conaffinity=0, group=0,
            )
        scissors_asset.add_collision_geoms(scissors_body, "scissors", **col_kwargs)

        # --- Wrench (dynamic) ---
        wrench_col_kwargs = dict(
            friction=[2.0, 0.05, 0.01],
            solref=[0.002, 1],
            solimp=[0.99, 0.999, 0.001, 0.5, 2],
        )
        wrench_asset = MeshAsset.load(self.ASSETS_DIR / "wrench")
        wrench_asset.add_meshes_to_spec(spec, "wrench")
        wrench_body = spec.worldbody.add_body(
            name="wrench",
            pos=[0.25, 0.10, self.WRENCH_SPAWN_Z],
            quat=self.TOOL_FLAT_QUAT,
        )
        wrench_body.add_freejoint(name="wrench_jnt")
        if wrench_asset.has_textures:
            wrench_asset.add_visual_assets_to_spec(spec, "wrench")
            wrench_asset.add_visual_geoms(wrench_body, "wrench", mass=0.2)
        else:
            m = spec.add_mesh(name="wrench_visual_mesh", file=str(wrench_asset.visual_mesh_path))
            if wrench_asset._scale_vec:
                m.scale = wrench_asset._scale_vec
            wrench_body.add_geom(
                name="wrench_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="wrench_visual_mesh",
                rgba=[0.5, 0.5, 0.5, 1],
                mass=0.2,
                contype=0, conaffinity=0, group=0,
            )
        wrench_asset.add_collision_geoms(wrench_body, "wrench", **wrench_col_kwargs)

    def setup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        self.pegboard_body_id  = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "pegboard")
        self.scissors_body_id  = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "scissors")
        self.wrench_body_id    = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "wrench")
        scissors_jnt_id        = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "scissors_jnt")
        wrench_jnt_id          = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "wrench_jnt")
        self.scissors_jnt_adr  = mjm.jnt_qposadr[scissors_jnt_id]
        self.wrench_jnt_adr    = mjm.jnt_qposadr[wrench_jnt_id]
        self.scissors_dof_adr  = mjm.jnt_dofadr[scissors_jnt_id]
        self.wrench_dof_adr    = mjm.jnt_dofadr[wrench_jnt_id]
        self._variant: str = "scissors"
        self._pool_single: PoissonPool | None = None
        self._pool_pair:   PoissonPool | None = None

    @staticmethod
    def _yaw_quat(yaw: float) -> list[float]:
        """Flat quat with yaw around Z (wxyz)."""
        c, s = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
        return [c, 0.0, 0.0, s]

    def _set_tool(self, mjd: mujoco.MjData, jnt_adr: int, xy: np.ndarray,
                  spawn_z: float, quat: list[float], dof_adr: int | None = None):
        mjd.qpos[jnt_adr:jnt_adr+3]   = [float(xy[0]), float(xy[1]), spawn_z]
        mjd.qpos[jnt_adr+3:jnt_adr+7] = quat
        if dof_adr is not None:
            mjd.qvel[dof_adr:dof_adr+6] = 0.0

    def _sink_tool(self, mjd: mujoco.MjData, jnt_adr: int, dof_adr: int | None = None):
        mjd.qpos[jnt_adr:jnt_adr+3]   = [0.0, 0.0, -10.0]
        mjd.qpos[jnt_adr+3:jnt_adr+7] = [1.0, 0.0, 0.0, 0.0]
        if dof_adr is not None:
            mjd.qvel[dof_adr:dof_adr+6] = 0.0

    def randomize(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, rng: np.random.Generator):
        self._variant = str(rng.choice(self.VARIANTS))
        yaw_s = float(rng.uniform(0, np.pi))
        yaw_w = float(rng.uniform(0, np.pi))

        if self._variant == "both":
            # n_target=2 gives ~24 cm min spacing — sufficient to prevent overlap
            if self._pool_pair is None:
                self._pool_pair = PoissonPool(np.array(self.TOOL_POLYGON), n_target=2, margin=0.05)
            pos_s = self._pool_pair.pop(rng)
            pos_w = self._pool_pair.pop(rng)
            self._set_tool(mjd, self.scissors_jnt_adr, pos_s, self.SCISSORS_SPAWN_Z, self._yaw_quat(yaw_s), dof_adr=self.scissors_dof_adr)
            self._set_tool(mjd, self.wrench_jnt_adr,   pos_w, self.WRENCH_SPAWN_Z,   self._yaw_quat(yaw_w), dof_adr=self.wrench_dof_adr)
        else:
            if self._pool_single is None:
                self._pool_single = PoissonPool(np.array(self.TOOL_POLYGON), n_target=50, margin=0.0)
            pos = self._pool_single.pop(rng)
            if self._variant == "scissors":
                self._set_tool(mjd, self.scissors_jnt_adr, pos, self.SCISSORS_SPAWN_Z, self._yaw_quat(yaw_s), dof_adr=self.scissors_dof_adr)
                self._sink_tool(mjd, self.wrench_jnt_adr, dof_adr=self.wrench_dof_adr)
            else:
                self._set_tool(mjd, self.wrench_jnt_adr, pos, self.WRENCH_SPAWN_Z, self._yaw_quat(yaw_w), dof_adr=self.wrench_dof_adr)
                self._sink_tool(mjd, self.scissors_jnt_adr, dof_adr=self.scissors_dof_adr)

    def generate_eval_configs(self, n: int, base_seed: int) -> list[dict]:
        rng     = np.random.default_rng(base_seed)
        pool_s  = PoissonPool(np.array(self.TOOL_POLYGON), n_target=n, margin=0.0)
        pool_p  = PoissonPool(np.array(self.TOOL_POLYGON), n_target=2,  margin=0.05)
        configs = []
        for i in range(n):
            variant = self.VARIANTS[i % len(self.VARIANTS)]
            yaw_s   = float(rng.uniform(0, np.pi))
            yaw_w   = float(rng.uniform(0, np.pi))
            obj: dict = {}
            if variant == "both":
                pos_s = pool_p.pop(rng)
                pos_w = pool_p.pop(rng)
                obj["scissors"] = {"pos": [float(pos_s[0]), float(pos_s[1]), self.SCISSORS_SPAWN_Z],
                                   "quat": self._yaw_quat(yaw_s)}
                obj["wrench"]   = {"pos": [float(pos_w[0]), float(pos_w[1]), self.WRENCH_SPAWN_Z],
                                   "quat": self._yaw_quat(yaw_w)}
            elif variant == "scissors":
                pos = pool_s.pop(rng)
                obj["scissors"] = {"pos": [float(pos[0]), float(pos[1]), self.SCISSORS_SPAWN_Z],
                                   "quat": self._yaw_quat(yaw_s)}
            else:
                pos = pool_s.pop(rng)
                obj["wrench"] = {"pos": [float(pos[0]), float(pos[1]), self.WRENCH_SPAWN_Z],
                                 "quat": self._yaw_quat(yaw_w)}
            configs.append({"seed": base_seed + i, "variant": variant, "objects": obj})
        return configs

    def apply_eval_config(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, config: dict):
        self._variant = config["variant"]
        obj = config["objects"]
        if "scissors" in obj:
            cfg = obj["scissors"]
            self._set_tool(mjd, self.scissors_jnt_adr,
                           np.array(cfg["pos"][:2]), cfg["pos"][2], cfg["quat"],
                           dof_adr=self.scissors_dof_adr)
        else:
            self._sink_tool(mjd, self.scissors_jnt_adr, dof_adr=self.scissors_dof_adr)
        if "wrench" in obj:
            cfg = obj["wrench"]
            self._set_tool(mjd, self.wrench_jnt_adr,
                           np.array(cfg["pos"][:2]), cfg["pos"][2], cfg["quat"],
                           dof_adr=self.wrench_dof_adr)
        else:
            self._sink_tool(mjd, self.wrench_jnt_adr, dof_adr=self.wrench_dof_adr)

    @property
    def stages(self) -> tuple[str, ...]:
        return ("pick_scissors", "hang_scissors", "pick_wrench", "hang_wrench")

    def check_stages(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> dict[str, bool]:
        result: dict[str, bool] = {s: False for s in self.stages}
        has_scissors = self._variant in ("scissors", "both")
        has_wrench   = self._variant in ("wrench",   "both")

        if has_scissors:
            spos = mjd.xpos[self.scissors_body_id]
            result["pick_scissors"] = bool(spos[2] > 0.05 + self.z_offset)
            near = bool(abs(spos[0] - self.PEGBOARD_POS[0]) < 0.15)
            result["hang_scissors"] = near and bool(spos[2] > 0.10 + self.z_offset)

        if has_wrench:
            wpos = mjd.xpos[self.wrench_body_id]
            result["pick_wrench"] = bool(wpos[2] > 0.05 + self.z_offset)
            near = bool(abs(wpos[0] - self.PEGBOARD_POS[0]) < 0.15)
            result["hang_wrench"] = near and bool(wpos[2] > 0.10 + self.z_offset)

        return result

    def check_success(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> bool:
        stages = self.check_stages(mjm, mjd)
        has_scissors = self._variant in ("scissors", "both")
        has_wrench   = self._variant in ("wrench",   "both")
        ok = True
        if has_scissors:
            ok = ok and stages["hang_scissors"]
        if has_wrench:
            ok = ok and stages["hang_wrench"]
        return ok

    def get_clip_embedding(self, device: torch.device) -> torch.Tensor:
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
        return embedding[0].float()
