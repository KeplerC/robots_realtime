"""YAM vOS bridge agent — sends sim observations to verifiable-OS, applies actions.

Mirrors FrankaOscClientCartesianAgent but for the 6-DOF YAM arm.
Connects as a msgpack TCP client to vOS's MsgpackNumpyServer and relays
observations from the YAM simulator (or real hardware), receiving joint-position
actions in return.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import viser
import viser.extras
import yourdfpy
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent
from robots_realtime.robots.inverse_kinematics.yam_pyroki import YamPyroki
from robots_realtime.utils.server_client_utils import SyncMsgpackNumpyClient


def _g(d: Dict[Any, Any], key: str) -> Any:
    """Get from dict trying both string and byte-string keys."""
    return d.get(key) if key in d else d.get(key.encode())


def _load_yam_urdf() -> yourdfpy.URDF:
    """Load a fresh YAM URDF instance (avoids deepcopy issues with dict_keys)."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(
        current_path, "..", "..", "..", "dependencies", "i2rt", "i2rt",
        "robot_models", "arm", "yam", "yam.urdf",
    )
    mesh_dir = os.path.join(
        current_path, "..", "..", "..", "dependencies", "i2rt", "i2rt",
        "robot_models", "arm", "yam", "assets",
    )
    return yourdfpy.URDF.load(
        urdf_path,
        mesh_dir=mesh_dir,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )


class YamVosClientAgent(Agent):
    """Bridge agent connecting YAM sim/hardware to verifiable-OS.

    Receives observations from ViserTeleopNode (via ZMQ), forwards them to vOS
    over the msgpack TCP protocol, and returns the joint-position actions that
    vOS computes.  Also hosts a Viser viewer showing both the real/sim arm state
    and the vOS-commanded ghost overlay.
    """

    def __init__(
        self,
        *,
        bimanual: bool = False,
        viser_port: int = 8080,
        client_host: str = "0.0.0.0",
        client_port: int = 9000,
    ) -> None:
        self.bimanual = bimanual

        # vOS commanded state (for visualization overlay)
        self.vos_joint_pos: Optional[np.ndarray] = None
        self.vos_gripper_pos: Optional[np.ndarray] = None

        # IK solver for Viser visualization
        self.viser_server = viser.ViserServer(port=viser_port)
        self.ik = YamPyroki(
            viser_server=self.viser_server,
            bimanual=bimanual,
        )
        self.ik_thread = threading.Thread(target=self.ik.run, name="yam_pyroki_ik")
        self.ik_thread.daemon = True
        self.ik_thread.start()

        # Msgpack client to vOS
        self.vos_client = SyncMsgpackNumpyClient(host=client_host, port=client_port)

        self.obs: Optional[Dict[str, Any]] = None
        self._update_period = 0.05
        self._last_reset_request_seq_seen = 0
        self._last_reset_ack_seq_seen = 0

        self._setup_visualization()

        self.vis_thread = threading.Thread(target=self._update_visualization, name="yam_vos_vis")
        self.vis_thread.daemon = True
        self.vis_thread.start()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _setup_visualization(self) -> None:
        """Set up Viser overlays for live robot state and vOS ghost."""
        self.base_frame_real = self.viser_server.scene.add_frame("/yam_real", show_axes=False)
        self.urdf_vis_real = viser.extras.ViserUrdf(
            self.viser_server,
            _load_yam_urdf(),
            root_node_name="/yam_real",
            mesh_color_override=(0.55, 0.75, 0.95),
        )
        for mesh in self.urdf_vis_real._meshes:
            mesh.opacity = 0.3

        self.base_frame_vos = self.viser_server.scene.add_frame("/yam_vos", show_axes=False)
        self.urdf_vis_vos = viser.extras.ViserUrdf(
            self.viser_server,
            _load_yam_urdf(),
            root_node_name="/yam_vos",
            mesh_color_override=(0.55, 0.35, 0.95),
        )
        for mesh in self.urdf_vis_vos._meshes:
            mesh.opacity = 0.3

        self.left_gripper_slider = self.viser_server.gui.add_slider(
            label="Gripper Width", min=0.0, max=1.0, step=0.01, initial_value=1.0
        )

        self.viser_cam_img_handles: Dict[str, viser.GuiImageHandle] = {}

    def _update_visualization(self) -> None:
        """Continuously sync live robot state and vOS ghost into Viser."""
        while self.obs is None:
            time.sleep(0.025)

        while True:
            obs_copy = self.obs
            if obs_copy is None:
                time.sleep(self._update_period)
                continue

            # Update real robot visualization (URDF expects 6 joints, not 7 with gripper)
            left_obs = obs_copy.get("left")
            if isinstance(left_obs, dict):
                joint_pos = left_obs.get("joint_pos")
                if joint_pos is not None:
                    self.urdf_vis_real.update_cfg(np.asarray(joint_pos)[:6])

            # Update vOS ghost visualization (URDF expects 6 joints only)
            if self.vos_joint_pos is not None:
                self.urdf_vis_vos.update_cfg(self.vos_joint_pos[:6])

            time.sleep(self._update_period)

    def _sync_fallback_state_from_obs(self) -> None:
        """Reset local fallback / visualization state to the latest observed robot config."""
        if self.obs is None:
            return
        for arm_key in ("left", "right"):
            arm_obs = self.obs.get(arm_key)
            if not isinstance(arm_obs, dict):
                continue
            joint_pos = arm_obs.get("joint_pos")
            if joint_pos is None:
                continue
            joint_pos = np.asarray(joint_pos, dtype=np.float32)
            if arm_key in self.ik.joints and joint_pos.shape[0] >= 6:
                self.ik.joints[arm_key] = joint_pos[:6].copy()
            if arm_key == "left" and joint_pos.shape[0] >= 7:
                # Best-effort resync of the teleop gripper UI to the reset state.
                self.left_gripper_slider.value = float(np.clip(joint_pos[6] / 0.0475, 0.0, 1.0))
        self.vos_joint_pos = None
        self.vos_gripper_pos = None

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._last_reset_request_seq_seen = 0
        self._last_reset_ack_seq_seen = 0

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.obs = dict(obs)

        # Enrich camera observations with pose/pose_mat if available
        for cam_key, cam_obs in self.obs.items():
            if isinstance(cam_obs, dict):
                extrinsics = cam_obs.get("extrinsics")
                if extrinsics is not None:
                    cam_obs["pose"] = np.concatenate([extrinsics["position"], extrinsics["wxyz"]])
                    cam_obs["pose_mat"] = extrinsics["pose_mat"]

        # Ack resets only after the sim node has applied them and published fresh state.
        observed_reset_ack = None
        for arm_key in ("left", "right"):
            arm_obs = self.obs.get(arm_key)
            if isinstance(arm_obs, dict):
                ack_seq = arm_obs.get("reset_ack_seq")
                if ack_seq is not None:
                    try:
                        ack_seq = int(ack_seq)
                    except (TypeError, ValueError):
                        ack_seq = None
                if ack_seq is not None:
                    observed_reset_ack = max(observed_reset_ack or ack_seq, ack_seq)
        if (
            observed_reset_ack is not None
            and observed_reset_ack > self._last_reset_ack_seq_seen
        ):
            self._last_reset_ack_seq_seen = observed_reset_ack
            self._sync_fallback_state_from_obs()
            print(f"[YamVosClient] observed reset applied seq={observed_reset_ack}")
        if self._last_reset_ack_seq_seen > 0:
            self.obs["vos_control"] = {"reset_ack_seq": self._last_reset_ack_seq_seen}

        # Send observation to vOS, receive action
        # DEBUG: print obs structure on first call
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"[YamVosClient] obs keys: {list(self.obs.keys())}")
            for k, v in self.obs.items():
                if isinstance(v, dict):
                    print(f"  [{k}] keys: {list(v.keys())}")
                elif isinstance(v, np.ndarray):
                    print(f"  [{k}] ndarray shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"  [{k}] type={type(v).__name__}")
        try:
            response = self.vos_client.send_request(self.obs)
        except Exception as e:
            print(f"[YamVosClient] send_request failed (falling back to IK): {e}")
            response = {}

        # Process per-arm responses
        action: Dict[str, Any] = {}
        for arm_key in ["left", "right"]:
            arm_resp = _g(response, arm_key) if response else None
            if arm_resp is not None:
                vos_jp = np.asarray(_g(arm_resp, "joint_pos"), dtype=np.float32)
                vos_grip = float(np.asarray(_g(arm_resp, "gripper"), dtype=np.float32))
                # vos_grip is a 0-1 fraction; the sim env multiplies by
                # _GRIPPER_CTRL_MAX internally, so pass the fraction directly.
                action[arm_key] = {"pos": np.concatenate([vos_jp, [vos_grip]])}

                # Store for visualization
                if arm_key == "left":
                    self.vos_joint_pos = vos_jp
                    self.vos_gripper_pos = np.float32(vos_grip)

        control = _g(response, "vos_control") if response else None
        reset_seq = _g(control, "reset_seq") if isinstance(control, dict) else None
        if reset_seq is not None:
            try:
                reset_seq = int(reset_seq)
            except (TypeError, ValueError):
                reset_seq = None
        if reset_seq is not None and reset_seq > self._last_reset_request_seq_seen:
            self._last_reset_request_seq_seen = reset_seq
            print(f"[YamVosClient] received remote reset request seq={reset_seq}")
            action["reset_env"] = True
            action["reset_env_seq"] = reset_seq

        # Fallback: if no vOS response for an arm, use IK gizmo (left only)
        if "left" not in action:
            left_joints = np.asarray(self.ik.joints["left"], dtype=np.float32)
            gripper_val = self.left_gripper_slider.value * 0.0475
            action["left"] = {"pos": np.concatenate([left_joints, [gripper_val]])}
        if "right" not in action:
            # Hold right arm at current position (read from obs)
            right_obs = self.obs.get("right")
            if right_obs and "joint_pos" in right_obs:
                action["right"] = {"pos": np.asarray(right_obs["joint_pos"], dtype=np.float32)}

        return action

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        return {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
            "right": {"pos": Array(shape=(7,), dtype=np.float32)},
        }
