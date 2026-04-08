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
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import viser
import viser.extras
import yourdfpy
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent
from robots_realtime.robots.inverse_kinematics.yam_pyroki import YamPyroki
from robots_realtime.utils.server_client_utils import SyncMsgpackNumpyClient


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

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        self.obs = deepcopy(obs)

        # Enrich camera observations with pose/pose_mat if available
        for cam_key, cam_obs in self.obs.items():
            if isinstance(cam_obs, dict):
                extrinsics = cam_obs.get("extrinsics")
                if extrinsics is not None:
                    cam_obs["pose"] = np.concatenate([extrinsics["position"], extrinsics["wxyz"]])
                    cam_obs["pose_mat"] = extrinsics["pose_mat"]

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
            print(f"[YamVosClient] send_request failed: {e}")
            import traceback; traceback.print_exc()
            raise

        if response.get(b"left") is not None:
            self.vos_joint_pos = np.asarray(response[b"left"].get(b"joint_pos"), dtype=np.float32)
            self.vos_gripper_pos = np.asarray(response[b"left"].get(b"gripper"), dtype=np.float32)

        # Default: use IK targets from Viser gizmo (6 joints) + gripper
        left_joints = np.asarray(self.ik.joints["left"], dtype=np.float32)
        gripper_val = self.left_gripper_slider.value * 0.0475  # fraction → meters
        left_target = np.concatenate([left_joints, [gripper_val]])
        action: Dict[str, Dict[str, np.ndarray]] = {"left": {"pos": left_target}}

        # Override with vOS response if available
        if response.get(b"left") is not None:
            gripper_act = float(self.vos_gripper_pos) * 0.0475  # fraction → meters
            action = {
                "left": {"pos": np.concatenate([self.vos_joint_pos, [gripper_act]])}
            }

        return action

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        return {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},  # 6 joints + 1 gripper
        }
