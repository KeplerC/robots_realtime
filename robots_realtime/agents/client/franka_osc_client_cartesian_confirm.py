"""Teleoperation agent with pose visualization and manual confirmation before execution."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent
from robots_realtime.robots.inverse_kinematics.franka_pyroki import FrankaPyroki
from robots_realtime.sensors.cameras.camera_utils import obs_get_rgb, resize_with_center_crop
from robots_realtime.utils.depth_utils import depth_color_to_pointcloud
from robots_realtime.utils.server_client_utils import SyncMsgpackNumpyClient


class FrankaOscClientCartesianConfirmAgent(Agent):
    """Interactive teleoperation agent for Franka OSC robots with manual confirmation.

    This agent extends the standard client agent by adding:
    - Visualization of desired vs current poses
    - Manual confirmation button before executing motions
    - Separate URDF visualizations for current and target states
    """

    def __init__(
        self,
        *,
        bimanual: bool = False,
        right_arm_extrinsic: Optional[Dict[str, Any]] = None,
        robot_description: Optional[str] = None,
        ik_rate: float = 100.0,
        visualize_rgbd: bool = False,
        robotiq_gripper: bool = False,
        viser_port: int = 8080,
        client_host: str = "0.0.0.0",
        client_port: int = 9000,
        debug_msgpack: bool = False,
        hold_on_empty_response: bool = False,
    ) -> None:
        self.bimanual = bimanual
        self.robotiq_gripper = robotiq_gripper
        self.right_arm_extrinsic = right_arm_extrinsic
        self.visualize_rgbd = visualize_rgbd
        self.debug_msgpack = debug_msgpack
        self.hold_on_empty_response = hold_on_empty_response
        if self.bimanual:
            assert right_arm_extrinsic is not None, (
                "right_arm_extrinsic must be provided for bimanual Franka configuration"
            )

        # Pending command state
        self.pending_joint_pos: Optional[np.ndarray] = None
        self.pending_gripper_pos: Optional[float] = None
        self.command_confirmed = threading.Event()
        self.command_lock = threading.Lock()

        # Executed command state
        self.hyrl_joint_pos = None
        self.hyrl_gripper_pos = None

        # Hold position while waiting for confirmation
        self.hold_joint_pos: Optional[np.ndarray] = None
        self.safe_hold_joint_pos: Optional[np.ndarray] = None

        # Auto-approve trajectory mode
        self.auto_approve_trajectory = False
        self.trajectory_waypoint_count = 0

        # Full trajectory commands from GaP. These are distinct from the
        # historical waypoint-stream auto-approve mode above: a full trajectory
        # is approved once, then this agent owns waypoint advancement.
        self.pending_trajectory: Optional[np.ndarray] = None
        self.pending_trajectory_id: Optional[str] = None
        self.pending_trajectory_tolerance: float = 0.05
        self.pending_trajectory_gripper_pos: Optional[float] = None
        self.active_trajectory: Optional[np.ndarray] = None
        self.active_trajectory_id: Optional[str] = None
        self.active_trajectory_index: int = 0
        self.active_trajectory_tolerance: float = 0.05
        self.active_trajectory_gripper_pos: Optional[float] = None
        self.active_trajectory_error: Optional[float] = None
        self.active_trajectory_times: Optional[np.ndarray] = None
        self.active_trajectory_start_time: Optional[float] = None
        self.active_trajectory_duration: float = 0.0
        self.active_trajectory_completion_tolerance: float = 0.12
        self.active_trajectory_settle_timeout_s: float = 5.0
        self.completed_trajectory_id: Optional[str] = None
        self.completed_trajectory_num_waypoints: int = 0
        self.trajectory_status: str = "idle"
        self.trajectory_joint_speed_limit: float = 0.35
        self.trajectory_min_segment_dt: float = 0.12
        self.trajectory_min_duration: float = 1.0
        self._trajectory_visualized_id: Optional[str] = None

        self.viser_server = viser.ViserServer(port=viser_port)
        self.ik = FrankaPyroki(
            rate=ik_rate,
            viser_server=self.viser_server,
            bimanual=bimanual,
            robot_description=robot_description,
        )
        self.ik_thread = threading.Thread(target=self.ik.run, name="franka_pyroki_ik")
        self.ik_thread.daemon = True
        self.ik_thread.start()

        if self.robotiq_gripper:
            self.ik.transform_handles.get("left").tcp_offset_frame.wxyz = vtf.SO3.from_rpy_radians(
                0.0, 0.0, np.pi / 4
            ).wxyz
            self.ik.transform_handles.get("left").tcp_offset_frame.position = (0.0, 0.0, -0.157)

        self.franka_client = SyncMsgpackNumpyClient(host=client_host, port=client_port)

        self.obs: Optional[Dict[str, Any]] = None
        self._update_period = 0.05
        self._setup_visualization()

        self.real_vis_thread = threading.Thread(target=self._update_visualization, name="franka_real_vis")
        self.real_vis_thread.daemon = True
        self.real_vis_thread.start()

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _extract_joint_pos(self, obs: Dict[str, Any], arm: str) -> Optional[np.ndarray]:
        """Best-effort extraction of joint positions for the requested arm from an observation."""

        arm_obs = obs.get(arm)
        if isinstance(arm_obs, dict):
            joint_pos = arm_obs.get("joint_pos")
            if joint_pos is not None:
                return np.asarray(joint_pos)

        if arm == "left" and obs.get("joint_pos") is not None:
            return np.asarray(obs["joint_pos"])

        return None

    @staticmethod
    def _wire_get(mapping: Dict[Any, Any], key: str) -> Any:
        """Get a msgpack field that may have string or byte-string keys."""
        if not isinstance(mapping, dict):
            return None
        if key in mapping:
            return mapping[key]
        return mapping.get(key.encode())

    @staticmethod
    def _wire_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _trajectory_status_payload(self) -> Dict[str, Any]:
        """Status sent back to GaP through the msgpack observation request."""
        command_id = (
            self.active_trajectory_id
            or self.pending_trajectory_id
            or self.completed_trajectory_id
            or ""
        )
        num_waypoints = 0
        if self.active_trajectory is not None:
            num_waypoints = int(len(self.active_trajectory))
        elif self.pending_trajectory is not None:
            num_waypoints = int(len(self.pending_trajectory))
        elif self.completed_trajectory_id is not None:
            num_waypoints = int(self.completed_trajectory_num_waypoints)

        return {
            "command_id": command_id,
            "status": self.trajectory_status,
            "active_waypoint": int(self.active_trajectory_index),
            "num_waypoints": num_waypoints,
            "error": float(self.active_trajectory_error)
            if self.active_trajectory_error is not None
            else float("nan"),
        }

    def _visualize_trajectory(self, trajectory: np.ndarray, command_id: str) -> None:
        """Show the planned trajectory as waypoint markers in Viser."""
        if self._trajectory_visualized_id == command_id:
            return
        self._trajectory_visualized_id = command_id

        try:
            urdf = deepcopy(self.ik.urdf)
            joint_names = list(urdf.actuated_joint_names[:7])
            points = []
            stride = max(1, len(trajectory) // 80)
            for waypoint in trajectory[::stride]:
                urdf.update_cfg(dict(zip(joint_names, waypoint[: len(joint_names)], strict=False)))
                target_tf = vtf.SE3.from_matrix(
                    urdf.get_transform(self.ik.target_link_names[0], "panda_link0")
                )
                points.append(target_tf.translation())
            if points:
                pts = np.asarray(points, dtype=np.float32)
                colors = np.tile(np.asarray([[255, 90, 80]], dtype=np.uint8), (len(pts), 1))
                self.viser_server.scene.add_point_cloud(
                    name="/gap_trajectory/waypoints",
                    points=pts,
                    colors=colors,
                    point_size=0.012,
                )
        except Exception as exc:
            print(f"[TRAJECTORY VIS] Failed to draw waypoint markers: {exc}")

    def _trajectory_times(self, trajectory: np.ndarray) -> np.ndarray:
        """Create a conservative time base for smooth joint-space playback."""
        if len(trajectory) <= 1:
            return np.zeros((len(trajectory),), dtype=np.float32)

        deltas = np.abs(np.diff(trajectory[:, :7], axis=0))
        speed_limit = max(float(self.trajectory_joint_speed_limit), 1e-3)
        segment_dt = np.max(deltas / speed_limit, axis=1)
        segment_dt = np.maximum(segment_dt, float(self.trajectory_min_segment_dt))

        total = float(np.sum(segment_dt))
        min_duration = max(
            float(self.trajectory_min_duration),
            float(self.trajectory_min_segment_dt) * (len(trajectory) - 1),
        )
        if total < min_duration:
            segment_dt *= min_duration / max(total, 1e-6)

        return np.concatenate([[0.0], np.cumsum(segment_dt)]).astype(np.float32)

    def _current_hold_action(self, obs: Dict[str, Any], reason: str) -> Optional[np.ndarray]:
        """Return an action that holds one latched measured robot state."""
        current_left = self._extract_joint_pos(obs, "left")
        if self.safe_hold_joint_pos is None and (current_left is None or len(current_left) < 7):
            print(f"[WAITING] {reason}; no current joints, publishing no command")
            return None

        if self.safe_hold_joint_pos is None:
            current_left = np.asarray(current_left, dtype=np.float32)
            if len(current_left) >= 8:
                self.safe_hold_joint_pos = current_left.copy()
                if not np.isfinite(self.safe_hold_joint_pos[-1]):
                    self.safe_hold_joint_pos[-1] = self.left_gripper_slider_handle.value
            else:
                self.safe_hold_joint_pos = np.concatenate(
                    [current_left[:7], [self.left_gripper_slider_handle.value]]
                ).astype(np.float32)

        left_target = self.safe_hold_joint_pos.copy()

        if not hasattr(self, "_hold_log_count"):
            self._hold_log_count = 0
        self._hold_log_count += 1
        if self._hold_log_count <= 5 or self._hold_log_count % 100 == 0:
            print(f"[HOLD] {reason}: holding current joints {left_target[:3]}")
        return left_target

    def _activate_pending_trajectory_locked(self) -> None:
        """Move the pending full trajectory into active execution."""
        if self.pending_trajectory is None:
            return

        self.active_trajectory = self.pending_trajectory.copy()
        self.active_trajectory_id = self.pending_trajectory_id
        self.active_trajectory_index = 0
        self.active_trajectory_tolerance = self.pending_trajectory_tolerance
        self.active_trajectory_completion_tolerance = max(
            float(self.pending_trajectory_tolerance),
            0.12,
        )
        self.active_trajectory_gripper_pos = self.pending_trajectory_gripper_pos
        self.active_trajectory_error = None
        self.active_trajectory_times = self._trajectory_times(self.active_trajectory)
        self.active_trajectory_duration = (
            float(self.active_trajectory_times[-1])
            if len(self.active_trajectory_times) > 0
            else 0.0
        )
        self.active_trajectory_start_time = time.monotonic()
        self.completed_trajectory_id = None
        self.completed_trajectory_num_waypoints = 0
        self.pending_trajectory = None
        self.pending_trajectory_id = None
        self.pending_trajectory_gripper_pos = None
        self.pending_joint_pos = None
        self.pending_gripper_pos = None
        self.hold_joint_pos = None
        self.safe_hold_joint_pos = None
        self.hyrl_joint_pos = None
        self.hyrl_gripper_pos = None
        self.trajectory_status = "executing"
        self.status_text.value = "✅ Trajectory approved - executing..."

    def _finish_active_trajectory_locked(
        self,
        *,
        status: str,
        target: np.ndarray,
        message: str,
    ) -> np.ndarray:
        """Mark the active trajectory terminal and let later cycles hold measured state."""
        self.trajectory_status = status
        self.completed_trajectory_id = self.active_trajectory_id
        self.completed_trajectory_num_waypoints = (
            int(len(self.active_trajectory))
            if self.active_trajectory is not None
            else int(self.completed_trajectory_num_waypoints)
        )
        self.hyrl_joint_pos = None
        self.hyrl_gripper_pos = None
        self.safe_hold_joint_pos = None
        self.active_trajectory = None
        self.active_trajectory_id = None
        self.active_trajectory_times = None
        self.active_trajectory_start_time = None
        self.active_trajectory_duration = 0.0
        self.status_text.value = message
        return target

    def _active_trajectory_target_locked(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Return a time-interpolated target for the active full trajectory."""
        if self.active_trajectory is None or len(self.active_trajectory) == 0:
            return None

        trajectory = self.active_trajectory
        times = self.active_trajectory_times
        if times is None or len(times) != len(trajectory):
            times = self._trajectory_times(trajectory)
            self.active_trajectory_times = times
            self.active_trajectory_duration = float(times[-1]) if len(times) > 0 else 0.0
        if self.active_trajectory_start_time is None:
            self.active_trajectory_start_time = time.monotonic()

        elapsed = max(0.0, time.monotonic() - self.active_trajectory_start_time)
        duration = max(float(self.active_trajectory_duration), 0.0)
        final_target = trajectory[-1]

        current_left = self._extract_joint_pos(obs, "left")
        current: Optional[np.ndarray] = None
        if current_left is not None and len(current_left) >= 7:
            current = np.asarray(current_left[:7], dtype=np.float32)

        if len(trajectory) == 1 or elapsed >= duration:
            target = final_target
            self.active_trajectory_index = len(trajectory) - 1
            if current is not None:
                final_error = float(np.linalg.norm(current - final_target))
                self.active_trajectory_error = final_error
                if final_error <= self.active_trajectory_completion_tolerance:
                    return self._finish_active_trajectory_locked(
                        status="succeeded",
                        target=target,
                        message="✅ Trajectory complete",
                    )
                if elapsed >= duration + self.active_trajectory_settle_timeout_s:
                    return self._finish_active_trajectory_locked(
                        status="failed",
                        target=target,
                        message=(
                            "⚠️ Trajectory did not settle "
                            f"(error {final_error:.3f} rad)"
                        ),
                    )
            else:
                self.active_trajectory_error = None
        else:
            upper = int(np.searchsorted(times, elapsed, side="right"))
            lower = max(0, min(upper - 1, len(trajectory) - 2))
            upper = lower + 1
            t0 = float(times[lower])
            t1 = float(times[upper])
            alpha = 0.0 if t1 <= t0 else (elapsed - t0) / (t1 - t0)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            target = ((1.0 - alpha) * trajectory[lower] + alpha * trajectory[upper]).astype(np.float32)
            self.active_trajectory_index = lower
            if current is not None:
                self.active_trajectory_error = float(np.linalg.norm(current - target))

        self.hyrl_joint_pos = target.copy()
        self.hyrl_gripper_pos = self.active_trajectory_gripper_pos
        self.trajectory_status = "executing"
        self.status_text.value = (
            f"🚀 Executing trajectory "
            f"{self.active_trajectory_index + 1}/{len(trajectory)}"
        )
        return target

    def _on_confirm_click(self) -> None:
        """Callback when the single waypoint confirmation button is clicked."""
        import sys
        print(f"[BUTTON CLICK] Confirm single waypoint clicked!", file=sys.stderr, flush=True)
        with self.command_lock:
            if self.pending_joint_pos is not None:
                print(f"[CONFIRM] Executing single waypoint")
                self.hyrl_joint_pos = self.pending_joint_pos.copy()
                self.hyrl_gripper_pos = self.pending_gripper_pos
                self.pending_joint_pos = None
                self.pending_gripper_pos = None
                self.hold_joint_pos = None
                self.command_confirmed.clear()
                self.confirm_button.disabled = True
                self.status_text.value = "✅ Motion confirmed - executing..."
                print(f"[CONFIRM] pending waypoint promoted to hyrl", file=sys.stderr, flush=True)
            elif self.pending_trajectory is not None:
                self.status_text.value = "⚠️ Use Approve Entire Trajectory for trajectory commands"
            else:
                print(f"[BUTTON CLICK] No pending motion!", file=sys.stderr, flush=True)
                self.status_text.value = "⚠️ No pending motion to confirm"

    def _on_confirm_trajectory_click(self) -> None:
        """Callback when the entire trajectory confirmation button is clicked."""
        import sys
        print(f"[BUTTON CLICK] Approve entire trajectory clicked!", file=sys.stderr, flush=True)
        with self.command_lock:
            if self.pending_trajectory is not None:
                print(f"[CONFIRM TRAJECTORY] Executing full trajectory")
                self._activate_pending_trajectory_locked()
                self.confirm_button.disabled = True
                self.confirm_trajectory_button.disabled = True
                print(f"[CONFIRM TRAJECTORY] Full trajectory activated!", file=sys.stderr, flush=True)
            elif self.pending_joint_pos is not None:
                print(f"[CONFIRM TRAJECTORY] Auto-approving all subsequent waypoints")
                self.auto_approve_trajectory = True
                self.trajectory_waypoint_count = 1  # First waypoint
                self.hyrl_joint_pos = self.pending_joint_pos.copy()
                self.hyrl_gripper_pos = self.pending_gripper_pos
                self.pending_joint_pos = None
                self.pending_gripper_pos = None
                self.hold_joint_pos = None
                self.command_confirmed.clear()
                self.confirm_button.disabled = True
                self.confirm_trajectory_button.disabled = True
                self.status_text.value = "🚀 Auto-approving entire trajectory..."
                print(f"[CONFIRM TRAJECTORY] Auto-approve enabled!", file=sys.stderr, flush=True)
            else:
                self.status_text.value = "⚠️ No pending motion to confirm"

    def _setup_visualization(self) -> None:
        """Prepare Viser overlays for live robot state and camera feeds."""

        # Current robot state (semi-transparent blue)
        self.base_frame_left_real = self.viser_server.scene.add_frame("/franka_real", show_axes=False)
        self.urdf_vis_left_real = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/franka_real",
            mesh_color_override=(0.55, 0.75, 0.95),
        )
        for mesh in self.urdf_vis_left_real._meshes:
            mesh.opacity = 0.3  # type: ignore[attr-defined]

        # Desired/target state (semi-transparent purple)
        self.base_frame_left_target = self.viser_server.scene.add_frame("/franka_target", show_axes=False)
        self.urdf_vis_left_target = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/franka_target",
            mesh_color_override=(0.95, 0.35, 0.55),  # Red for target
        )
        for mesh in self.urdf_vis_left_target._meshes:
            mesh.opacity = 0.5  # type: ignore[attr-defined]

        # Executed hyrl command (semi-transparent green)
        self.base_frame_left_hyrl = self.viser_server.scene.add_frame("/franka_hyrl", show_axes=False)
        self.urdf_vis_left_hyrl = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/franka_hyrl",
            mesh_color_override=(0.55, 0.95, 0.35),  # Green for executed
        )
        for mesh in self.urdf_vis_left_hyrl._meshes:
            mesh.opacity = 0.3  # type: ignore[attr-defined]

        if self.bimanual and self.right_arm_extrinsic is not None:
            self.ik.base_frame_right.position = np.array(self.right_arm_extrinsic["position"])
            self.ik.base_frame_right.wxyz = np.array(self.right_arm_extrinsic["rotation"])

            self.base_frame_right_real = self.viser_server.scene.add_frame("/franka_real/right", show_axes=False)
            self.base_frame_right_real.position = self.ik.base_frame_right.position
            self.urdf_vis_right_real = viser.extras.ViserUrdf(
                self.viser_server,
                deepcopy(self.ik.urdf),
                root_node_name="/franka_real/right",
                mesh_color_override=(0.55, 0.75, 0.95),
            )
            for mesh in self.urdf_vis_right_real._meshes:
                mesh.opacity = 0.3  # type: ignore[attr-defined]

        self.viser_cam_img_handles: Dict[str, viser.GuiImageHandle] = {}

        # GUI controls
        if not self.robotiq_gripper:
            self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width", min=0.0, max=0.1, step=0.001, initial_value=0.1
            )
        else:
            self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width", min=0.0, max=1.0, step=0.005, initial_value=1.0
            )

        if self.bimanual:
            self.right_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width (R)", min=0.0, max=0.1, step=0.001, initial_value=0.1
            )

        # Confirmation controls
        self.viser_server.gui.add_markdown("---\n## Motion Confirmation")
        self.status_text = self.viser_server.gui.add_text(
            label="Status",
            initial_value="⏳ Waiting for command...",
        )
        self.confirm_button = self.viser_server.gui.add_button(
            label="✅ Confirm Single Waypoint",
            color="green",
            disabled=True,
        )
        self.confirm_button.on_click(lambda _: self._on_confirm_click())

        self.confirm_trajectory_button = self.viser_server.gui.add_button(
            label="🚀 Approve Entire Trajectory",
            color="blue",
            disabled=True,
        )
        self.confirm_trajectory_button.on_click(lambda _: self._on_confirm_trajectory_click())

        self.viser_server.gui.add_markdown(
            """
**Legend:**
- 🔵 Blue ghost: Current robot state
- 🔴 Red ghost: Pending target pose
- 🟢 Green ghost: Last executed command
"""
        )

        self.camera_frustum_handles: Dict[str, viser.CameraFrustumHandle] = {}

    def _update_visualization(self) -> None:
        """Continuously sync live robot state and camera frames into Viser."""

        while self.obs is None:
            time.sleep(0.025)

        while True:
            obs_copy = self.obs
            if obs_copy is None:
                time.sleep(self._update_period)
                continue

            # Update current robot state (blue)
            left_joint_pos = self._extract_joint_pos(obs_copy, "left")
            if left_joint_pos is not None:
                # Handle 7-joint config (arm only) by appending default gripper value
                if len(left_joint_pos) == 7:
                    gripper_val = 0.04 if not self.robotiq_gripper else 0.5
                    left_joint_pos = np.concatenate([left_joint_pos, [gripper_val]])
                self.urdf_vis_left_real.update_cfg(left_joint_pos)

            # Update pending target state (red)
            with self.command_lock:
                target_joint_pos = self.pending_joint_pos
                target_gripper_pos = self.pending_gripper_pos
                if target_joint_pos is None and self.pending_trajectory is not None:
                    target_joint_pos = self.pending_trajectory[-1]
                    target_gripper_pos = self.pending_trajectory_gripper_pos
                if target_joint_pos is not None:
                    gripper_val = self.pending_gripper_pos if self.pending_gripper_pos is not None else 0.0
                    gripper_val = target_gripper_pos if target_gripper_pos is not None else 0.0
                    if not self.robotiq_gripper:
                        gripper_val = gripper_val * 0.08
                    self.urdf_vis_left_target.update_cfg(
                        np.concatenate([target_joint_pos, [gripper_val]])
                    )

            # Update executed hyrl command (green)
            if self.hyrl_joint_pos is not None:
                gripper_val = self.hyrl_gripper_pos if self.hyrl_gripper_pos is not None else 0.0
                if not self.robotiq_gripper:
                    gripper_val = gripper_val * 0.08
                self.urdf_vis_left_hyrl.update_cfg(
                    np.concatenate([self.hyrl_joint_pos, [gripper_val]])
                )

            if self.bimanual:
                right_joint_pos = self._extract_joint_pos(obs_copy, "right")
                if right_joint_pos is not None:
                    # Handle 7-joint config (arm only) by appending default gripper value
                    if len(right_joint_pos) == 7:
                        gripper_val = 0.04 if not self.robotiq_gripper else 0.5
                        right_joint_pos = np.concatenate([right_joint_pos, [gripper_val]])
                    self.urdf_vis_right_real.update_cfg(right_joint_pos)

            rgb_images = obs_get_rgb(obs_copy)
            if rgb_images:
                for key, image in rgb_images.items():
                    if key not in self.viser_cam_img_handles:
                        self.viser_cam_img_handles[key] = self.viser_server.gui.add_image(
                            resize_with_center_crop(image, 224, 224), label=key
                        )
                    if self.visualize_rgbd:
                        self.viser_cam_img_handles[key].image = resize_with_center_crop(image, 224, 224)

                    if key not in self.camera_frustum_handles:
                        self.camera_frustum_handles[key] = self.viser_server.scene.add_camera_frustum(
                            name=f"camera_frustum_{key}",
                            fov=1.2,
                            aspect=1.0,
                            scale=0.05,
                            cast_shadow=False,
                            receive_shadow=False,
                        )
                    if self.visualize_rgbd:
                        self.camera_frustum_handles[key].image = resize_with_center_crop(image, 224, 224)

                    extrinsics = obs_copy.get(key, {}).get("extrinsics")
                    if extrinsics is not None:
                        self.camera_frustum_handles[key].position = tuple(extrinsics["position"])
                        self.camera_frustum_handles[key].wxyz = extrinsics["wxyz"]

                    if "depth_data" in obs_copy[key] and self.visualize_rgbd:
                        depth_data = obs_copy[key]["depth_data"]
                        points, colors = depth_color_to_pointcloud(
                            depth=depth_data,
                            rgb_img=image,
                            intrinsics=obs_copy[key]["intrinsics"]["left"][
                                "intrinsics_matrix"
                            ],  # We assume we're taking left camera image from a stereo pair
                            subsample_factor=4,
                            depth_clip_range=(0.015, 1.2),
                        )
                        self.viser_server.scene.add_point_cloud(
                            name=f"camera_frustum_{key}/point_cloud_{key}",
                            points=points,
                            colors=colors,
                            point_size=0.002,
                        )

                time.sleep(self._update_period)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        import sys
        if not hasattr(self, '_act_count'):
            self._act_count = 0
        self._act_count += 1

        if self._act_count == 1:
            print(f"[Agent.act] First call! obs keys: {list(obs.keys())}", file=sys.stderr, flush=True)
            for k, v in obs.items():
                if isinstance(v, dict):
                    print(f"  [{k}]: {list(v.keys())}", file=sys.stderr, flush=True)
                else:
                    print(f"  [{k}]: {type(v)}", file=sys.stderr, flush=True)
        elif self._act_count % 100 == 0:
            print(f"[Agent.act] Call #{self._act_count}", file=sys.stderr, flush=True)

        # SAFETY: Hold current position for first 10 cycles to allow robot to stabilize
        # This prevents commanding stale/invalid positions during startup
        if self._act_count <= 10:
            hold_action = self._current_hold_action(obs, f"Startup cycle {self._act_count}")
            return {"left": {"pos": hold_action}} if hold_action is not None else {}

        self.obs = deepcopy(obs)
        with self.command_lock:
            self.obs["hyrl_status"] = {"left": self._trajectory_status_payload()}

        # Populate pose/pose_mat from extrinsics loaded by the camera driver.
        for cam_key, cam_obs in self.obs.items():
            if isinstance(cam_obs, dict):
                extrinsics = cam_obs.get("extrinsics")
                if extrinsics is not None:
                    cam_obs["pose"] = np.concatenate([extrinsics["position"], extrinsics["wxyz"]])
                    cam_obs["pose_mat"] = extrinsics["pose_mat"]
        if self.debug_msgpack:
            print(f"[DEBUG msgpack] top-level keys: {list(self.obs.keys())}")
            for k, v in self.obs.items():
                if isinstance(v, dict):
                    img_keys = list(v.get("images", {}).keys()) if "images" in v else None
                    print(f"  [{k}] keys={list(v.keys())} images={img_keys}")

        # Send observation to VOS and get desired command
        response = self.franka_client.send_request(self.obs)

        # Check if there's a new command from VOS.
        left_response = self._wire_get(response, "left")
        if isinstance(left_response, dict):
            trajectory_payload = self._wire_get(left_response, "trajectory")
            if trajectory_payload is not None:
                new_trajectory = np.asarray(trajectory_payload, dtype=np.float32)
                command_id = (
                    self._wire_text(self._wire_get(left_response, "command_id"))
                    or self._wire_text(self._wire_get(left_response, "trajectory_id"))
                    or f"trajectory-{hash(new_trajectory.tobytes())}"
                )
                tolerance_raw = self._wire_get(left_response, "tolerance")
                gripper_raw = self._wire_get(left_response, "gripper")
                new_tolerance = float(tolerance_raw) if tolerance_raw is not None else 0.05
                new_gripper_pos = float(np.asarray(gripper_raw, dtype=np.float32)) if gripper_raw is not None else 1.0

                if new_trajectory.ndim == 2 and new_trajectory.shape[1] >= 7:
                    new_trajectory = new_trajectory[:, :7].astype(np.float32)
                    with self.command_lock:
                        known_command = command_id in {
                            self.pending_trajectory_id,
                            self.active_trajectory_id,
                            self.completed_trajectory_id,
                        }
                        if not known_command:
                            print(f"[NEW TRAJECTORY] Received {len(new_trajectory)} waypoints from VOS")
                            print(f"  Command ID: {command_id}")
                            print(f"  First waypoint: {new_trajectory[0]}")
                            print(f"  Final waypoint: {new_trajectory[-1]}")

                            current_left = self._extract_joint_pos(obs, "left")
                            if current_left is not None and len(current_left) >= 7:
                                self.hold_joint_pos = np.asarray(current_left[:7], dtype=np.float32)
                                print(f"  Will hold current position: {self.hold_joint_pos}")
                            else:
                                print("  WARNING: Could not extract current position; no command will publish until state arrives")
                                self.hold_joint_pos = None

                            self.pending_trajectory = new_trajectory
                            self.pending_trajectory_id = command_id
                            self.pending_trajectory_tolerance = new_tolerance
                            self.pending_trajectory_gripper_pos = new_gripper_pos
                            self.pending_joint_pos = None
                            self.pending_gripper_pos = None
                            self.hyrl_joint_pos = None
                            self.hyrl_gripper_pos = None
                            self.active_trajectory_error = None
                            self.trajectory_status = "pending_approval"
                            self.confirm_button.disabled = True
                            self.confirm_trajectory_button.disabled = False
                            self.status_text.value = (
                                f"🔴 New trajectory received ({len(new_trajectory)} waypoints) - please review and approve"
                            )
                            self._visualize_trajectory(new_trajectory, command_id)
                            print("[WAITING] User must click 'Approve Entire Trajectory' in viser to execute")
                else:
                    print(f"[NEW TRAJECTORY] Ignoring malformed trajectory shape={new_trajectory.shape}")
            else:
                joint_payload = self._wire_get(left_response, "joint_pos")
                if joint_payload is None:
                    joint_payload = self._wire_get(left_response, "pos")
                if joint_payload is not None:
                    new_joint_pos = np.asarray(joint_payload, dtype=np.float32)
                    new_gripper_raw = self._wire_get(left_response, "gripper")
                    new_gripper_pos = (
                        float(np.asarray(new_gripper_raw, dtype=np.float32))
                        if new_gripper_raw is not None
                        else 1.0
                    )

                    with self.command_lock:
                        # Check if this is a new command (different from pending AND executed)
                        is_new_command = (
                            (self.pending_joint_pos is None or not np.allclose(new_joint_pos, self.pending_joint_pos, atol=1e-4))
                            and (self.hyrl_joint_pos is None or not np.allclose(new_joint_pos, self.hyrl_joint_pos, atol=1e-4))
                        )

                        if is_new_command:
                            print(f"[NEW COMMAND] Received target pose from VOS")
                            print(f"  Joint positions: {new_joint_pos}")
                            print(f"  Gripper: {new_gripper_pos}")

                            # Save current robot position to hold while waiting for confirmation
                            current_left = self._extract_joint_pos(obs, "left")
                            if current_left is not None and len(current_left) >= 7:
                                self.hold_joint_pos = np.asarray(current_left[:7], dtype=np.float32)
                                print(f"  Will hold current position: {self.hold_joint_pos}")
                            else:
                                print(f"  WARNING: Could not extract current position; no command will publish until state arrives")
                                self.hold_joint_pos = None

                            # Store pending command
                            self.pending_joint_pos = new_joint_pos
                            self.pending_gripper_pos = new_gripper_pos

                            # Check if we're in legacy auto-approve waypoint-stream mode
                            if self.auto_approve_trajectory:
                                self.trajectory_waypoint_count += 1
                                print(f"[AUTO-APPROVE] Waypoint #{self.trajectory_waypoint_count} (legacy stream mode)")
                                self.hyrl_joint_pos = self.pending_joint_pos.copy()
                                self.hyrl_gripper_pos = self.pending_gripper_pos
                                self.pending_joint_pos = None
                                self.pending_gripper_pos = None
                                self.hold_joint_pos = None
                                self.status_text.value = f"🚀 Auto-approved waypoint #{self.trajectory_waypoint_count}"
                            else:
                                # Manual confirmation required
                                self.command_confirmed.clear()
                                self.hyrl_joint_pos = None
                                self.hyrl_gripper_pos = None
                                self.confirm_button.disabled = False
                                self.confirm_trajectory_button.disabled = False
                                self.status_text.value = "🔴 New command received - please review and confirm"
                                print(f"[WAITING] User must click 'Confirm' button in viser to execute")

                        # Backward compatibility for older button-confirm flow.
                        elif self.pending_joint_pos is not None and self.command_confirmed.is_set():
                            print(f"[CONFIRMED] User approved - executing motion")
                            self.hyrl_joint_pos = self.pending_joint_pos.copy()
                            self.hyrl_gripper_pos = self.pending_gripper_pos
                            self.pending_joint_pos = None
                            self.pending_gripper_pos = None
                            self.hold_joint_pos = None
                            self.command_confirmed.clear()
                            self.status_text.value = "✅ Motion executing..."

        if self.debug_msgpack:
            print(response)

        # Initialize action logging counter
        if not hasattr(self, '_action_log_count'):
            self._action_log_count = 0
        self._action_log_count += 1

        # When waiting for confirmation, hold current position instead of moving to IK handle
        with self.command_lock:
            active_target = self._active_trajectory_target_locked(obs)
            if active_target is not None:
                gripper_val = (
                    self.active_trajectory_gripper_pos
                    if self.active_trajectory_gripper_pos is not None
                    else self.left_gripper_slider_handle.value
                )
                left_target = np.concatenate([active_target, [gripper_val]]).astype(np.float32)
            elif self.pending_joint_pos is not None or self.pending_trajectory is not None:
                # Command is pending user confirmation - hold the saved position
                if self.hold_joint_pos is not None:
                    # Use the saved hold position (7 joints, need to add gripper)
                    left_target = np.concatenate([self.hold_joint_pos, [self.left_gripper_slider_handle.value]]).astype(np.float32)
                else:
                    # Fallback: extract current position from obs
                    current_left = self._extract_joint_pos(obs, "left")
                    if current_left is not None and len(current_left) >= 7:
                        current_left = np.asarray(current_left, dtype=np.float32)
                        # Pad to 8 joints if needed
                        if len(current_left) == 7:
                            left_target = np.concatenate([current_left, [self.left_gripper_slider_handle.value]]).astype(np.float32)
                        else:
                            left_target = current_left.copy()
                            left_target[-1] = self.left_gripper_slider_handle.value
                    else:
                        print("[WAITING] Pending command but no current joints; publishing no command")
                        return {}
            else:
                # In GaP-confirm mode, never fall back to a potentially stale
                # Viser IK handle before a command has arrived.
                if self.hold_on_empty_response:
                    left_target = self._current_hold_action(obs, "No active GaP command")
                    if left_target is None:
                        return {}
                else:
                    # Use IK solution from viser handle (normal teleop mode)
                    left_target = np.asarray(self.ik.joints["left"], dtype=np.float32)
                    left_target[-1] = self.left_gripper_slider_handle.value
                    if self._action_log_count <= 5:
                        print(f"[TELEOP] Following viser IK handle: {left_target[:3]}")

        action: Dict[str, Dict[str, np.ndarray]] = {"left": {"pos": left_target}}

        # Debug: log what action we're about to return
        if self._action_log_count <= 5 or self._action_log_count % 50 == 0:
            print(f"[ACTION #{self._action_log_count}] Returning action: pos={left_target[:3]} (first 3 joints), pending={self.pending_joint_pos is not None}, hyrl={self.hyrl_joint_pos is not None}")

        # Execute confirmed command
        if self.hyrl_joint_pos is not None:
            gripper_source = (
                self.hyrl_gripper_pos
                if self.hyrl_gripper_pos is not None
                else self.left_gripper_slider_handle.value
            )
            if not self.robotiq_gripper:
                gripper_act = gripper_source * 0.08
            else:
                gripper_act = gripper_source
            action: Dict[str, Dict[str, np.ndarray]] = {
                "left": {"pos": np.concatenate([self.hyrl_joint_pos, [gripper_act]])}
            }
            # Log the actual confirmed action being sent
            if self._action_log_count <= 5 or self._action_log_count % 50 == 0:
                print(f"[CONFIRMED ACTION] Sending hyrl target: {self.hyrl_joint_pos[:3]} (first 3 joints), full={action['left']['pos']}")
            if self.trajectory_status not in {"executing", "pending_approval", "succeeded", "failed"}:
                self.status_text.value = "⏳ Waiting for next command..."

        if self.bimanual:
            assert "right" in self.ik.joints, "bimanual mode requires both IK solutions"
            right_target = np.asarray(self.ik.joints["right"], dtype=np.float32)
            right_target[-1] = self.right_gripper_slider_handle.value
            action["right"] = {"pos": right_target}

        return action

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        """Expose the joint-position action specification for Franka OSC."""

        action_spec = {
            "left": {"pos": Array(shape=(self.ik.joint_count,), dtype=np.float32)},
        }
        if self.bimanual:
            action_spec["right"] = {"pos": Array(shape=(self.ik.joint_count,), dtype=np.float32)}
        return action_spec


__all__ = ["FrankaOscClientCartesianConfirmAgent"]
