"""Teleoperation agent that combines PyRoKi IK with Franka OSC control."""

from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
from dm_env.specs import Array
from scipy.spatial.transform import Rotation

from robots_realtime.agents.agent import Agent
from robots_realtime.robots.inverse_kinematics.franka_pyroki import FrankaPyroki
from robots_realtime.sensors.cameras.camera_utils import obs_get_rgb, resize_with_center_crop
from robots_realtime.utils.depth_utils import depth_color_to_pointcloud
from robots_realtime.utils.server_client_utils import SyncMsgpackNumpyClient


class FrankaOscClientCartesianAgent(Agent):
    """Interactive teleoperation agent for Franka OSC robots.

    The agent exposes Viser transform gizmos (powered by :class:`FrankaPyroki`) that
    continuously solve for joint targets using PyRoKi. The resulting joint targets are fed to
    the :class:`robots_realtime.robots.franka_osc.FrankaPanda` controller through the environment
    interface, making it possible to drive the real robot with Viser while monitoring live
    state feedback.
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
        client_port: int = 9001,
        debug_msgpack: bool = False,
        hold_on_empty_response: bool = False,
        wrist_cam_key: Optional[str] = None,
        wrist_cam_extrinsic_file: Optional[str] = None,
    ) -> None:
        self.bimanual = bimanual
        self.robotiq_gripper = robotiq_gripper
        self.right_arm_extrinsic = right_arm_extrinsic
        self.visualize_rgbd = visualize_rgbd
        self.debug_msgpack = debug_msgpack
        self.hold_on_empty_response = hold_on_empty_response

        # Wrist-cam dynamic extrinsic — same machinery as the confirm
        # agent. CameraNode drivers only populate static `extrinsics`
        # for cameras with `extrinsics_file:` set; the wrist ZED is
        # streamed and has none, so its on-wire pose would be a
        # zero-quaternion (which downstream scipy.Rotation.from_quat
        # chokes on). Compute it live every act() via
        # FK × T_gripper_cam.
        self.wrist_cam_key = wrist_cam_key
        self._T_gripper_cam: Optional[np.ndarray] = None
        if wrist_cam_extrinsic_file is not None:
            with open(Path(wrist_cam_extrinsic_file)) as f:
                self._T_gripper_cam = np.array(json.load(f)["T_gripper_cam"])
        if self.bimanual:
            assert right_arm_extrinsic is not None, (
                "right_arm_extrinsic must be provided for bimanual Franka configuration"
            )

        self.hyrl_joint_pos = None
        self.hyrl_gripper_pos = None

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

        # Forward-kinematics URDF for the wrist-cam pose calculation.
        # Use a deepcopy so calling `update_cfg`/`get_transform` here
        # doesn't race the IK thread which mutates `self.ik.urdf`.
        self._fk_urdf = deepcopy(self.ik.urdf) if self._T_gripper_cam is not None else None

        if self.robotiq_gripper:
            self.ik.transform_handles.get("left").tcp_offset_frame.wxyz = vtf.SO3.from_rpy_radians(
                0.0, 0.0, np.pi / 4
            ).wxyz
            self.ik.transform_handles.get("left").tcp_offset_frame.position = (0.0, 0.0, -0.157)

        self.franka_client = SyncMsgpackNumpyClient(host=client_host, port=client_port)

        self.obs: Optional[Dict[str, Any]] = None
        self._update_period = 0.05

        # Latched arm-hold target used by the `hold_on_empty_response`
        # fallback. Echoing the live measurement every tick collapses
        # the OSC controller's position stiffness (commanded ≈ current
        # → zero task-space error → near-zero torque), and the arm
        # drifts under gravity. We latch the first valid observation
        # and keep returning it. Mirrors tiptop's _SharedState
        # `last_arm_target` pattern (see ../tiptop/tiptop/realtime/
        # franka_rr.py:127-141, 240-255).
        self._hold_arm_joints: Optional[np.ndarray] = None

        # Full-trajectory dispatch state. When the bridge sends a wire
        # frame containing `left.trajectory` + a new `command_id`, we
        # activate it locally (no approval gate) and walk the trajectory
        # by time-interpolating between waypoints inside `act()`. This
        # avoids the bridge's per-waypoint blocking loop. Tunables match
        # the confirm-agent's trajectory mode.
        self._active_trajectory: Optional[np.ndarray] = None
        self._active_trajectory_id: Optional[bytes] = None
        self._active_trajectory_times: Optional[np.ndarray] = None
        self._active_trajectory_start_time: Optional[float] = None
        self._active_trajectory_gripper: Optional[float] = None
        self._active_trajectory_tolerance: float = 0.05
        self._completed_trajectory_id: Optional[bytes] = None
        self._trajectory_joint_speed_limit: float = 0.35  # rad/s
        self._trajectory_min_segment_dt: float = 0.12     # s
        self._trajectory_min_duration: float = 1.0        # s
        self._active_trajectory_settle_timeout_s: float = 5.0

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

    def _setup_visualization(self) -> None:
        """Prepare Viser overlays for live robot state and camera feeds."""

        self.base_frame_left_real = self.viser_server.scene.add_frame("/franka_real", show_axes=False)
        self.urdf_vis_left_real = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/franka_real",
            mesh_color_override=(0.55, 0.75, 0.95),
        )
        for mesh in self.urdf_vis_left_real._meshes:
            mesh.opacity = 0.3  # type: ignore[attr-defined]

        self.base_frame_left_hyrl = self.viser_server.scene.add_frame("/franka_hyrl", show_axes=False)
        self.urdf_vis_left_hyrl = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/franka_hyrl",
            mesh_color_override=(0.55, 0.35, 0.95),
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

            left_joint_pos = self._extract_joint_pos(obs_copy, "left")
            if left_joint_pos is not None:
                self.urdf_vis_left_real.update_cfg(left_joint_pos)

            if self.hyrl_joint_pos is not None:
                self.urdf_vis_left_hyrl.update_cfg(
                    np.concatenate([self.hyrl_joint_pos, [self.hyrl_gripper_pos * 0.08]])
                )

            if self.bimanual:
                right_joint_pos = self._extract_joint_pos(obs_copy, "right")
                if right_joint_pos is not None:
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

                    if key == self.wrist_cam_key and self._T_gripper_cam is not None:
                        if left_joint_pos is not None:
                            joint_names = [f"panda_joint{i}" for i in range(1, 8)]
                            self._fk_urdf.update_cfg(
                                dict(zip(joint_names, np.asarray(left_joint_pos)[:7]))
                            )
                            T_base_hand = self._fk_urdf.get_transform(
                                "panda_hand", "panda_link0"
                            )
                            T_base_cam = T_base_hand @ self._T_gripper_cam
                            pos = T_base_cam[:3, 3].astype(np.float32)
                            q_xyzw = Rotation.from_matrix(T_base_cam[:3, :3]).as_quat()
                            wxyz = np.array(
                                [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
                                dtype=np.float32,
                            )
                            self.camera_frustum_handles[key].position = tuple(pos)
                            self.camera_frustum_handles[key].wxyz = wxyz
                    else:
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
    # ------------------------------------------------------------------
    # Trajectory mode helpers
    # ------------------------------------------------------------------
    def _trajectory_times(self, trajectory: np.ndarray) -> np.ndarray:
        """Conservative time base for smooth joint-space playback.

        Ported from the confirm agent. Per-segment duration is
        ``max(joint_delta) / speed_limit``, clamped to a minimum so
        consecutive near-identical waypoints don't collapse to zero
        duration. If the total comes out below `trajectory_min_duration`,
        scale up uniformly.
        """
        if len(trajectory) <= 1:
            return np.zeros((len(trajectory),), dtype=np.float32)
        deltas = np.abs(np.diff(trajectory[:, :7], axis=0))
        speed_limit = max(float(self._trajectory_joint_speed_limit), 1e-3)
        segment_dt = np.max(deltas / speed_limit, axis=1)
        segment_dt = np.maximum(segment_dt, float(self._trajectory_min_segment_dt))
        total = float(np.sum(segment_dt))
        min_duration = max(
            float(self._trajectory_min_duration),
            float(self._trajectory_min_segment_dt) * (len(trajectory) - 1),
        )
        if total < min_duration:
            segment_dt *= min_duration / max(total, 1e-6)
        return np.concatenate([[0.0], np.cumsum(segment_dt)]).astype(np.float32)

    def _activate_trajectory(
        self,
        trajectory: np.ndarray,
        command_id: bytes,
        *,
        tolerance: float,
        gripper: float,
    ) -> None:
        self._active_trajectory = trajectory.astype(np.float32).copy()
        self._active_trajectory_id = command_id
        self._active_trajectory_times = self._trajectory_times(self._active_trajectory)
        self._active_trajectory_start_time = time.monotonic()
        self._active_trajectory_gripper = float(gripper)
        self._active_trajectory_tolerance = float(tolerance)
        print(
            f"[TRAJECTORY] Activated {len(trajectory)} waypoints "
            f"(id={command_id!r}, tolerance={tolerance:.3f}, "
            f"duration={float(self._active_trajectory_times[-1]):.2f}s)"
        )

    def _finish_trajectory(self, *, reason: str) -> None:
        self._completed_trajectory_id = self._active_trajectory_id
        self._active_trajectory = None
        self._active_trajectory_id = None
        self._active_trajectory_times = None
        self._active_trajectory_start_time = None
        self._active_trajectory_gripper = None
        print(f"[TRAJECTORY] Finished ({reason})")

    def _active_trajectory_target(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Return the time-interpolated 7-joint target, or None if no active trajectory."""
        if self._active_trajectory is None or self._active_trajectory_times is None:
            return None
        trajectory = self._active_trajectory
        times = self._active_trajectory_times
        if self._active_trajectory_start_time is None:
            self._active_trajectory_start_time = time.monotonic()
        elapsed = max(0.0, time.monotonic() - self._active_trajectory_start_time)
        duration = float(times[-1]) if len(times) > 0 else 0.0
        final_target = trajectory[-1]

        current_left = self._extract_joint_pos(obs, "left")
        current: Optional[np.ndarray] = None
        if current_left is not None and len(current_left) >= 7:
            current = np.asarray(current_left[:7], dtype=np.float32)

        if len(trajectory) == 1 or elapsed >= duration:
            target = final_target
            if current is not None:
                final_error = float(np.linalg.norm(current - final_target))
                # Use a generous completion tolerance (max of caller's
                # tolerance and 0.12 rad) to avoid getting stuck waiting
                # on the last millirads — matches the confirm agent.
                completion_tol = max(self._active_trajectory_tolerance, 0.12)
                if final_error <= completion_tol:
                    self._finish_trajectory(reason="reached final waypoint")
                    return target
                if elapsed >= duration + self._active_trajectory_settle_timeout_s:
                    self._finish_trajectory(
                        reason=f"settle timeout, final_error={final_error:.3f}",
                    )
                    return target
            else:
                # Without joint feedback, end after the nominal duration.
                if elapsed >= duration + self._active_trajectory_settle_timeout_s:
                    self._finish_trajectory(reason="settle timeout (no joints)")
                    return target
            return target

        upper = int(np.searchsorted(times, elapsed, side="right"))
        lower = max(0, min(upper - 1, len(trajectory) - 2))
        upper = lower + 1
        t0 = float(times[lower])
        t1 = float(times[upper])
        alpha = 0.0 if t1 <= t0 else (elapsed - t0) / (t1 - t0)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        target = (
            (1.0 - alpha) * trajectory[lower] + alpha * trajectory[upper]
        ).astype(np.float32)
        return target

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        self.obs = deepcopy(obs)

        # Populate pose/pose_mat from extrinsics loaded by the camera driver.
        for cam_key, cam_obs in self.obs.items():
            if isinstance(cam_obs, dict):
                extrinsics = cam_obs.get("extrinsics")
                if extrinsics is not None:
                    cam_obs["pose"] = np.concatenate([extrinsics["position"], extrinsics["wxyz"]])
                    cam_obs["pose_mat"] = extrinsics["pose_mat"]

        # Wrist camera: streamed ZED has no `extrinsics_file`, so the
        # loop above leaves its pose/pose_mat absent. Compute them live
        # from FK × T_gripper_cam so the on-wire wrist obs carries a
        # valid pose (downstream scipy.Rotation.from_quat would error
        # on a zero quaternion). Mirrors the confirm agent.
        if (
            self.wrist_cam_key is not None
            and self._T_gripper_cam is not None
            and self._fk_urdf is not None
            and self.wrist_cam_key in self.obs
            and isinstance(self.obs[self.wrist_cam_key], dict)
        ):
            arm_obs = obs.get("left")
            left_joint_pos = (
                arm_obs.get("joint_pos") if isinstance(arm_obs, dict) else None
            )
            if left_joint_pos is not None:
                joint_names = [f"panda_joint{i}" for i in range(1, 8)]
                self._fk_urdf.update_cfg(
                    dict(zip(joint_names, np.asarray(left_joint_pos)[:7]))
                )
                T_base_hand = np.asarray(
                    self._fk_urdf.get_transform("panda_hand", "panda_link0"),
                    dtype=np.float64,
                )
                T_base_cam = T_base_hand @ self._T_gripper_cam
                pos = T_base_cam[:3, 3].astype(np.float32)
                q_xyzw = Rotation.from_matrix(T_base_cam[:3, :3]).as_quat()
                wxyz = np.array(
                    [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
                    dtype=np.float32,
                )
                self.obs[self.wrist_cam_key]["pose"] = np.concatenate([pos, wxyz])
                self.obs[self.wrist_cam_key]["pose_mat"] = T_base_cam.astype(np.float32)
        if self.debug_msgpack:
            print(f"[DEBUG msgpack] top-level keys: {list(self.obs.keys())}")
            for k, v in self.obs.items():
                if isinstance(v, dict):
                    img_keys = list(v.get("images", {}).keys()) if "images" in v else None
                    print(f"  [{k}] keys={list(v.keys())} images={img_keys}")
        response = self.franka_client.send_request(self.obs)

        left_response = response.get(b"left") if response else None
        if left_response is not None:
            self.hyrl_joint_pos = np.asarray(left_response.get(b"joint_pos"), dtype=np.float32)
            self.hyrl_gripper_pos = np.asarray(left_response.get(b"gripper"), dtype=np.float32)

            # Full-trajectory dispatch: when the wire frame carries
            # `trajectory` + a new `command_id`, activate it locally
            # (no approval gate) and walk it via time-interpolation. The
            # bridge keeps `joint_pos` populated as a fallback (equal to
            # the final waypoint), so this codepath is purely opt-in.
            traj_payload = left_response.get(b"trajectory")
            command_id = left_response.get(b"command_id")
            if (
                traj_payload is not None
                and command_id is not None
                and command_id != self._active_trajectory_id
                and command_id != self._completed_trajectory_id
            ):
                trajectory = np.asarray(traj_payload, dtype=np.float32)
                if trajectory.ndim == 2 and trajectory.shape[1] >= 7:
                    trajectory = trajectory[:, :7]
                    tol_raw = left_response.get(b"tolerance")
                    tol = float(tol_raw) if tol_raw is not None else 0.05
                    grip_raw = left_response.get(b"gripper")
                    grip = (
                        float(np.asarray(grip_raw, dtype=np.float32))
                        if grip_raw is not None
                        else 1.0
                    )
                    self._activate_trajectory(
                        trajectory, command_id, tolerance=tol, gripper=grip,
                    )
        if self.debug_msgpack:
            print(response)

        # Trajectory mode wins over the single-point path. When an active
        # trajectory exists, emit its time-interpolated target every tick.
        traj_target = self._active_trajectory_target(obs)
        if traj_target is not None:
            grip_val = (
                self._active_trajectory_gripper
                if self._active_trajectory_gripper is not None
                else self.left_gripper_slider_handle.value
            )
            if not self.robotiq_gripper:
                grip_act = grip_val * 0.08
            else:
                grip_act = grip_val
            action: Dict[str, Dict[str, np.ndarray]] = {
                "left": {"pos": np.concatenate([traj_target, [grip_act]]).astype(np.float32)}
            }
            if self.bimanual:
                right_target = np.asarray(self.ik.joints["right"], dtype=np.float32)
                right_target[-1] = self.right_gripper_slider_handle.value
                action["right"] = {"pos": right_target}
            return action

        left_target = np.asarray(self.ik.joints["left"], dtype=np.float32)
        if self.hold_on_empty_response and left_response is None:
            # Latch the first observed arm pose and reuse it. Re-reading
            # the live measurement every tick removes OSC position
            # stiffness and the arm floats away under gravity — see
            # ../tiptop/tiptop/realtime/franka_rr.py:240-255 for the
            # same fix on the server side.
            if self._hold_arm_joints is None:
                current_left = self._extract_joint_pos(obs, "left")
                if current_left is not None and len(current_left) >= 7:
                    self._hold_arm_joints = np.asarray(
                        current_left[:7], dtype=np.float32
                    ).copy()
            if self._hold_arm_joints is not None:
                if len(left_target) == 7:
                    left_target = self._hold_arm_joints.copy()
                else:
                    left_target = np.concatenate(
                        [self._hold_arm_joints, [self.left_gripper_slider_handle.value]]
                    ).astype(np.float32)
        else:
            # A live policy response is being followed: drop the latched
            # hold so the next quiet period re-latches at the new pose.
            self._hold_arm_joints = None
        left_target[-1] = self.left_gripper_slider_handle.value
        action: Dict[str, Dict[str, np.ndarray]] = {"left": {"pos": left_target}}

        if response.get(b"left") is not None:
            if not self.robotiq_gripper:
                gripper_act = self.hyrl_gripper_pos * 0.08
            else:
                gripper_act = self.hyrl_gripper_pos
            action: Dict[str, Dict[str, np.ndarray]] = {
                "left": {"pos": np.concatenate([self.hyrl_joint_pos, [gripper_act]])}
            }

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


__all__ = ["FrankaPyrokiViserAgent"]
