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


class FrankaPyrokiViserAgent(Agent):
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
        visualize_rgbd: bool = True,
        robotiq_gripper: bool = False,
        viser_port: int = 8080,
        wrist_cam_key: Optional[str] = None,
        wrist_cam_extrinsic_file: Optional[str] = None,
    ) -> None:
        self.bimanual = bimanual
        self.right_arm_extrinsic = right_arm_extrinsic
        self.visualize_rgbd = visualize_rgbd
        self.robotiq_gripper = robotiq_gripper
        self.wrist_cam_key = wrist_cam_key
        self._T_gripper_cam: Optional[np.ndarray] = None
        if wrist_cam_extrinsic_file is not None:
            path = Path(wrist_cam_extrinsic_file)
            with open(path) as f:
                cal = json.load(f)
            self._T_gripper_cam = np.array(cal["T_gripper_cam"])
        if self.bimanual:
            assert right_arm_extrinsic is not None, (
                "right_arm_extrinsic must be provided for bimanual Franka configuration"
            )

        self.viser_server = viser.ViserServer(port=viser_port)
        self.ik = FrankaPyroki(
            rate=ik_rate,
            viser_server=self.viser_server,
            bimanual=bimanual,
            robot_description=robot_description,
        )
        if self.robotiq_gripper:
            self.ik.transform_handles.get("left").tcp_offset_frame.wxyz = vtf.SO3.from_rpy_radians(
                0.0, 0.0, np.pi / 4
            ).wxyz
            self.ik.transform_handles.get("left").tcp_offset_frame.position = (0.0, 0.0, -0.157)
        self.ik_thread = threading.Thread(target=self.ik.run, name="franka_pyroki_ik")
        self.ik_thread.daemon = True
        self.ik_thread.start()

        self.obs: Optional[Dict[str, Any]] = None
        self._synced_to_real = False
        self._sync_cooldown: int = 0  # frames to hold IK snapshot after sync (lets IK thread settle)
        self._hold_joints: Optional[np.ndarray] = None  # IK snapshot held during cooldown
        self._user_set_gripper: bool = False  # echo observed gripper until user touches slider
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

        # Fall back to top-level fields if the environment exposes single-arm observations.
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

        if self.wrist_cam_key is not None:
            self.wrist_cam_pointcloud_toggle = self.viser_server.gui.add_checkbox(
                label="Wrist Cam Point Cloud", initial_value=True
            )
            self.cam_color_mode_toggle = self.viser_server.gui.add_checkbox(
                label="Color by Camera (green=side, red=wrist)", initial_value=False
            )

        if self.robotiq_gripper:
            self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width (echo observed)", min=0.0, max=1.0, step=0.005, initial_value=1.0
            )
        else:
            self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width (echo observed)", min=0.0, max=0.1, step=0.001, initial_value=0.1
            )

        @self.left_gripper_slider_handle.on_update
        def _on_gripper_update(_):
            if not self._user_set_gripper:
                self._user_set_gripper = True
                self.left_gripper_slider_handle.label = "Gripper Width"

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
                        left_joint_pos = self._extract_joint_pos(obs_copy, "left")
                        if left_joint_pos is not None:
                            joint_names = [f"panda_joint{i}" for i in range(1, 8)]
                            self.ik.urdf.update_cfg(dict(zip(joint_names, left_joint_pos[:7])))
                            T_base_hand = self.ik.urdf.get_transform("panda_hand", "panda_link0")
                            T_base_cam = T_base_hand @ self._T_gripper_cam
                            pos = T_base_cam[:3, 3].astype(np.float32)
                            q_xyzw = Rotation.from_matrix(T_base_cam[:3, :3]).as_quat()
                            wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
                            self.camera_frustum_handles[key].position = tuple(pos)
                            self.camera_frustum_handles[key].wxyz = wxyz
                    else:
                        extrinsics = obs_copy.get(key, {}).get("extrinsics")
                        if extrinsics is not None:
                            self.camera_frustum_handles[key].position = tuple(extrinsics["position"])
                            self.camera_frustum_handles[key].wxyz = extrinsics["wxyz"]

                    wrist_pc_enabled = (
                        key != self.wrist_cam_key
                        or not hasattr(self, "wrist_cam_pointcloud_toggle")
                        or self.wrist_cam_pointcloud_toggle.value
                    )
                    if "depth_data" in obs_copy[key] and self.visualize_rgbd and wrist_pc_enabled:
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
                        color_mode_on = (
                            hasattr(self, "cam_color_mode_toggle")
                            and self.cam_color_mode_toggle.value
                        )
                        if color_mode_on:
                            solid = [255, 0, 0] if key == self.wrist_cam_key else [0, 255, 0]
                            colors = np.full((len(points), 3), solid, dtype=np.uint8)
                        self.viser_server.scene.add_point_cloud(
                            name=f"camera_frustum_{key}/point_cloud_{key}",
                            points=points,
                            colors=colors,
                            point_size=0.0008 if color_mode_on else 0.002,
                        )

                time.sleep(self._update_period)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        self.obs = deepcopy(obs)

        left_joint_pos = self._extract_joint_pos(self.obs, "left")
        if not self._synced_to_real:
            if left_joint_pos is None:
                return {}
            self.ik.sync_to_joint_pos(left_joint_pos, "left")
            # Snapshot IK output immediately after sync (ik.joints["left"] was just set to
            # real joints by sync_to_joint_pos). Hold this snapshot during cooldown so we
            # don't follow a transient bad IK solution caused by the IK thread racing with
            # the gizmo update inside sync_to_joint_pos.
            self._hold_joints = np.asarray(self.ik.joints["left"], dtype=np.float32).copy()
            self._sync_cooldown = 10  # ~0.33 s at 30 Hz
            self._synced_to_real = True

        # Hold the snapshot steady while IK settles.
        if self._sync_cooldown > 0:
            self._sync_cooldown -= 1
            return {"left": {"pos": self._hold_joints.copy()}}

        left_target = np.asarray(self.ik.joints["left"], dtype=np.float32)
        # Echo the observed gripper value until the user moves the slider.
        if not self._user_set_gripper and left_joint_pos is not None and len(left_joint_pos) > 7:
            left_target[-1] = float(left_joint_pos[-1])
        else:
            left_target[-1] = self.left_gripper_slider_handle.value

        action: Dict[str, Dict[str, np.ndarray]] = {"left": {"pos": left_target}}

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
