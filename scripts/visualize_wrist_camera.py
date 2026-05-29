"""Visualize calibrated wrist camera extrinsic on the live Franka in Viser.

Reads joint states from rr-session ZMQ bus, applies FK, then shows the
camera frustum at T_base_hand @ T_gripper_cam in a Viser window.

Usage:
    uv run python scripts/visualize_wrist_camera.py \
        --extrinsic ../graph-as-policy/outputs/franka_wrist_camera_extrinsic.json
"""

import argparse
import json
import time
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np
import viser
import viser.extras
import zmq
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scipy.spatial.transform import Rotation

msgpack_numpy.patch()

_BROKER = "127.0.0.1"
_PORT = 5556
_TOPIC = "franka/joint_state"
_PANDA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def recv_latest_joints(sock) -> np.ndarray | None:
    joints = None
    while True:
        try:
            _, data = sock.recv_multipart(flags=zmq.NOBLOCK)
            env = msgpack.unpackb(data, raw=False)
            payload = env.get("data", env)
            jp = np.asarray(payload.get("joint_pos", []), dtype=np.float64)
            if jp.size >= 7:
                joints = jp[:7]
        except zmq.Again:
            break
    return joints


def fk_panda_hand(urdf, joints: np.ndarray) -> np.ndarray:
    cfg = dict(zip(_PANDA_JOINT_NAMES, joints))
    urdf.update_cfg(cfg)
    return urdf.get_transform("panda_hand")  # 4x4


def mat_to_pos_wxyz(T: np.ndarray):
    pos = T[:3, 3].astype(np.float32)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return pos, np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)  # wxyz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extrinsic", default="../graph-as-policy/outputs/franka_wrist_camera_extrinsic.json",
                        help="Path to franka_wrist_camera_extrinsic.json")
    parser.add_argument("--broker", default=_BROKER)
    parser.add_argument("--port", type=int, default=8766,
                        help="Viser port (default 8766 to avoid clashing with rr-session at 8765)")
    args = parser.parse_args()

    extrinsic_path = Path(args.extrinsic)
    with open(extrinsic_path) as f:
        cal = json.load(f)
    T_gripper_cam = np.array(cal["T_gripper_cam"])
    print(f"Loaded T_gripper_cam from {extrinsic_path}")
    print(f"  translation: {T_gripper_cam[:3, 3]}")

    urdf = load_robot_description("panda_description")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{args.broker}:{_PORT}")
    sock.setsockopt_string(zmq.SUBSCRIBE, _TOPIC)

    server = viser.ViserServer(port=args.port)
    print(f"Viser running at http://localhost:{args.port}")

    server.scene.add_frame("/franka", show_axes=False)
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/franka")

    cam_frustum = server.scene.add_camera_frustum(
        "/franka/wrist_cam",
        fov=1.2,
        aspect=16 / 9,
        scale=0.08,
        color=(255, 100, 0),
    )
    server.scene.add_frame("/franka/wrist_cam/axes", axes_length=0.05, axes_radius=0.003)

    joints = np.zeros(7)
    print("Waiting for joint states on ZMQ...")

    while True:
        fresh = recv_latest_joints(sock)
        if fresh is not None:
            joints = fresh

        T_base_hand = fk_panda_hand(urdf, joints)
        T_base_cam = T_base_hand @ T_gripper_cam

        urdf_vis.update_cfg(dict(zip(_PANDA_JOINT_NAMES, joints)))

        cam_pos, cam_wxyz = mat_to_pos_wxyz(T_base_cam)
        cam_frustum.position = cam_pos
        cam_frustum.wxyz = cam_wxyz

        time.sleep(0.02)  # 50 Hz


if __name__ == "__main__":
    main()
