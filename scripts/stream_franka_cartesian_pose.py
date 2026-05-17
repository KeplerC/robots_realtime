#!/usr/bin/env python3
"""Stream Franka Cartesian pose directly from libfranka.

This script does not use the robots_realtime message bus. It opens a direct
libfranka read connection through panda_py and prints RobotState.O_T_EE.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import panda_py
from scipy.spatial.transform import Rotation


DEFAULT_HOST = "172.16.0.1"

# Same Robotiq TCP transform used by the GaP real-Franka bridge.
DEFAULT_TCP_OFFSET_EE = np.array([0.0, 0.0, -0.157], dtype=np.float64)
DEFAULT_TCP_YAW_RAD = np.pi / 4.0


@dataclass
class Pose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    matrix: np.ndarray


def _duration_to_sec(d: Any) -> float:
    """Convert a libfranka Duration (or plain float/int) to seconds."""
    if hasattr(d, "to_sec"):
        return float(d.to_sec())
    return float(d)


def _libfranka_transform_to_matrix(value: Any) -> np.ndarray:
    """Convert a libfranka 16-float transform into a 4x4 numpy matrix.

    libfranka stores homogeneous transforms in column-major order.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr.copy()
    if arr.size != 16:
        raise ValueError(f"Expected 16 values for transform, got shape {arr.shape}")
    return arr.reshape((4, 4), order="F")


def _pose_from_matrix(matrix: np.ndarray) -> Pose:
    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    return Pose(
        position=np.asarray(matrix[:3, 3], dtype=np.float64),
        quaternion_wxyz=quat_wxyz,
        matrix=np.asarray(matrix, dtype=np.float64),
    )


def _tcp_from_ee(ee_pose: Pose, offset_ee: np.ndarray, yaw_rad: float) -> Pose:
    tcp_matrix = ee_pose.matrix.copy()
    tcp_matrix[:3, 3] = ee_pose.position + ee_pose.matrix[:3, :3] @ offset_ee
    tcp_matrix[:3, :3] = (
        Rotation.from_matrix(ee_pose.matrix[:3, :3]) * Rotation.from_euler("z", yaw_rad)
    ).as_matrix()
    return _pose_from_matrix(tcp_matrix)


def _pose_dict(pose: Pose, *, include_matrix: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "position": pose.position.tolist(),
        "quaternion_wxyz": pose.quaternion_wxyz.tolist(),
    }
    if include_matrix:
        out["matrix_row_major"] = pose.matrix.tolist()
    return out


def _print_json(
    *,
    state: Any,
    ee_pose: Pose,
    tcp_pose: Pose,
    tcp_offset_ee: np.ndarray,
    tcp_yaw_rad: float,
    include_matrix: bool,
) -> None:
    now = time.time()
    payload = {
        "timestamp": now,
        "robot_time_s": _duration_to_sec(getattr(state, "time", 0.0)),
        "joint_pos": np.asarray(state.q, dtype=np.float64).tolist(),
        "joint_vel": np.asarray(state.dq, dtype=np.float64).tolist(),
        "libfranka_O_T_EE": _pose_dict(ee_pose, include_matrix=include_matrix),
        "derived_robotiq_tcp": {
            **_pose_dict(tcp_pose, include_matrix=include_matrix),
            "offset_in_EE": tcp_offset_ee.tolist(),
            "mount_yaw_rad": tcp_yaw_rad,
        },
    }
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _print_human(
    *,
    state: Any,
    ee_pose: Pose,
    tcp_pose: Pose,
) -> None:
    now = time.time()
    q = np.asarray(state.q, dtype=np.float64)
    ee_xyz = " ".join(f"{v:+0.4f}" for v in ee_pose.position)
    ee_q = " ".join(f"{v:+0.4f}" for v in ee_pose.quaternion_wxyz)
    tcp_xyz = " ".join(f"{v:+0.4f}" for v in tcp_pose.position)
    tcp_q = " ".join(f"{v:+0.4f}" for v in tcp_pose.quaternion_wxyz)
    q_str = " ".join(f"{v:+0.4f}" for v in q)
    print(
        f"t={now:0.3f} robot_time={_duration_to_sec(getattr(state, 'time', 0.0)):0.6f} "
        f"O_T_EE.xyz=[{ee_xyz}] O_T_EE.wxyz=[{ee_q}] "
        f"tcp.xyz=[{tcp_xyz}] tcp.wxyz=[{tcp_q}] q=[{q_str}]",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Franka Cartesian pose directly from libfranka RobotState.O_T_EE."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Franka control IP/hostname (default: {DEFAULT_HOST})")
    parser.add_argument("--rate", type=float, default=30.0, help="Print rate in Hz (default: 30)")
    parser.add_argument("--once", action="store_true", help="Print one state then exit")
    parser.add_argument("--json", action="store_true", help="Print compact JSON lines")
    parser.add_argument("--matrix", action="store_true", help="Include 4x4 row-major transforms in JSON output")
    parser.add_argument(
        "--tcp-offset",
        type=float,
        nargs=3,
        default=DEFAULT_TCP_OFFSET_EE.tolist(),
        metavar=("X", "Y", "Z"),
        help="Derived TCP offset in EE frame, meters (default: 0 0 -0.157)",
    )
    parser.add_argument(
        "--tcp-yaw-rad",
        type=float,
        default=DEFAULT_TCP_YAW_RAD,
        help="Derived TCP yaw rotation about EE Z (default: pi/4)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    period = 0.0 if args.rate <= 0 else 1.0 / args.rate
    tcp_offset = np.asarray(args.tcp_offset, dtype=np.float64)

    print(
        f"[stream_franka_cartesian_pose] connecting directly to libfranka at {args.host}",
        file=sys.stderr,
        flush=True,
    )

    try:
        robot = panda_py.libfranka.Robot(args.host)
    except Exception as exc:
        print(
            f"Failed to connect to Franka at {args.host}: {exc}\n"
            "If robots_realtime is currently controlling the arm, stop it first; "
            "libfranka generally allows only one active robot connection.",
            file=sys.stderr,
        )
        return 1

    try:
        while True:
            state = robot.read_once()
            ee_matrix = _libfranka_transform_to_matrix(state.O_T_EE)
            ee_pose = _pose_from_matrix(ee_matrix)
            tcp_pose = _tcp_from_ee(ee_pose, tcp_offset, args.tcp_yaw_rad)

            if args.json:
                _print_json(
                    state=state,
                    ee_pose=ee_pose,
                    tcp_pose=tcp_pose,
                    tcp_offset_ee=tcp_offset,
                    tcp_yaw_rad=args.tcp_yaw_rad,
                    include_matrix=args.matrix,
                )
            else:
                _print_human(state=state, ee_pose=ee_pose, tcp_pose=tcp_pose)

            if args.once:
                return 0
            if period > 0:
                time.sleep(period)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error while reading Franka state: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
