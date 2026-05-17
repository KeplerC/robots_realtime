#!/usr/bin/env python3
"""
Simple script to rotate the Franka gripper 90 degrees.

This script acts as a simple msgpack server that robots_realtime connects to.
It receives the current robot state, computes a rotated joint position, and
sends it back as a command.

Usage:
    1. Run this script first:
       python scripts/rotate_gripper_90deg.py

    2. In another terminal, start robots_realtime (it will connect to this script):
       python robots_realtime/runtime/main.py --config configs/franka/franka_robotiq_client_confirm_no_camera.yaml

    3. When prompted in viser, click "Confirm Single Waypoint" to execute the rotation

Options:
    python scripts/rotate_gripper_90deg.py --degrees 45  # Rotate 45 degrees instead of 90
    python scripts/rotate_gripper_90deg.py --incremental  # Rotate in small steps
"""

import argparse
import asyncio
import struct
import sys
import time
import numpy as np
import msgpack
import msgpack_numpy as m

m.patch()  # Enable numpy array serialization


# ---------------------------------------------------------------------------
# Msgpack framing (same protocol as VOS msgpack bridge)
# ---------------------------------------------------------------------------

def encode_msg(obj: dict) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def decode_msg(raw: bytes) -> dict:
    return msgpack.unpackb(raw, raw=True)


async def send_framed(writer: asyncio.StreamWriter, obj: dict) -> None:
    payload = encode_msg(obj)
    header = struct.pack("!I", len(payload))
    writer.write(header + payload)
    await writer.drain()


async def recv_framed(reader: asyncio.StreamReader) -> dict:
    header = await reader.readexactly(4)
    (msg_len,) = struct.unpack("!I", header)
    payload = await reader.readexactly(msg_len)
    return decode_msg(payload)


# ---------------------------------------------------------------------------
# Simple rotation server
# ---------------------------------------------------------------------------

class GripperRotationServer:
    """Simple msgpack server that commands robot to rotate gripper."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, rotation_degrees: float = 90.0, incremental: bool = False):
        self.host = host
        self.port = port
        self.rotation_degrees = rotation_degrees
        self.incremental = incremental

        self.target_joints = None
        self.rotation_sent = False
        self.initial_joints = None

    async def start(self) -> None:
        import socket
        server = await asyncio.start_server(
            self._handle, self.host, self.port,
            reuse_address=True,
            start_serving=False,
        )
        # Enable SO_REUSEADDR to avoid "address already in use"
        for sock in server.sockets:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        await server.start_serving()
        print(f"=== Gripper Rotation Server ===")
        print(f"Listening on {self.host}:{self.port}")
        print(f"Rotation: {self.rotation_degrees}° around wrist axis")
        print(f"Mode: {'incremental' if self.incremental else 'single command'}")
        print()
        print("Waiting for robots_realtime to connect...")
        print("Start it with:")
        print("  python robots_realtime/runtime/main.py --config configs/franka/franka_robotiq_client_confirm_no_camera.yaml")
        print()

        async with server:
            await server.serve_forever()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        print("[Server] robots_realtime connected!")
        request_count = 0

        try:
            while True:
                # Receive observation from robots_realtime
                obs = await recv_framed(reader)
                request_count += 1

                # Extract current joint positions
                current_joints = None
                if b'left' in obs and b'joint_pos' in obs[b'left']:
                    current_joints = obs[b'left'][b'joint_pos']
                elif b'joint_pos' in obs:
                    current_joints = obs[b'joint_pos']

                if current_joints is None:
                    print(f"[Server] WARNING: Could not find joint positions. Keys: {list(obs.keys())}")
                    action = {}
                    await send_framed(writer, action)
                    continue

                current_joints = np.array(current_joints, dtype=np.float32)

                # Save initial joints on first observation
                if self.initial_joints is None:
                    self.initial_joints = current_joints.copy()
                    print(f"[Server] Initial joint positions: {self.initial_joints}")
                    print(f"[Server] Initial joint 7 (wrist): {np.rad2deg(self.initial_joints[6]):.1f}°")

                # Compute target on first observation
                if self.target_joints is None:
                    self.target_joints = current_joints.copy()
                    # Rotate joint 7 (wrist rotation, index 6)
                    self.target_joints[6] += np.deg2rad(self.rotation_degrees)
                    print(f"[Server] Target joint 7 (wrist): {np.rad2deg(self.target_joints[6]):.1f}°")
                    print(f"[Server] Rotation: {self.rotation_degrees}°")
                    print()
                    print("[Server] Ready to send rotation command!")
                    print("[Server] In viser interface:")
                    print("  1. Click '✅ Confirm Single Waypoint' to execute rotation")
                    print("  OR")
                    print("  2. Click '🚀 Approve Entire Trajectory' to auto-execute")
                    print()

                # Send rotation command
                if not self.rotation_sent:
                    action = {
                        "timestamp": time.time(),
                        "left": {
                            "joint_pos": self.target_joints.tolist(),
                            "gripper": 1.0  # Keep gripper open
                        }
                    }
                    self.rotation_sent = True
                    print(f"[Server] ✓ Sending rotation command!")
                else:
                    # Hold target position
                    action = {
                        "timestamp": time.time(),
                        "left": {
                            "joint_pos": self.target_joints.tolist(),
                            "gripper": 1.0
                        }
                    }

                if request_count % 100 == 0:
                    current_wrist = np.rad2deg(current_joints[6])
                    target_wrist = np.rad2deg(self.target_joints[6])
                    error = abs(target_wrist - current_wrist)
                    print(f"[Server] Request {request_count}: wrist at {current_wrist:.1f}°, target {target_wrist:.1f}°, error {error:.1f}°")

                await send_framed(writer, action)

        except asyncio.IncompleteReadError:
            print(f"[Server] robots_realtime disconnected (after {request_count} requests)")
        except Exception as e:
            import traceback
            print(f"[Server] Error: {e}")
            traceback.print_exc()


async def main_async(args):
    server = GripperRotationServer(
        host=args.host,
        port=args.port,
        rotation_degrees=args.degrees,
        incremental=args.incremental
    )
    await server.start()


def main():
    parser = argparse.ArgumentParser(description="Rotate Franka gripper via msgpack server")
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9000,
                        help='Server port (default: 9000)')
    parser.add_argument('--degrees', type=float, default=90.0,
                        help='Rotation amount in degrees (default: 90)')
    parser.add_argument('--incremental', action='store_true',
                        help='Rotate incrementally in small steps')
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")


if __name__ == '__main__':
    main()
