# Rotate Gripper Script

Simple standalone script to rotate the Franka gripper without needing VOS/graph-as-policy.

## How It Works

This script acts as a simple msgpack server that robots_realtime connects to. It:
1. Receives the current robot state from robots_realtime
2. Computes target joint positions with wrist rotated by specified degrees
3. Sends the rotation command back to robots_realtime
4. You confirm the motion in the viser interface

## Usage

### Step 1: Start the rotation server

In terminal 1:
```bash
cd /home/r2d2/robots_realtime
python scripts/rotate_gripper_90deg.py
```

This will start a msgpack server on port 9000 and wait for robots_realtime to connect.

### Step 2: Start robots_realtime

In terminal 2:
```bash
cd /home/r2d2/robots_realtime
python robots_realtime/runtime/main.py --config configs/franka/franka_robotiq_client_confirm_no_camera.yaml
```

robots_realtime will connect to the rotation server and receive the rotation command.

### Step 3: Confirm in viser

1. Open http://localhost:8765 in your browser
2. You'll see the rotation command pending
3. Click "✅ Confirm Single Waypoint" to execute the rotation

## Options

Rotate by different amounts:
```bash
python scripts/rotate_gripper_90deg.py --degrees 45   # Rotate 45°
python scripts/rotate_gripper_90deg.py --degrees 180  # Rotate 180°
python scripts/rotate_gripper_90deg.py --degrees -90  # Rotate -90° (opposite direction)
```

Use different port:
```bash
python scripts/rotate_gripper_90deg.py --port 9001
```

## Notes

- This script only rotates joint 7 (the wrist joint)
- The rotation is around the wrist axis (like turning a screwdriver)
- For more complex Cartesian rotations or collision-free motion, use VOS with CuRobo
- The script holds the target position, so the robot will stay at the rotated pose
- Press Ctrl+C in the script terminal to stop the server

## Example Output

```
=== Gripper Rotation Server ===
Listening on 127.0.0.1:9000
Rotation: 90.0° around wrist axis
Mode: single command

Waiting for robots_realtime to connect...

[Server] robots_realtime connected!
[Server] Initial joint positions: [0.1, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]
[Server] Initial joint 7 (wrist): 45.8°
[Server] Target joint 7 (wrist): 135.8°
[Server] Rotation: 90.0°

[Server] Ready to send rotation command!
[Server] In viser interface:
  1. Click '✅ Confirm Single Waypoint' to execute rotation
  OR
  2. Click '🚀 Approve Entire Trajectory' to auto-execute

[Server] ✓ Sending rotation command!
```
