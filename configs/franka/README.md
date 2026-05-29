# Franka configs

Per-config notes live below. For higher-level workflow notes on the confirm flow
(red/green/blue ghosts, button semantics) see [README_CONFIRM.md](README_CONFIRM.md).

## GaP-driven confirm flow (current launch path)

Two processes need to be up: the GaP `sim_bridge` server (provides the policy
commands) and the `rr-session` client (drives the real Franka and shows Viser).

### 1. Start the GaP sim_bridge server

```bash
cd /home/r2d2/gap_popcorn
uv run python services/sim_bridge/server.py --port 50060 --init-suite franka_real
```

What this brings up:
- `127.0.0.1:50060` — gRPC services (`SimBridge`, `Observation`, `Gripper`, `RobotControl`)
- `0.0.0.0:9001` — `MsgpackNumpyServer` inside `FrankaRealEnv`, the rendezvous
  point the rr-session confirm agent connects to

`--init-suite franka_real` runs the `Init` step at startup so port 9001 opens
immediately. Without it, port 9001 only opens after some external caller issues
`Init(suite_name="franka_real")` over gRPC.

The server does **not** connect to the robot itself; it just waits for the
rr-session client to push observations and (optionally) for a GaP policy to
push commands back through it. Safe to leave running.

Log to follow: `/tmp/sim_bridge.log` (if you tee it) — look for
`[MsgpackServer] listening on 0.0.0.0:9001` and `[SimBridge.Init] Success!`.

### 2. Start the rr-session client

```bash
cd /home/r2d2/robots_realtime
uv run rr-session configs/franka/franka_curobo_all_cam.yaml
```

Connects to the real Franka at `172.16.0.1` (FCI must be active, joints
unlocked, e-stop released) and to the Robotiq on `/dev/robotiq`. Opens Viser on
`http://localhost:8765`.

The agent in this config (`FrankaOscClientCartesianConfirmAgent`) has
`hold_on_empty_response: true`, so the arm holds its measured pose until GaP
pushes a command. The Viser IK gizmo is **ignored** in this mode — confirm-flow
configs only execute commands routed through the msgpack server.

## GaP `vos eval` workflow (delta_move example)

> **Stale — written against the older `graph-as-policy` checkout.** The
> `delta_move` example and `examples/franka_platform_curobo_only.yaml` do not
> exist in `gap_popcorn`, and `gap_popcorn`'s msgpack bridge now owns port
> 9001 — which collides with the Ray Serve `grpc_port: 9001` described below.
> This section needs a rewrite against `gap_popcorn` before use.

Runs a pre-built CuRobo workflow that moves the EE by a fixed `(dx, dy, dz)`
delta in the world frame. Three processes need to be up:

1. **sim_bridge** (this side of the GaP repo, gRPC + msgpack rendezvous)
2. **rr-session** (this repo, drives the real Franka, connects to sim_bridge msgpack)
3. **`vos eval`** (GaP side, spawns Ray Serve internally to host CuRobo)

### 1. Start sim_bridge

```bash
cd /home/r2d2/graph-as-policy
uv run python -m services.sim_bridge.server --port 50060 --init-suite franka_real
```

(Use `python -m ...`, not `python services/sim_bridge/server.py`, so the
`services` package is importable — required when the curobo IK path is
exercised.)

### 2. Start rr-session

```bash
cd /home/r2d2/robots_realtime
uv run rr-session configs/franka/franka_curobo_all_cam.yaml
```

### 3. Run the workflow

```bash
cd /home/r2d2/graph-as-policy
CUDA_HOME=/usr/local/cuda-12.1 CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 \
  uv run vos eval \
    --task-file examples/real_franka/delta_move/task.yaml \
    --workflow-dir examples/real_franka/delta_move/graph \
    --external-sim-port 50060 \
    --enable-tracing --log-level INFO
```

`vos eval` brings up its own Ray Serve gRPC proxy on **port 9001** (not the
default 9000 — sim_bridge's msgpack owns 9000). This is set by
`ray_serve.grpc_port: 9001` in `examples/franka_platform_curobo_only.yaml`.

`--external-sim-port 50060` tells `vos eval` to reuse the sim_bridge from
step 1 instead of spawning its own.

### Editing the delta

The `(dx, dy, dz)` values are hardcoded in
`examples/real_franka/delta_move/graph/workflow.json`, in the
`compute_target.inputs` block. Units are meters; world frame. Currently set to
`(0, 0, -0.20)` = 20 cm down.

The workflow uses `PlanDirectedLinear` with `allowed_axes: ["X", "Y", "Z"]`,
which is the only CuRobo function currently ported to v0.8 in
`third_party/curobo_api.py`. `SolveIK`, `PlanLinear`, and `PlanToPose` are
stubbed for v0.8 — don't swap the method.

## What if I just want Viser-gizmo teleop, no GaP?

Use one of the non-confirm configs — they don't open a msgpack client at all
and you don't need the sim_bridge server:

- `franka_robotiq_viser_teleop.yaml` — pyroki teleop, Robotiq, no cameras
- `franka_robotiq_cams.yaml` — same agent, adds top + wrist cameras

## Other configs in this directory

| Config | Agent | GaP server required? | Notes |
|---|---|---|---|
| `franka_robotiq_client_confirm_hold_no_camera.yaml` | confirm | yes | No cameras; safe for first-run bring-up |
| `franka_robotiq_client_confirm_no_camera.yaml` | confirm | yes | Like above but does not hold on empty response |
| `franka_robotiq_client_confirm.yaml` | confirm | yes | Adds cameras |
| `franka_robotiq_client_confirm_all_cams.yaml` | confirm | yes | All cameras attached |
| `franka_curobo_all_cam.yaml` | confirm (fake gripper) | yes | Used by the `vos eval` workflow above; top + wrist cameras, no-op gripper shim |
| `franka_robotiq_client*.yaml` (non-confirm) | client | yes | Auto-executes GaP commands without approval |
| `franka_robotiq_viser_teleop.yaml`, `franka_robotiq_cams.yaml` | pyroki teleop | no | Direct Viser-gizmo teleop |
| `franka_fake_gripper_*.yaml` | varies | varies | Same agents but with a no-op gripper shim |
