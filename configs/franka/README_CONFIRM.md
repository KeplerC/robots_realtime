# Franka Client with Motion Confirmation

This configuration adds **manual motion confirmation** before executing commands from VOS.

## Features

- **Pose Visualization**: Shows three robot states in viser:
  - 🔵 **Blue ghost**: Current robot state (real-time feedback)
  - 🔴 **Red ghost**: Pending target pose (from VOS command)
  - 🟢 **Green ghost**: Last executed command
  
- **Manual Confirmation**: A button in the viser UI that must be clicked before any motion executes

- **Safety**: No automatic motion execution - operator reviews and approves each command

## Usage

### 1. Start VOS serve (Terminal 1)

```bash
cd graph-as-policy
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDA_HOME=/usr/local/cuda-12.1 \
  uv run vos serve --config examples/franka_platform_pyroki_only.yaml
```

### 2. Start robots_realtime with confirmation (Terminal 2)

```bash
cd robots_realtime
uv run rr-session configs/franka/franka_robotiq_client_confirm.yaml --no-tui
```

### 3. Open viser UI

Navigate to `http://localhost:8765` in your browser to see:
- 3D visualization of the robot
- Status text showing pending commands
- **"✅ Confirm and Execute Motion"** button

### 4. Run VOS workflow (Terminal 3)

```bash
cd graph-as-policy
CUDA_HOME=/usr/local/cuda-12.1 CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 \
  uv run vos eval \
    --task-file examples/real_franka/forward_backward/task.yaml \
    --workflow-dir examples/real_franka/forward_backward/graph \
    --enable-tracing --log-level INFO --no-ray-start
```

### 5. Approve motions in viser

When VOS sends a command:
1. The **red ghost** (target pose) appears in viser
2. Status changes to "🔴 New command received - please review and confirm"
3. The **"✅ Confirm and Execute Motion"** button becomes enabled
4. **Review the target pose visually**
5. **Click the button** to execute
6. The motion executes and the green ghost shows the executed state

## Workflow Flow

```
VOS Command → robots_realtime receives → Red ghost appears → 
User reviews in viser → Click confirm button → Motion executes → 
Green ghost shows executed state → Wait for next command
```

## Safety Notes

- The robot will **NOT move** until you click the confirmation button
- Each motion requires separate confirmation
- You can see the exact target pose before execution
- Terminal output shows when commands are received and confirmed

## Comparison with Standard Client

| Feature | Standard Client | Confirmation Client |
|---------|----------------|---------------------|
| Auto-execute | ✅ Yes | ❌ No |
| Manual approval | ❌ No | ✅ Yes |
| Target visualization | ❌ No | ✅ Red ghost |
| Executed command viz | ✅ Purple ghost | ✅ Green ghost |
| Current state viz | ✅ Blue ghost | ✅ Blue ghost |

## Configuration File

File: `configs/franka/franka_robotiq_client_confirm.yaml`

Uses agent: `FrankaOscClientCartesianConfirmAgent` instead of `FrankaOscClientCartesianAgent`
