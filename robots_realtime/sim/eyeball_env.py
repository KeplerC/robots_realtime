"""MuJoCo environment with eyeball task system (dynamic scene via MjSpec).

Subclasses MuJoCoYAMEnv to support the eyeball task ABC from the vendored
``robots_realtime.sim.eyeball.tasks`` module.  Tasks dynamically add objects
to the MuJoCo scene via ``MjSpec`` before compilation, and provide randomized
eval configs, multi-stage success checking, and object placement.

Usage::

    from robots_realtime.sim.eyeball.tasks import TASK_REGISTRY
    from robots_realtime.sim.eyeball_env import MuJoCoEyeballEnv

    task = TASK_REGISTRY["tape_handover_random"]()
    env = MuJoCoEyeballEnv(
        task=task,
        eyeball_xml="robots_realtime/sim/eyeball/robot_xmls/eyeball_bimanual_scene.xml",
    )
    obs, _ = env.reset()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np

from robots_realtime.sim.config import SimConfig, CameraConfig, RobotConfig
from robots_realtime.sim.env import MuJoCoYAMEnv

logger = logging.getLogger(__name__)


def _eyeball_sim_config() -> SimConfig:
    """Default SimConfig for the eyeball bimanual scene.

    Uses the exo_camera (overhead view) and both wrist cameras, which are
    the most useful viewpoints for teleoperation and data collection.
    """
    return SimConfig(
        cameras={
            "exo_camera": CameraConfig(height=480, width=640, fps=30),
            "left_wrist_camera": CameraConfig(height=480, width=640, fps=30),
            "right_wrist_camera": CameraConfig(height=480, width=640, fps=30),
        },
        robots={
            "left": RobotConfig(
                root_pos=[0.0, 0.3048, 0.0],
                root_ori=[1.0, 0.0, 0.0, 0.0],
                init_q=[
                    -0.20656902,
                    0.47283894,
                    0.99431604,
                    -0.7043946,
                    -0.30842298,
                    -0.32864118,
                    0.9987507,
                ],
            ),
            "right": RobotConfig(
                root_pos=[0.0, -0.3048, 0.0],
                root_ori=[1.0, 0.0, 0.0, 0.0],
                init_q=[
                    0.20160982,
                    0.39005876,
                    1.1182956,
                    -0.8726253,
                    0.13332571,
                    0.42629892,
                    0.9895317,
                ],
            ),
        },
    )


class MuJoCoEyeballEnv(MuJoCoYAMEnv):
    """MuJoCo YAM env with eyeball task objects built via MjSpec.

    Overrides the static-XML loading of ``MuJoCoYAMEnv`` with the MjSpec
    pipeline used by the eyeball ``EvalEnv``:

        1. Load base XML via ``MjSpec.from_file``
        2. Task adds objects via ``task.configure_scene(spec)``
        3. Compile to ``MjModel`` with task objects included
        4. Task caches IDs via ``task.setup(model, data)``

    On each ``reset()``, a deterministic eval config is applied to randomize
    object positions, followed by warmup physics steps to let objects settle.

    Args:
        task: An eyeball ``Task`` instance (from ``robots_realtime.sim.eyeball.tasks``).
        eyeball_xml: Path to the eyeball base scene XML.
        eval_seed: Base seed for deterministic eval config generation.
        n_eval_configs: Number of configs to pre-generate.
        warmup_steps: Physics steps to run after placing objects (settling).
        config: SimConfig override (defaults to eyeball-appropriate cameras).
        **kwargs: Passed through to ``MuJoCoYAMEnv``.
    """

    def __init__(
        self,
        task: Any,
        eyeball_xml: str | Path,
        eval_seed: int = 0,
        n_eval_configs: int = 100,
        warmup_steps: int = 500,
        config: SimConfig | None = None,
        **kwargs,
    ):
        # Store task before super().__init__() calls _setup_model()
        self._eyeball_task = task
        self._warmup_steps = warmup_steps
        self._eval_seed = eval_seed
        self._n_eval_configs = n_eval_configs
        self._eval_configs: list[dict] | None = None
        self._config_idx = 0
        self._gripper_ctrl_max: float = 0.0475  # overwritten in _setup_model

        if config is None:
            config = _eyeball_sim_config()

        super().__init__(
            config=config,
            scene_xml=str(eyeball_xml),
            prompt=task.prompt,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Override: MjSpec pipeline instead of static XML
    # ------------------------------------------------------------------

    def _setup_model(self):
        spec = mujoco.MjSpec.from_file(str(self._scene_xml))
        self._eyeball_task.configure_scene(spec)

        # Task freejoints change qpos size — delete keyframes to avoid
        # compile errors (MuJoCo <3.5 enforces matching sizes).
        for k in list(spec.keys):
            if hasattr(k, "delete"):
                k.delete()

        self.model = spec.compile()
        self.model.opt.timestep = self._physics_dt
        self.data = mujoco.MjData(self.model)

        if self._render_cameras_flag:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self._camera_height,
                width=self._camera_width,
            )

        self._eyeball_task.setup(self.model, self.data)
        self._build_index_maps()

        # Detect gripper ctrl max from the compiled model so the [0,1]
        # agent mapping uses the correct range for this XML.
        for robot_name in self.robot_names:
            grip_act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot_name}_gripper"
            )
            if grip_act_id >= 0:
                finger_jnt = f"{robot_name}_left_finger"
                finger_jnt_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, finger_jnt
                )
                if finger_jnt_id >= 0:
                    self._gripper_ctrl_max = float(
                        self.model.jnt_range[finger_jnt_id, 1]
                    )
                break

        logger.info(
            "EyeballEnv compiled: %d bodies, %d actuators, gripper_max=%.4f",
            self.model.nbody,
            self.model.nu,
            self._gripper_ctrl_max,
        )

    # ------------------------------------------------------------------
    # Override: gripper mapping uses model-specific range
    # ------------------------------------------------------------------

    def _step_single(self, action_14d: np.ndarray) -> None:
        ctrl = np.zeros(self.model.nu)
        for i in range(self.single_timestep_action_dim):
            val = float(action_14d[i])
            if i in self._gripper_set:
                val = val * self._gripper_ctrl_max
            ctrl[self._ctrl_indices[i]] = val
        self.data.ctrl[:] = ctrl
        for _ in range(self._control_decimation):
            mujoco.mj_step(self.model, self.data)

    def get_obs(self) -> dict[str, Any]:
        state = np.zeros(self.state_dim, dtype=np.float32)
        for i, qpos_idx in enumerate(self._qpos_indices):
            val = float(self.data.qpos[qpos_idx])
            if i in self._gripper_set:
                val = float(np.clip(val / self._gripper_ctrl_max, 0.0, 1.0))
            state[i] = val

        if self._render_cameras_flag:
            images = self._render_cameras()
        else:
            images = {
                name: np.zeros(
                    (self._camera_height, self._camera_width, 3), dtype=np.uint8
                )
                for name in self.camera_names
            }

        sim_time = float(self.data.time)
        return {
            "images": images,
            "state": state,
            "prompt": self.prompt,
            "camera_timestamps": {name: sim_time for name in self.camera_names},
            "masks": {name: True for name in self.camera_names},
        }

    # ------------------------------------------------------------------
    # Override: reset applies task eval configs + warmup
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed, options=options)

        # Lazily generate eval configs
        if self._eval_configs is None:
            self._eval_configs = self._eyeball_task.generate_eval_configs(
                self._n_eval_configs, self._eval_seed
            )

        # Apply the next config (cycles through)
        config = self._eval_configs[self._config_idx % len(self._eval_configs)]
        self._config_idx += 1
        self._eyeball_task.apply_eval_config(self.model, self.data, config)

        # Warmup: let objects settle under gravity
        for _ in range(self._warmup_steps):
            mujoco.mj_step(self.model, self.data)
        self._eyeball_task.post_warmup(self.model, self.data, config)
        mujoco.mj_forward(self.model, self.data)

        self.cur_step = 0
        return self.get_obs(), {}

    # ------------------------------------------------------------------
    # Task success queries
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        """Check if the task goal has been achieved."""
        return self._eyeball_task.check_success(self.model, self.data)

    def check_stages(self) -> dict[str, bool]:
        """Check all task sub-stages. Returns ``{stage_name: achieved}``."""
        return self._eyeball_task.check_stages(self.model, self.data)

    @property
    def task_prompt(self) -> str:
        return self._eyeball_task.prompt

    @property
    def task_stages(self) -> tuple[str, ...]:
        return self._eyeball_task.stages

    @property
    def eyeball_task(self) -> Any:
        return self._eyeball_task
