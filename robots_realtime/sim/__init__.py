"""Minimal self-contained MuJoCo simulation for the bimanual YAM robot.

Does not depend on the private xdof-sim package. Assets (XML + STL meshes)
are bundled under robots_realtime/sim/models/.

Quick start:
    import robots_realtime.sim as sim
    env = sim.make_env(scene="hybrid")
    obs, _ = env.reset()
    obs, history, *_ = env.step(env.action_space.sample())
"""

from robots_realtime.sim.env import MuJoCoYAMEnv
from robots_realtime.sim.config import SimConfig, default_sim_config, flat_init_config
from robots_realtime.sim.scene_variants import apply_scene_variant, list_variants


def make_env(
    scene_variant: str = "hybrid",
    task: str = "bottles",
    render_cameras: bool = True,
    prompt: str = "",
    chunk_dim: int = 30,
    config: SimConfig | None = None,
    **kwargs,
) -> MuJoCoYAMEnv:
    """Create and configure a MuJoCo YAM environment.

    Args:
        scene_variant: Visual variant — "eval", "training", or "hybrid".
        task: Scene XML to load — currently "bottles".
        render_cameras: Whether to render camera images in observations.
        prompt: Task description included in obs["prompt"].
        chunk_dim: Timesteps per action chunk.
        config: Optional SimConfig; defaults to default_sim_config().
        **kwargs: Passed through to MuJoCoYAMEnv.
    """
    env = MuJoCoYAMEnv(
        config=config,
        scene=task,
        render_cameras=render_cameras,
        prompt=prompt,
        chunk_dim=chunk_dim,
        **kwargs,
    )
    apply_scene_variant(env.model, scene_variant)
    return env


def make_eyeball_env(
    task_name: str,
    eyeball_xml: str | None = None,
    render_cameras: bool = True,
    config: SimConfig | None = None,
    eval_seed: int = 0,
    n_eval_configs: int = 100,
    warmup_steps: int = 500,
    **kwargs,
) -> "MuJoCoEyeballEnv":
    """Create a MuJoCo YAM environment with an eyeball task.

    Requires the ``eye`` package (``pip install -e /path/to/eyeball``).

    Args:
        task_name: Key in ``robots_realtime.sim.eyeball.tasks.TASK_REGISTRY``
            (e.g. "tape_handover_random", "pick_up_tiger").
        eyeball_xml: Path to base scene XML. Defaults to the standard
            ``eyeball_bimanual_scene.xml`` inside the eye package.
        render_cameras: Whether to render camera images in observations.
        config: Optional SimConfig; defaults to eyeball cameras.
        eval_seed: Base seed for reproducible eval configs.
        n_eval_configs: Number of configs to pre-generate.
        warmup_steps: Physics steps for object settling after placement.
        **kwargs: Passed through to MuJoCoEyeballEnv.
    """
    from robots_realtime.sim.eyeball.tasks import TASK_REGISTRY

    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown eyeball task '{task_name}'. "
            f"Available: {sorted(TASK_REGISTRY.keys())}"
        )

    task_cls = TASK_REGISTRY[task_name]
    task = task_cls() if callable(task_cls) else task_cls

    if eyeball_xml is None:
        from pathlib import Path
        eyeball_xml = str(
            Path(__file__).parent
            / "eyeball"
            / "robot_xmls"
            / "eyeball_bimanual_scene.xml"
        )

    from robots_realtime.sim.eyeball_env import MuJoCoEyeballEnv

    return MuJoCoEyeballEnv(
        task=task,
        eyeball_xml=eyeball_xml,
        render_cameras=render_cameras,
        config=config,
        eval_seed=eval_seed,
        n_eval_configs=n_eval_configs,
        warmup_steps=warmup_steps,
        **kwargs,
    )


__all__ = [
    "MuJoCoYAMEnv",
    "SimConfig",
    "default_sim_config",
    "flat_init_config",
    "apply_scene_variant",
    "list_variants",
    "make_env",
    "make_eyeball_env",
]
