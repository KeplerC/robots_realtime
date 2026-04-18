"""Poisson-disc pool for reusable position sampling."""

from __future__ import annotations

import numpy as np

from .polygon_utils import adaptive_poisson_disc


class PoissonPool:
    """Lazily-generated pool of Poisson-disc positions for a polygon region.

    Generates ``n_target`` well-spaced points inside the polygon on first
    use, then pops one per call.  Automatically refills when exhausted.
    """

    def __init__(
        self,
        polygon: np.ndarray,
        n_target: int,
        margin: float = 0.0,
    ):
        self._polygon = np.asarray(polygon, dtype=np.float64)
        self._n_target = n_target
        self._margin = margin
        self._pool: list[np.ndarray] = []

    def pop(self, rng: np.random.Generator) -> np.ndarray:
        """Return the next position, refilling the pool if empty."""
        if len(self._pool) == 0:
            self._refill(rng)
        return self._pool.pop()

    def _refill(self, rng: np.random.Generator):
        points, _, _ = adaptive_poisson_disc(
            self._polygon,
            n_target=self._n_target,
            rng=rng,
            margin=self._margin,
        )
        self._pool = list(points)


def generate_poisson_configs(
    polygons: dict[str, np.ndarray | tuple],
    radii: dict[str, float],
    n: int,
    seed: int = 0,
) -> list[dict[str, list[float]]]:
    """Pre-generate n Poisson-disc spawn configs.

    Each config maps object name to [x, y] coordinates.  One PoissonPool
    per object generates well-spaced positions; pools are paired by index.

    Parameters
    ----------
    polygons : Per-object spawn polygon vertices (world XY).
    radii : Per-object bounding radius used as margin.
    n : Number of configs to generate.
    seed : RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    pools = {
        name: PoissonPool(
            polygon=np.asarray(poly, dtype=np.float64),
            n_target=n,
            margin=radii[name],
        )
        for name, poly in polygons.items()
    }
    configs: list[dict[str, list[float]]] = []
    for _ in range(n):
        config = {}
        for name, pool in pools.items():
            xy = pool.pop(rng)
            config[name] = [float(xy[0]), float(xy[1])]
        configs.append(config)
    return configs
