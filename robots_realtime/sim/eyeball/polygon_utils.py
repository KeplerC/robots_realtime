"""Polygon-based spawn region utilities for MuJoCo tasks.

Provides uniform and Poisson-disc sampling within arbitrary 2D polygons,
orientation jitter, bounding radius computation, and collision-aware placement.
"""

from __future__ import annotations

import mujoco
import numpy as np
from matplotlib.path import Path


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _shoelace_area(vertices: np.ndarray) -> float:
    """Signed area of a simple polygon via the shoelace formula.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 2)

    Returns
    -------
    float  Signed area (positive for CCW winding).
    """
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_to_segments_distance_batch(
    points: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """Min distance from each point to the nearest polygon edge.

    Parameters
    ----------
    points : np.ndarray, shape (M, 2)
    vertices : np.ndarray, shape (N, 2)  (polygon vertices, auto-closed)

    Returns
    -------
    np.ndarray, shape (M,)  Minimum distance per point.
    """
    # Build segment start/end arrays (N edges for N vertices, closed)
    seg_start = vertices                          # (N, 2)
    seg_end = np.roll(vertices, -1, axis=0)       # (N, 2)

    # Vectorize over (M points) x (N segments)
    # p: (M, 1, 2), a: (1, N, 2), b: (1, N, 2)
    p = points[:, None, :]
    a = seg_start[None, :, :]
    b = seg_end[None, :, :]

    ab = b - a                                     # (1, N, 2)
    ap = p - a                                     # (M, N, 2)

    # Project parameter t, clamp to [0, 1]
    t = np.sum(ap * ab, axis=-1) / (np.sum(ab * ab, axis=-1) + 1e-12)
    t = np.clip(t, 0.0, 1.0)                      # (M, N)

    # Closest point on each segment
    closest = a + t[..., None] * ab                # (M, N, 2)
    dist = np.linalg.norm(p - closest, axis=-1)    # (M, N)

    return dist.min(axis=1)                        # (M,)


# ---------------------------------------------------------------------------
# Polygon sampling
# ---------------------------------------------------------------------------

def sample_point_uniform_in_polygon(
    vertices: np.ndarray,
    rng: np.random.Generator,
    margin: float = 0.0,
    max_attempts: int = 1000,
    label: str = "",
) -> np.ndarray:
    """Sample a uniformly random (x, y) point inside a polygon.

    When *margin* > 0 the sampled point must be at least *margin* metres
    from every polygon edge, ensuring an object of that bounding radius
    fits entirely inside the polygon.

    Uses batch rejection sampling (64 candidates per round).

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 2)
    rng : np.random.Generator
    margin : float  Inset distance in metres.
    max_attempts : int  Total candidate budget before raising.
    label : str  Object/polygon label for error messages.

    Returns
    -------
    np.ndarray, shape (2,)

    Raises
    ------
    RuntimeError  If no valid point found within *max_attempts*.
    """
    path = Path(vertices)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    batch_size = 64

    attempts = 0
    while attempts < max_attempts:
        n = min(batch_size, max_attempts - attempts)
        candidates = np.column_stack([
            rng.uniform(bbox_min[0], bbox_max[0], size=n),
            rng.uniform(bbox_min[1], bbox_max[1], size=n),
        ])

        inside = path.contains_points(candidates)
        if margin > 0:
            dists = point_to_segments_distance_batch(candidates, vertices)
            inside &= dists >= margin

        valid = np.where(inside)[0]
        if len(valid) > 0:
            return candidates[valid[0]]

        attempts += n

    raise RuntimeError(
        f"Failed to sample point in polygon after {max_attempts} attempts. "
        f"label={label!r}, margin={margin:.4f}, "
        f"vertices={vertices.tolist()}"
    )


# Backward-compatible alias
sample_point_in_polygon = sample_point_uniform_in_polygon


# ---------------------------------------------------------------------------
# Poisson-disc sampling (Bridson's algorithm)
# ---------------------------------------------------------------------------

def bridson_poisson_disc_polygon(
    vertices: np.ndarray,
    min_dist: float,
    rng: np.random.Generator,
    k: int = 30,
    margin: float = 0.0,
) -> np.ndarray:
    """Generate a Poisson-disc point set inside a polygon using Bridson's algorithm.

    Points are guaranteed to be at least *min_dist* apart from each other
    and at least *margin* from every polygon edge.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 2)  Polygon boundary.
    min_dist : float  Minimum distance between any two points.
    rng : np.random.Generator
    k : int  Candidates tested per active point before removal.
    margin : float  Inset distance from polygon edges.

    Returns
    -------
    np.ndarray, shape (M, 2)  Well-spaced points inside the polygon.
    """
    path = Path(vertices)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    # Grid for spatial acceleration
    cell_size = min_dist / np.sqrt(2)
    grid_w = max(1, int(np.ceil((bbox_max[0] - bbox_min[0]) / cell_size)))
    grid_h = max(1, int(np.ceil((bbox_max[1] - bbox_min[1]) / cell_size)))
    grid: dict[tuple[int, int], int] = {}  # cell -> point index

    def _to_grid(pt: np.ndarray) -> tuple[int, int]:
        return (
            int((pt[0] - bbox_min[0]) / cell_size),
            int((pt[1] - bbox_min[1]) / cell_size),
        )

    def _is_valid(pt: np.ndarray) -> bool:
        # Check polygon containment + margin
        if not path.contains_point(pt):
            return False
        if margin > 0:
            d = point_to_segments_distance_batch(pt.reshape(1, 2), vertices)[0]
            if d < margin:
                return False
        # Check min_dist from neighbors via grid
        gi, gj = _to_grid(pt)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni, nj = gi + di, gj + dj
                if (ni, nj) in grid:
                    idx = grid[(ni, nj)]
                    if np.linalg.norm(pt - points[idx]) < min_dist:
                        return False
        return True

    # Seed point: sample uniformly inside inset polygon
    points: list[np.ndarray] = []
    seed_pt = sample_point_uniform_in_polygon(
        vertices, rng, margin=margin, label="bridson_seed",
    )
    points.append(seed_pt)
    grid[_to_grid(seed_pt)] = 0
    active = [0]

    while active:
        idx = rng.integers(len(active))
        active_idx = active[idx]
        center = points[active_idx]

        found = False
        for _ in range(k):
            # Random point in annulus [min_dist, 2*min_dist]
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(min_dist, 2 * min_dist)
            candidate = center + r * np.array([np.cos(angle), np.sin(angle)])

            if _is_valid(candidate):
                pt_idx = len(points)
                points.append(candidate)
                grid[_to_grid(candidate)] = pt_idx
                active.append(pt_idx)
                found = True
                break

        if not found:
            active.pop(idx)

    return np.array(points) if points else np.empty((0, 2))


def adaptive_poisson_disc(
    vertices: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
    margin: float = 0.0,
    k: int = 30,
    decay: float = 0.9,
    floor: float = 0.001,
    bisect_iters: int = 8,
) -> tuple[np.ndarray, float, int]:
    """Generate exactly *n_target* maximally-spaced Poisson-disc points.

    Two phases:

    1. **Coarse search** — starts from the polygon bounding-box diagonal
       and decays by *decay* until Bridson produces >= *n_target* points.
       This brackets the ideal ``min_dist`` between two consecutive values.

    2. **Binary search** — refines within the bracket to find the largest
       ``min_dist`` that still yields >= *n_target* points.  This ensures
       maximum spacing for the requested count.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 2)
    n_target : int  Desired number of points.
    rng : np.random.Generator
    margin : float  Inset from polygon edges.
    k : int  Bridson candidates per active point.
    decay : float  Multiplicative factor for coarse min_dist reduction.
    floor : float  Minimum allowed min_dist before giving up.
    bisect_iters : int  Binary search iterations after coarse bracket.

    Returns
    -------
    points : np.ndarray, shape (n_target, 2)  Maximally-spaced point set.
    final_min_dist : float  The min_dist that produced the returned set.
    n_retries : int  Number of coarse decay steps taken.
    """
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))

    # Phase 1: coarse decay to find bracket [lo, hi] where lo yields
    # >= n_target and hi yields < n_target.
    hi_dist = bbox_diag  # too large — yields ~1 point
    lo_dist = None
    lo_points = np.empty((0, 2))
    n_retries = 0
    best_points = np.empty((0, 2))
    best_dist = bbox_diag

    min_dist = bbox_diag
    while min_dist >= floor:
        points = bridson_poisson_disc_polygon(
            vertices, min_dist, rng, k=k, margin=margin,
        )
        if len(points) > len(best_points):
            best_points = points
            best_dist = min_dist
        if len(points) >= n_target:
            lo_dist = min_dist
            lo_points = points
            break
        hi_dist = min_dist
        min_dist *= decay
        n_retries += 1

    if lo_dist is None:
        # Never reached n_target — return best effort
        return best_points, best_dist, n_retries

    # Phase 2: binary search between hi_dist (< n_target) and lo_dist
    # (>= n_target) to find the largest min_dist yielding >= n_target.
    for _ in range(bisect_iters):
        mid = (lo_dist + hi_dist) / 2
        points = bridson_poisson_disc_polygon(
            vertices, mid, rng, k=k, margin=margin,
        )
        if len(points) >= n_target:
            lo_dist = mid
            lo_points = points
        else:
            hi_dist = mid

    # lo_points is from the largest min_dist that yields >= n_target.
    # Trim to exactly n_target (overshoot should be minimal after bisect).
    rng.shuffle(lo_points)
    return lo_points[:n_target], lo_dist, n_retries


def rpy_jitter_quat(
    rpy_max_deg: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a jittered orientation quaternion around upright.

    Samples roll, pitch, yaw independently from Uniform(-max, +max) and
    composes the resulting rotation on top of the identity (upright)
    orientation.

    Convention
    ----------
    - Euler angles: **intrinsic xyz** (rotate about body-x, then body-y,
      then body-z).
    - Quaternion format: **wxyz** (MuJoCo convention).
    - Output is explicitly normalised.

    Parameters
    ----------
    rpy_max_deg : tuple of 3 floats  (roll_max, pitch_max, yaw_max) in degrees.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (4,)  wxyz quaternion.
    """
    roll_max, pitch_max, yaw_max = rpy_max_deg

    # Sample angles in radians
    roll = np.deg2rad(rng.uniform(-roll_max, roll_max)) if roll_max > 0 else 0.0
    pitch = np.deg2rad(rng.uniform(-pitch_max, pitch_max)) if pitch_max > 0 else 0.0
    yaw = np.deg2rad(rng.uniform(-yaw_max, yaw_max)) if yaw_max > 0 else 0.0

    # Build rotation matrices for intrinsic xyz
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rx * Ry * Rz  (intrinsic xyz = extrinsic zyx)
    R = np.array([
        [cp * cy,                  cp * sy,                 -sp],
        [sr * sp * cy - cr * sy,   sr * sp * sy + cr * cy,   sr * cp],
        [cr * sp * cy + sr * sy,   cr * sp * sy - sr * cy,   cr * cp],
    ])

    # Rotation matrix -> wxyz quaternion
    quat = _rotmat_to_quat_wxyz(R)

    # Normalise for safety
    quat /= np.linalg.norm(quat) + 1e-12
    return quat


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion.

    Uses Shepperd's method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_spawn_config(
    objects: dict,
    polygons: dict[str, tuple[tuple[float, float], ...]],
    jitter: dict[str, tuple[float, float, float]],
) -> None:
    """Validate that spawn config is consistent with OBJECTS.

    Checks that every object has a polygon (>= 3 vertices, non-zero area)
    and an RPY jitter entry.  Raises ValueError on any mismatch.
    """
    for obj_name in objects:
        if obj_name not in polygons:
            raise ValueError(
                f"Object {obj_name!r} missing from spawn_polygons_xy. "
                f"Available keys: {list(polygons.keys())}"
            )
        if obj_name not in jitter:
            raise ValueError(
                f"Object {obj_name!r} missing from rpy_jitter_deg. "
                f"Available keys: {list(jitter.keys())}"
            )

        verts = polygons[obj_name]
        if len(verts) < 3:
            raise ValueError(
                f"Polygon for {obj_name!r} has {len(verts)} vertices "
                f"(need >= 3)."
            )

        area = abs(_shoelace_area(np.array(verts)))
        if area < 1e-8:
            raise ValueError(
                f"Polygon for {obj_name!r} has near-zero area ({area:.2e}). "
                f"Vertices: {verts}"
            )


# ---------------------------------------------------------------------------
# Bounding radius
# ---------------------------------------------------------------------------

def compute_body_bounding_radius_xy(
    mjm: mujoco.MjModel,
    body_id: int,
) -> float:
    """Conservative XY bounding radius for all geoms attached to a body.

    For each geom owned by *body_id*, computes the XY offset of the geom
    from the body origin plus the geom's bounding sphere radius
    (``mjm.geom_rbound``).  Returns the maximum across all geoms.

    This is a conservative 3D bound applied to XY (slight overestimate, safe).
    """
    max_r = 0.0
    for gid in range(mjm.ngeom):
        if mjm.geom_bodyid[gid] != body_id:
            continue
        # geom_pos is relative to parent body
        pos_xy = mjm.geom_pos[gid, :2]
        xy_offset = float(np.linalg.norm(pos_xy))
        rbound = float(mjm.geom_rbound[gid])
        max_r = max(max_r, xy_offset + rbound)
    return max_r


# ---------------------------------------------------------------------------
# Collision-aware placement
# ---------------------------------------------------------------------------

def _check_collision(
    mjm: mujoco.MjModel,
    mjd: mujoco.MjData,
    body_id: int,
    placed_body_ids: set[int],
) -> bool:
    """Return True if *body_id* is in contact with any body in *placed_body_ids*.

    Resolves body IDs via ``mjm.geom_bodyid`` (never geom names).
    """
    for i in range(mjd.ncon):
        c = mjd.contact[i]
        b1 = int(mjm.geom_bodyid[c.geom1])
        b2 = int(mjm.geom_bodyid[c.geom2])

        if b1 == body_id and b2 in placed_body_ids:
            return True
        if b2 == body_id and b1 in placed_body_ids:
            return True
    return False


def _is_inside_inset_polygon(
    xy: np.ndarray,
    polygon_verts: np.ndarray,
    margin: float,
    path: Path,
) -> bool:
    """Check if a single (x, y) point is inside the inset polygon."""
    pt = xy.reshape(1, 2)
    if not path.contains_point(xy):
        return False
    if margin > 0:
        dist = point_to_segments_distance_batch(pt, polygon_verts)[0]
        if dist < margin:
            return False
    return True


def resolve_spawn_collision(
    mjm: mujoco.MjModel,
    mjd: mujoco.MjData,
    body_id: int,
    jnt_adr: int,
    spawn_z: float,
    polygon_verts: np.ndarray,
    margin: float,
    placed_body_ids: set[int],
    rng: np.random.Generator,
    dr: float = 0.005,
    max_r: float = 0.08,
    n_angles: int = 32,
    max_resamples: int = 10,
    label: str = "",
) -> float:
    """Resolve collisions between a newly placed object and previously placed ones.

    After placing an object at its sampled position, runs ``mj_forward()``
    and checks for contacts with bodies in *placed_body_ids* only (ignores
    robot, table, etc.).

    If a collision is detected:

    1. **Ring search** around the original (x, y): tries concentric rings
       at radii ``dr, 2*dr, ..., max_r`` with *n_angles* equally spaced
       angles per ring.  Each candidate must be inside the inset polygon
       (margin check) and collision-free.  Orientation is preserved.

    2. If the ring search exhausts all candidates, **resamples** from the
       full polygon and retries (up to *max_resamples* times).

    Raises ``RuntimeError`` if no valid placement is found.

    Parameters
    ----------
    mjm, mjd : MuJoCo model and data.
    body_id : int  Body ID of the object being placed.
    jnt_adr : int  Joint qpos address for the object's freejoint.
    spawn_z : float  Fixed z height.
    polygon_verts : np.ndarray (N, 2)  Polygon vertices.
    margin : float  Inset distance.
    placed_body_ids : set of int  Bodies already placed.
    rng : np.random.Generator
    dr, max_r, n_angles : Ring search parameters.
    max_resamples : int  Number of full resamples before giving up.
    label : str  Object label for error messages.

    Returns
    -------
    float  Displacement distance from the original sampled position
           (0.0 if no collision was detected).
    """
    original_xy = mjd.qpos[jnt_adr : jnt_adr + 2].copy()

    if not placed_body_ids:
        # First object — nothing to collide with.
        mujoco.mj_forward(mjm, mjd)
        return 0.0

    path = Path(polygon_verts)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    for resample_i in range(max_resamples):
        # Forward to detect contacts
        mujoco.mj_forward(mjm, mjd)
        if not _check_collision(mjm, mjd, body_id, placed_body_ids):
            final_xy = mjd.qpos[jnt_adr : jnt_adr + 2]
            return float(np.linalg.norm(final_xy - original_xy))

        # Ring search
        search_center = mjd.qpos[jnt_adr : jnt_adr + 2].copy()
        found = False
        r = dr
        while r <= max_r + 1e-9:
            for angle in angles:
                candidate = search_center + r * np.array([np.cos(angle), np.sin(angle)])
                if not _is_inside_inset_polygon(candidate, polygon_verts, margin, path):
                    continue

                mjd.qpos[jnt_adr : jnt_adr + 2] = candidate
                mujoco.mj_forward(mjm, mjd)
                if not _check_collision(mjm, mjd, body_id, placed_body_ids):
                    found = True
                    break
            if found:
                final_xy = mjd.qpos[jnt_adr : jnt_adr + 2]
                return float(np.linalg.norm(final_xy - original_xy))
            r += dr

        # Ring search failed — resample from polygon
        xy = sample_point_in_polygon(
            polygon_verts, rng, margin=margin, label=label,
        )
        mjd.qpos[jnt_adr : jnt_adr + 2] = xy

    # Final check after last resample
    mujoco.mj_forward(mjm, mjd)
    if not _check_collision(mjm, mjd, body_id, placed_body_ids):
        final_xy = mjd.qpos[jnt_adr : jnt_adr + 2]
        return float(np.linalg.norm(final_xy - original_xy))

    raise RuntimeError(
        f"Failed to resolve spawn collision for {label!r} after "
        f"{max_resamples} resamples with ring search "
        f"(dr={dr}, max_r={max_r}, n_angles={n_angles}). "
        f"polygon={polygon_verts.tolist()}, margin={margin:.4f}"
    )
