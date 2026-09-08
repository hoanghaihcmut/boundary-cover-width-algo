# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Adaptive epsilon-approximation of W(Q; G)."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AdaptiveApproximationResult:
    """Result of certified or exploratory adaptive direction refinement."""

    epsilon: float
    theta_0: float
    width: float
    lower_bound: float
    upper_bound: float
    levels: NDArray[np.int64]
    point_lower_bounds: FloatArray
    point_upper_bounds: FloatArray
    point_best_directions: FloatArray
    refinements: int
    direction_evaluations: int
    converged: bool
    termination_reason: str

    @property
    def certified_gap(self) -> float:
        return self.upper_bound - self.lower_bound


def direction_set(level: int, theta_0: float = 0.0) -> FloatArray:
    """Return the nested direction set F(level), modulo pi."""
    if level < 1:
        raise ValueError("level must be at least 1")
    count = 1 << (level - 1)
    angles = theta_0 + np.arange(count) * np.pi / count
    return np.column_stack([np.cos(angles), np.sin(angles)])


def new_directions(level: int, theta_0: float = 0.0) -> FloatArray:
    """Return F(level) minus F(level - 1), for a refinement step."""
    if level < 2:
        raise ValueError("a refinement level must be at least 2")
    count = 1 << (level - 2)
    denominator = 1 << (level - 1)
    angles = theta_0 + (2 * np.arange(count) + 1) * np.pi / denominator
    return np.column_stack([np.cos(angles), np.sin(angles)])


def directional_width_polygon(
    polygon: ArrayLike,
    point: ArrayLike,
    direction: ArrayLike,
    *,
    tolerance: float = 1e-10,
) -> float:
    """Compute one directional width W^tau(point)."""
    widths = directional_widths_polygon(
        polygon,
        point,
        np.asarray(direction, dtype=float)[np.newaxis, :],
        tolerance=tolerance,
    )
    return float(widths[0])


def directional_widths_polygon(
    polygon: ArrayLike,
    point: ArrayLike,
    directions: ArrayLike,
    *,
    tolerance: float = 1e-10,
    batch_size: int = 65_536,
) -> FloatArray:
    """Compute directional widths for many directions using vectorised batches."""
    vertices = _polygon(polygon)
    p = _vector2(point, "point")
    tau = np.asarray(directions, dtype=float)
    if tau.ndim != 2 or tau.shape[1] != 2 or not len(tau) or not np.all(np.isfinite(tau)):
        raise ValueError("directions must be a nonempty finite (m, 2) array")
    norms = np.linalg.norm(tau, axis=1)
    if np.any(norms <= tolerance):
        raise ValueError("directions must be nonzero")
    edges = np.roll(vertices, -1, axis=0) - vertices
    return _directional_widths(
        vertices, edges, p, tau / norms[:, np.newaxis], tolerance, batch_size
    )


def _directional_widths(
    vertices: FloatArray,
    edges: FloatArray,
    p: FloatArray,
    tau: FloatArray,
    tolerance: float,
    batch_size: int,
) -> FloatArray:
    """Kernel of :func:`directional_widths_polygon` on validated, unit input.

    The refinement loop calls it once per level with the polygon edges already
    formed, so that no input is revalidated between levels.
    """
    output = np.empty(len(tau))
    offsets = vertices - p
    numerators = offsets[:, 0] * edges[:, 1] - offsets[:, 1] * edges[:, 0]
    for start in range(0, len(tau), batch_size):
        batch = tau[start : start + batch_size]
        denominator = (
            batch[:, 0, np.newaxis] * edges[np.newaxis, :, 1]
            - batch[:, 1, np.newaxis] * edges[np.newaxis, :, 0]
        )
        edge_numerator = (
            offsets[np.newaxis, :, 0] * batch[:, 1, np.newaxis]
            - offsets[np.newaxis, :, 1] * batch[:, 0, np.newaxis]
        )
        valid_denominator = np.abs(denominator) > tolerance
        edge_parameter = np.divide(
            edge_numerator,
            denominator,
            out=np.full_like(denominator, np.nan),
            where=valid_denominator,
        )
        valid = (
            valid_denominator & (edge_parameter >= -tolerance) & (edge_parameter <= 1.0 + tolerance)
        )
        ray_parameter = np.divide(
            numerators[np.newaxis, :],
            denominator,
            out=np.full_like(denominator, np.nan),
            where=valid_denominator,
        )
        positive = np.where(valid & (ray_parameter > tolerance), ray_parameter, np.inf)
        negative = np.where(valid & (ray_parameter < -tolerance), ray_parameter, -np.inf)
        forward = np.min(positive, axis=1)
        backward = np.max(negative, axis=1)
        if np.any(~np.isfinite(forward)) or np.any(~np.isfinite(backward)):
            raise ValueError("point must be strictly inside the polygon")
        output[start : start + len(batch)] = forward - backward
    return output


def adaptive_epsilon_width_2d(
    polygon: ArrayLike,
    query_points: ArrayLike,
    epsilon: float,
    *,
    theta_0: float = 0.0,
    tolerance: float = 1e-12,
    max_level: int = 30,
) -> AdaptiveApproximationResult:
    """Run the certified adaptive epsilon-approximation of W(Q; G)."""
    return _adaptive_direction_refinement_2d(
        polygon,
        query_points,
        epsilon,
        theta_0=theta_0,
        tolerance=tolerance,
        max_level=max_level,
        require_convex=True,
        max_runtime_seconds=None,
    )


def algorithm_2(
    polygon: ArrayLike,
    query_points: ArrayLike,
    epsilon: float,
    *,
    theta_0: float = 0.0,
    tolerance: float = 1e-12,
    max_level: int = 30,
) -> AdaptiveApproximationResult:
    """Return W_hat with 0 <= W_hat - W(Q; G) <= epsilon."""
    return adaptive_epsilon_width_2d(
        polygon,
        query_points,
        epsilon,
        theta_0=theta_0,
        tolerance=tolerance,
        max_level=max_level,
    )


def heuristic_adaptive_width_2d(
    polygon: ArrayLike,
    query_points: ArrayLike,
    epsilon: float,
    *,
    theta_0: float = 0.0,
    tolerance: float = 1e-12,
    max_level: int = 30,
    max_runtime_seconds: float | None = None,
) -> AdaptiveApproximationResult:
    """Apply the adaptive direction refinement heuristically to a simple polygon.

    The certified error bound requires convexity. For a nonconvex polygon,
    the returned lower bound and gap are only stopping surrogates and must not be
    interpreted as a certificate.  When ``max_runtime_seconds`` is given, the
    refinement stops between direction batches once that runtime is exceeded.
    """
    return _adaptive_direction_refinement_2d(
        polygon,
        query_points,
        epsilon,
        theta_0=theta_0,
        tolerance=tolerance,
        max_level=max_level,
        require_convex=False,
        max_runtime_seconds=max_runtime_seconds,
    )


def _adaptive_direction_refinement_2d(
    polygon: ArrayLike,
    query_points: ArrayLike,
    epsilon: float,
    *,
    theta_0: float,
    tolerance: float,
    max_level: int,
    require_convex: bool,
    max_runtime_seconds: float | None,
) -> AdaptiveApproximationResult:
    """Shared implementation for certified and exploratory refinement."""
    start_time = perf_counter()
    vertices = _polygon(polygon)
    if require_convex and not _is_convex(vertices, tolerance):
        raise ValueError("the adaptive algorithm requires a convex polygon")
    points = np.asarray(query_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("query_points must be a nonempty (n, 2) array")
    if not np.all(np.isfinite(points)):
        raise ValueError("query_points must be finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if max_runtime_seconds is not None and (
        not np.isfinite(max_runtime_seconds) or max_runtime_seconds <= 0.0
    ):
        raise ValueError("max_runtime_seconds must be positive and finite")

    # The polygon is validated once and its edges reused at every level, so that
    # the cost of a level is that of its new directions and nothing else.
    edges = np.roll(vertices, -1, axis=0) - vertices
    radii = _distances_to_boundary(vertices, points)
    if np.any(radii <= tolerance):
        raise ValueError("every query point must be strictly inside the polygon")
    outer_radii = np.max(
        np.linalg.norm(vertices[np.newaxis, :, :] - points[:, np.newaxis, :], axis=2), axis=1
    )
    levels = np.ones(len(points), dtype=np.int64)
    initial_direction = direction_set(1, theta_0)
    upper = _directional_widths_at_points(vertices, edges, points, initial_direction[0], 1e-10)
    best_directions = np.repeat(initial_direction, len(points), axis=0)
    theta = 2.0 * np.pi * outer_radii**2 / (radii * 2.0**levels)
    lower = np.maximum(2.0 * radii, upper - theta)
    direction_evaluations = len(points)
    refinements = 0

    global_lower = float(np.max(lower))
    global_upper = float(np.max(upper))
    timed_out = False
    while global_upper - global_lower > epsilon:
        if max_runtime_seconds is not None and perf_counter() - start_time > max_runtime_seconds:
            timed_out = True
            break
        index = int(np.argmax(upper))
        level = int(levels[index]) + 1
        if level > max_level:
            raise RuntimeError(f"adaptive approximation exceeded max_level={max_level}")
        deadline = None if max_runtime_seconds is None else start_time + max_runtime_seconds
        new_upper, best_direction, evaluated, completed = (
            _minimum_new_directional_width_until_deadline(
                vertices, edges, points[index], level, theta_0, deadline
            )
        )
        direction_evaluations += evaluated
        if new_upper < upper[index]:
            upper[index] = new_upper
            best_directions[index] = best_direction
        if not completed:
            global_upper = float(np.max(upper))
            timed_out = True
            break
        levels[index] = level
        theta_value = 2.0 * np.pi * outer_radii[index] ** 2 / (radii[index] * 2.0**level)
        lower[index] = max(2.0 * radii[index], upper[index] - theta_value)
        refinements += 1
        global_lower = float(np.max(lower))
        global_upper = float(np.max(upper))
        if max_runtime_seconds is not None and perf_counter() - start_time > max_runtime_seconds:
            timed_out = True
            break

    return AdaptiveApproximationResult(
        epsilon=float(epsilon),
        theta_0=float(theta_0),
        width=global_upper,
        lower_bound=global_lower,
        upper_bound=global_upper,
        levels=levels,
        point_lower_bounds=lower,
        point_upper_bounds=upper,
        point_best_directions=best_directions,
        refinements=refinements,
        direction_evaluations=direction_evaluations,
        converged=not timed_out,
        termination_reason="time_limit" if timed_out else "epsilon",
    )


def _minimum_new_directional_width_until_deadline(
    vertices: FloatArray,
    edges: FloatArray,
    point: FloatArray,
    level: int,
    theta_0: float,
    deadline: float | None,
    batch_size: int = 65_536,
) -> tuple[float, FloatArray, int, bool]:
    """Minimize over the new level directions, stopping between batches."""
    count = 1 << (level - 2)
    denominator = 1 << (level - 1)
    best_width = np.inf
    best_direction = np.zeros(2)
    evaluated = 0
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        indices = np.arange(start, stop)
        angles = theta_0 + (2 * indices + 1) * np.pi / denominator
        directions = np.column_stack([np.cos(angles), np.sin(angles)])
        widths = _directional_widths(vertices, edges, point, directions, 1e-10, batch_size)
        batch_index = int(np.argmin(widths))
        if widths[batch_index] < best_width:
            best_width = float(widths[batch_index])
            best_direction = directions[batch_index]
        evaluated = stop
        if deadline is not None and perf_counter() > deadline and stop < count:
            return best_width, best_direction, evaluated, False
    return best_width, best_direction, evaluated, True


def _directional_widths_at_points(
    vertices: FloatArray,
    edges: FloatArray,
    points: FloatArray,
    direction: FloatArray,
    tolerance: float,
) -> FloatArray:
    """One direction, every query point, in a single vectorised pass.

    The level-one pass of Algorithm 2 evaluates the same direction at all of
    ``G``, so it costs one array operation rather than ``|G|`` of them.
    """
    denominator = direction[0] * edges[:, 1] - direction[1] * edges[:, 0]
    valid_denominator = np.abs(denominator) > tolerance
    offsets = vertices[np.newaxis, :, :] - points[:, np.newaxis, :]
    edge_numerator = offsets[:, :, 0] * direction[1] - offsets[:, :, 1] * direction[0]
    numerators = (
        offsets[:, :, 0] * edges[np.newaxis, :, 1] - offsets[:, :, 1] * edges[np.newaxis, :, 0]
    )
    safe = np.where(valid_denominator, denominator, 1.0)[np.newaxis, :]
    edge_parameter = edge_numerator / safe
    ray_parameter = numerators / safe
    valid = (
        valid_denominator[np.newaxis, :]
        & (edge_parameter >= -tolerance)
        & (edge_parameter <= 1.0 + tolerance)
    )
    forward = np.min(np.where(valid & (ray_parameter > tolerance), ray_parameter, np.inf), axis=1)
    backward = np.max(
        np.where(valid & (ray_parameter < -tolerance), ray_parameter, -np.inf), axis=1
    )
    if np.any(~np.isfinite(forward)) or np.any(~np.isfinite(backward)):
        raise ValueError("point must be strictly inside the polygon")
    return forward - backward


def _distance_to_boundary(vertices: FloatArray, point: FloatArray) -> float:
    return float(_distances_to_boundary(vertices, np.asarray(point, dtype=float)[np.newaxis, :])[0])


def _distances_to_boundary(vertices: FloatArray, points: FloatArray) -> FloatArray:
    """Clearance of every query point, vectorised over points and edges."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    offsets = points[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    parameters = np.clip(
        (offsets * edges[np.newaxis, :, :]).sum(axis=2) / (edges * edges).sum(axis=1), 0.0, 1.0
    )
    residuals = offsets - parameters[:, :, np.newaxis] * edges[np.newaxis, :, :]
    return np.min(np.linalg.norm(residuals, axis=2), axis=1)


def _is_convex(vertices: FloatArray, tolerance: float) -> bool:
    first = np.roll(vertices, -1, axis=0) - vertices
    second = np.roll(vertices, -2, axis=0) - np.roll(vertices, -1, axis=0)
    crosses = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    nonzero = crosses[np.abs(crosses) > tolerance]
    return bool(len(nonzero) and (np.all(nonzero > 0.0) or np.all(nonzero < 0.0)))


def _polygon(polygon: ArrayLike) -> FloatArray:
    vertices = np.asarray(polygon, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("polygon must be an (m, 2) array with m >= 3")
    if np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("polygon must contain at least three finite vertices")
    if np.any(np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1) == 0.0):
        raise ValueError("polygon edges must be nondegenerate")
    return vertices


def _vector2(value: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector in R^2")
    return vector
