# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Closed-form geometric primitives for planar polygons.

Two segments ``e_i = [a_1, b_1]`` and ``e_j = [a_2, b_2]`` and a point ``p``
determine

    ell(e_i, e_j; p) = min { ||x - y|| : x in e_i, y in e_j, p in [x, y] }.

The value is evaluated exactly using two affine equations and one cubic, plus
a one-dimensional branch when ``p`` lies on the line of an edge, which a
nonconvex polygon can produce. Both branches are implemented here, so one
evaluation costs O(1)
instead of a nonlinear program.  The membership test
``p in conv(e_i union e_j)`` is a constant-time predicate as well: the join of
two segments is the convex hull of four points, and by Caratheodory a planar
point lies in it exactly when it lies in one of the triangles they span.

Everything below runs on scalars: these two predicates are the inner loop of
Algorithm 1, and array bookkeeping would dominate their cost. The coefficient
names ``alpha``, ``beta``, ``gamma``, ``mu``, and ``nu`` follow the closed-form
derivation.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from width_function.geometry.stationarity_polynomial import stationary_candidates

FloatArray = NDArray[np.float64]

Evaluation = tuple[float, "FloatArray | None", "FloatArray | None"]

_INFEASIBLE: Evaluation = (float("inf"), None, None)


def _coordinates(value: ArrayLike, name: str) -> tuple[float, float]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,):
        raise ValueError(f"{name} must be a finite vector in R^2")
    x, y = float(vector[0]), float(vector[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{name} must be a finite vector in R^2")
    return x, y


def _spread(points: tuple[tuple[float, float], ...]) -> float:
    """Characteristic length turning absolute tolerances into relative tests."""
    largest = 0.0
    for index, (x, y) in enumerate(points):
        for other_x, other_y in points[index + 1 :]:
            squared = (x - other_x) ** 2 + (y - other_y) ** 2
            largest = max(largest, squared)
    return math.sqrt(largest) if largest > 0.0 else 1.0


def _on_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
    threshold: float,
) -> bool:
    edge_x, edge_y = end[0] - start[0], end[1] - start[1]
    squared = edge_x * edge_x + edge_y * edge_y
    offset_x, offset_y = point[0] - start[0], point[1] - start[1]
    if squared <= threshold * threshold:
        return offset_x * offset_x + offset_y * offset_y <= threshold * threshold
    length = math.sqrt(squared)
    if abs(edge_x * offset_y - edge_y * offset_x) > threshold * length:
        return False
    margin = threshold / length
    return -margin <= (offset_x * edge_x + offset_y * edge_y) / squared <= 1.0 + margin


def _in_triangle(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
    point: tuple[float, float],
    threshold: float,
    area_threshold: float,
) -> bool:
    first_x, first_y = second[0] - first[0], second[1] - first[1]
    second_x, second_y = third[0] - second[0], third[1] - second[1]
    if abs(first_x * (third[1] - first[1]) - first_y * (third[0] - first[0])) <= area_threshold:
        return (
            _on_segment(first, second, point, threshold)
            or _on_segment(second, third, point, threshold)
            or _on_segment(first, third, point, threshold)
        )
    areas = (
        first_x * (point[1] - first[1]) - first_y * (point[0] - first[0]),
        second_x * (point[1] - second[1]) - second_y * (point[0] - second[0]),
        (first[0] - third[0]) * (point[1] - third[1])
        - (first[1] - third[1]) * (point[0] - third[0]),
    )
    return not (min(areas) < -area_threshold and max(areas) > area_threshold)


def point_in_join_2d(
    first_start: ArrayLike,
    first_end: ArrayLike,
    second_start: ArrayLike,
    second_end: ArrayLike,
    point: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Test ``p in conv(e_i union e_j)`` for two planar segments in O(1)."""
    corners = (
        _coordinates(first_start, "the first segment"),
        _coordinates(first_end, "the first segment"),
        _coordinates(second_start, "the second segment"),
        _coordinates(second_end, "the second segment"),
    )
    p = _coordinates(point, "point")
    threshold = tolerance * _spread((*corners, p))
    area_threshold = threshold * _spread(corners)
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(j + 1, 4):
                if _in_triangle(corners[i], corners[j], corners[k], p, threshold, area_threshold):
                    return True
    return False


def ell_segments_2d(
    first_start: ArrayLike,
    first_end: ArrayLike,
    second_start: ArrayLike,
    second_end: ArrayLike,
    point: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> Evaluation:
    """Return ``(ell, x, y)`` for two planar segments, or ``(inf, None, None)``.

    The returned pair satisfies ``x in e_i``, ``y in e_j``, ``p in [x, y]`` and
    ``||x - y|| = ell(e_i, e_j; p)``; the value is ``+inf`` exactly when no such
    pair exists.
    """
    a_1 = _coordinates(first_start, "the first segment")
    b_1 = _coordinates(first_end, "the first segment")
    a_2 = _coordinates(second_start, "the second segment")
    b_2 = _coordinates(second_end, "the second segment")
    p = _coordinates(point, "point")
    spread = _spread((a_1, b_1, a_2, b_2, p))

    first_length = math.hypot(b_1[0] - a_1[0], b_1[1] - a_1[1])
    second_length = math.hypot(b_2[0] - a_2[0], b_2[1] - a_2[1])
    if first_length <= tolerance * spread or second_length <= tolerance * spread:
        raise ValueError("both segments must be nondegenerate")

    # A vanishing dist(p, aff(e_j)) is the excluded case alpha = gamma.
    height_second = (
        abs((b_2[0] - a_2[0]) * (p[1] - a_2[1]) - (b_2[1] - a_2[1]) * (p[0] - a_2[0]))
        / second_length
    )
    height_first = (
        abs((b_1[0] - a_1[0]) * (p[1] - a_1[1]) - (b_1[1] - a_1[1]) * (p[0] - a_1[0]))
        / first_length
    )

    if height_second > tolerance * spread:
        return _ell_generic(a_1, b_1, a_2, b_2, p, tolerance, spread)
    if height_first > tolerance * spread:
        length, second_point, first_point = _ell_generic(a_2, b_2, a_1, b_1, p, tolerance, spread)
        return length, first_point, second_point
    return _ell_collinear(a_1, b_1, a_2, b_2, p, tolerance, spread)


def _ell_generic(
    a_1: tuple[float, float],
    b_1: tuple[float, float],
    a_2: tuple[float, float],
    b_2: tuple[float, float],
    p: tuple[float, float],
    tolerance: float,
    spread: float,
) -> Evaluation:
    """Candidate enumeration of Remark "Planar polygons"; needs ``p`` off ``aff(e_j)``."""
    first_x, first_y = b_1[0] - a_1[0], b_1[1] - a_1[1]
    second_x, second_y = b_2[0] - a_2[0], b_2[1] - a_2[1]
    offset_x, offset_y = p[0] - a_1[0], p[1] - a_1[1]
    gap_x, gap_y = a_2[0] - a_1[0], a_2[1] - a_1[1]

    # Use the quarter-turn convention x . y^perp = x_2 y_1 - x_1 y_2.
    alpha = offset_y * second_x - offset_x * second_y
    beta = second_y * first_x - second_x * first_y
    gamma = gap_y * second_x - gap_x * second_y
    mu = gap_y * offset_x - gap_x * offset_y
    nu = (p[1] - a_2[1]) * first_x - (p[0] - a_2[0]) * first_y

    eta_0 = offset_x * offset_x + offset_y * offset_y
    eta_1 = -2.0 * (offset_x * first_x + offset_y * first_y)
    eta_2 = first_x * first_x + first_y * first_y

    candidates = [0.0, 1.0]
    if nu != 0.0:
        candidates.append(-mu / nu)
    if nu != beta:
        candidates.append((alpha - mu) / (nu - beta))
    candidates.extend(stationary_candidates(alpha, beta, gamma, eta_0, eta_1, eta_2))

    denominator_floor = tolerance * spread * math.hypot(second_x, second_y)
    best = float("inf")
    best_pair: tuple[FloatArray, FloatArray] | None = None
    for raw in candidates:
        if not math.isfinite(raw) or raw < -tolerance or raw > 1.0 + tolerance:
            continue
        s = min(max(raw, 0.0), 1.0)
        denominator = alpha + beta * s
        if abs(denominator) <= denominator_floor:
            continue
        if (gamma + beta * s) / denominator < 1.0 - tolerance:
            continue
        edge_parameter = (mu + nu * s) / denominator
        if edge_parameter < -tolerance or edge_parameter > 1.0 + tolerance:
            continue
        t = min(max(edge_parameter, 0.0), 1.0)
        x = (a_1[0] + s * first_x, a_1[1] + s * first_y)
        y = (a_2[0] + t * second_x, a_2[1] + t * second_y)
        length = math.hypot(y[0] - x[0], y[1] - x[1])
        if length < best:
            best = length
            best_pair = (np.array(x), np.array(y))
    if best_pair is None:
        return _INFEASIBLE
    return best, best_pair[0], best_pair[1]


def _ell_collinear(
    a_1: tuple[float, float],
    b_1: tuple[float, float],
    a_2: tuple[float, float],
    b_2: tuple[float, float],
    p: tuple[float, float],
    tolerance: float,
    spread: float,
) -> Evaluation:
    """Branch of Remark "degenerate": ``p`` lies on the line of both edges.

    Every admissible segment then lies on that common line, so the two
    endpoints minimise independently on the two opposite rays issued from
    ``p``.  Noncollinear edges are infeasible here: the line through ``p`` and
    ``y`` would have to be both ``aff(e_i)`` and ``aff(e_j)``.
    """
    axis_x, axis_y = b_2[0] - a_2[0], b_2[1] - a_2[1]
    norm = math.hypot(axis_x, axis_y)
    axis_x, axis_y = axis_x / norm, axis_y / norm
    if abs(axis_x * (b_1[1] - a_1[1]) - axis_y * (b_1[0] - a_1[0])) > tolerance * spread:
        return _INFEASIBLE

    def coordinate(vertex: tuple[float, float]) -> float:
        return (vertex[0] - p[0]) * axis_x + (vertex[1] - p[1]) * axis_y

    threshold = tolerance * spread
    first_range = (coordinate(a_1), coordinate(b_1))
    second_range = (coordinate(a_2), coordinate(b_2))

    best = float("inf")
    best_pair: tuple[FloatArray, FloatArray] | None = None
    for sign in (1.0, -1.0):
        second = _closest_on_ray(second_range, sign, threshold)
        first = _closest_on_ray(first_range, -sign, threshold)
        if second is None or first is None:
            continue
        length = abs(second) + abs(first)
        if length < best:
            best = length
            best_pair = (
                np.array([p[0] + first * axis_x, p[1] + first * axis_y]),
                np.array([p[0] + second * axis_x, p[1] + second * axis_y]),
            )
    if best_pair is None:
        return _INFEASIBLE
    return best, best_pair[0], best_pair[1]


def _closest_on_ray(extremes: tuple[float, float], sign: float, threshold: float) -> float | None:
    """Coordinate of the point of a collinear segment nearest ``0`` on one ray."""
    low, high = min(extremes), max(extremes)
    if sign > 0.0:
        return max(low, 0.0) if high >= -threshold else None
    return min(high, 0.0) if low <= threshold else None
