# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Closed-form geometric primitives for triangulated polyhedra in R^3.

Algorithm 1 needs three predicates per pair of boundary pieces: the distance
``dist(p, T)``, the membership test ``p in conv(T_i union T_j)``, and the exact
value

    ell(T_i, T_j; p) = min { ||u - v|| : u in T_i, v in T_j, p in [u, v] }.

All three are available in closed form for triangles.

The third one reduces to the planar construction.  Write ``n_1``, ``n_2`` for
the two unit facet normals and ``d`` for the direction of the chord, so that
``u = p - s d`` and ``v = p + t d`` with

    s = alpha_1 / (n_1 . d),   t = -alpha_2 / (n_2 . d),
    alpha_r = n_r . (p - a_r),   ell = s + t .

Each barycentric constraint on ``u`` or ``v`` becomes a *linear* inequality in
``d`` once the denominators are cleared, so the admissible directions form a
spherical polygon and the minimiser lies in one of three strata:

* both endpoints interior --- the gradient of ``s + t`` is a combination of
  ``n_1`` and ``n_2``, so ``d`` lies in ``span{n_1, n_2}``; equivalently the
  orthogonal projection along ``n_1 x n_2`` shortens every chord and turns the
  two planes into two lines, which is the planar problem;
* one endpoint on an edge --- the chord is parametrised by that edge, and the
  stationarity condition is the *same* cubic as in the plane;
* both endpoints on edges --- finitely many chords, one per pair of edges.

Enumerating the three strata gives at most forty-three candidate directions,
each obtained from a linear or a cubic equation.  Everything below runs on
scalars: these predicates are the inner loop of Algorithm 1, and array
bookkeeping would dominate their cost.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from width_function.geometry.stationarity_polynomial import real_roots, stationary_candidates

FloatArray = NDArray[np.float64]

Evaluation = tuple[float, "FloatArray | None", "FloatArray | None"]
Vector = tuple[float, float, float]

_INFEASIBLE: Evaluation = (float("inf"), None, None)


def _vertices(value: ArrayLike, name: str) -> tuple[Vector, Vector, Vector]:
    points = np.asarray(value, dtype=float)
    if points.shape != (3, 3) or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be three finite points of R^3")
    return tuple(tuple(float(x) for x in row) for row in points)  # type: ignore[return-value]


def _coordinates(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector in R^3")
    return float(vector[0]), float(vector[1]), float(vector[2])


def _subtract(left: Vector, right: Vector) -> Vector:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def _dot(left: Vector, right: Vector) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Vector, right: Vector) -> Vector:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _spread(points: tuple[Vector, ...]) -> float:
    largest = 0.0
    for index, first in enumerate(points):
        for second in points[index + 1 :]:
            squared = (
                (first[0] - second[0]) ** 2
                + (first[1] - second[1]) ** 2
                + (first[2] - second[2]) ** 2
            )
            largest = max(largest, squared)
    return math.sqrt(largest) if largest > 0.0 else 1.0


def point_triangle_distance(vertices: ArrayLike, point: ArrayLike) -> float:
    """Euclidean distance from a point to a triangle, by Voronoi region."""
    a, b, c = _vertices(vertices, "the triangle")
    p = _coordinates(point, "point")
    ab, ac, ap = _subtract(b, a), _subtract(c, a), _subtract(p, a)
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return math.sqrt(_dot(ap, ap))
    bp = _subtract(p, b)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return math.sqrt(_dot(bp, bp))
    weight_c = d1 * d4 - d3 * d2
    if weight_c <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return _distance_to_offset(ap, ab, d1 / (d1 - d3))
    cp = _subtract(p, c)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return math.sqrt(_dot(cp, cp))
    weight_b = d5 * d2 - d1 * d6
    if weight_b <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return _distance_to_offset(ap, ac, d2 / (d2 - d6))
    weight_a = d3 * d6 - d5 * d4
    if weight_a <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return _distance_to_offset(bp, _subtract(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)))
    total = weight_a + weight_b + weight_c
    first, second = weight_b / total, weight_c / total
    residual = (
        ap[0] - first * ab[0] - second * ac[0],
        ap[1] - first * ab[1] - second * ac[1],
        ap[2] - first * ab[2] - second * ac[2],
    )
    return math.sqrt(_dot(residual, residual))


def _distance_to_offset(offset: Vector, edge: Vector, parameter: float) -> float:
    residual = (
        offset[0] - parameter * edge[0],
        offset[1] - parameter * edge[1],
        offset[2] - parameter * edge[2],
    )
    return math.sqrt(_dot(residual, residual))


def point_in_join_3d(
    first: ArrayLike,
    second: ArrayLike,
    point: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Test ``p in conv(T_i union T_j)`` for two triangles of R^3 in O(1).

    The join is the convex hull of at most six points, so by Caratheodory the
    test reduces to a fixed number of simplices spanned by those points.
    """
    corners = (*_vertices(first, "the first triangle"), *_vertices(second, "the second triangle"))
    p = _coordinates(point, "point")
    threshold = tolerance * _spread((*corners, p))
    unique = _distinct(corners, threshold)
    count = len(unique)
    if count == 1:
        return _distance(unique[0], p) <= threshold
    if count == 2:
        return _on_segment(unique[0], unique[1], p, threshold)

    volume_threshold = threshold**3
    full_dimensional = False
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                for m in range(k + 1, count):
                    inside, degenerate = _in_simplex(
                        unique[i], unique[j], unique[k], unique[m], p, threshold, volume_threshold
                    )
                    if inside:
                        return True
                    full_dimensional |= not degenerate
    if full_dimensional:
        return False
    return _in_flat_hull(unique, p, threshold)


def _in_simplex(
    first: Vector,
    second: Vector,
    third: Vector,
    fourth: Vector,
    p: Vector,
    threshold: float,
    volume_threshold: float,
) -> tuple[bool, bool]:
    """Barycentric test against one tetrahedron; also reports degeneracy."""
    u = _subtract(second, first)
    v = _subtract(third, first)
    w = _subtract(fourth, first)
    determinant = _dot(u, _cross(v, w))
    if abs(determinant) <= volume_threshold:
        return False, True
    offset = _subtract(p, first)
    first_weight = _dot(offset, _cross(v, w)) / determinant
    second_weight = _dot(u, _cross(offset, w)) / determinant
    third_weight = _dot(u, _cross(v, offset)) / determinant
    inside = (
        first_weight >= -threshold
        and second_weight >= -threshold
        and third_weight >= -threshold
        and first_weight + second_weight + third_weight <= 1.0 + threshold
    )
    return inside, False


def _in_flat_hull(points: tuple[Vector, ...], p: Vector, threshold: float) -> bool:
    """Membership when every four points are coplanar: one dimension lower."""
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                a, b, c = points[i], points[j], points[k]
                u, v = _subtract(b, a), _subtract(c, a)
                normal = _cross(u, v)
                area = math.sqrt(_dot(normal, normal))
                if area <= threshold * threshold:
                    continue
                offset = _subtract(p, a)
                if abs(_dot(offset, normal)) > threshold * area:
                    continue
                first = _dot(_cross(offset, v), normal) / _dot(normal, normal)
                second = _dot(_cross(u, offset), normal) / _dot(normal, normal)
                if (
                    first >= -threshold
                    and second >= -threshold
                    and first + second <= 1.0 + threshold
                ):
                    return True
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _on_segment(points[i], points[j], p, threshold):
                return True
    return False


def _distinct(points: tuple[Vector, ...], threshold: float) -> tuple[Vector, ...]:
    kept: list[Vector] = []
    for point in points:
        if all(_distance(point, other) > threshold for other in kept):
            kept.append(point)
    return tuple(kept)


def _distance(left: Vector, right: Vector) -> float:
    offset = _subtract(left, right)
    return math.sqrt(_dot(offset, offset))


def _on_segment(start: Vector, end: Vector, p: Vector, threshold: float) -> bool:
    edge = _subtract(end, start)
    squared = _dot(edge, edge)
    if squared <= threshold * threshold:
        return _distance(start, p) <= threshold
    parameter = min(max(_dot(_subtract(p, start), edge) / squared, 0.0), 1.0)
    return _distance_to_offset(_subtract(p, start), edge, parameter) <= threshold


class _Prepared:
    """A triangle with the quantities the candidate loop reuses."""

    __slots__ = ("first", "inverse", "normal", "origin", "second")

    def __init__(self, vertices: tuple[Vector, Vector, Vector]) -> None:
        a, b, c = vertices
        self.origin = a
        self.first = _subtract(b, a)
        self.second = _subtract(c, a)
        gram_11 = _dot(self.first, self.first)
        gram_12 = _dot(self.first, self.second)
        gram_22 = _dot(self.second, self.second)
        determinant = gram_11 * gram_22 - gram_12 * gram_12
        self.inverse = (gram_22 / determinant, -gram_12 / determinant, gram_11 / determinant)
        normal = _cross(self.first, self.second)
        area = math.sqrt(_dot(normal, normal))
        self.normal = (normal[0] / area, normal[1] / area, normal[2] / area)

    def contains(self, point: Vector, tolerance: float) -> bool:
        offset = _subtract(point, self.origin)
        d1, d2 = _dot(self.first, offset), _dot(self.second, offset)
        i11, i12, i22 = self.inverse
        first = i11 * d1 + i12 * d2
        second = i12 * d1 + i22 * d2
        return first >= -tolerance and second >= -tolerance and first + second <= 1.0 + tolerance

    def vertex(self, index: int) -> Vector:
        if index == 0:
            return self.origin
        edge = self.first if index == 1 else self.second
        return (
            self.origin[0] + edge[0],
            self.origin[1] + edge[1],
            self.origin[2] + edge[2],
        )

    def edges(self) -> list[tuple[Vector, Vector]]:
        vertices = [self.vertex(0), self.vertex(1), self.vertex(2)]
        return [(vertices[i], _subtract(vertices[(i + 1) % 3], vertices[i])) for i in range(3)]


def ell_triangles_3d(
    first: ArrayLike,
    second: ArrayLike,
    point: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> Evaluation:
    """Return ``(ell, u, v)`` for two triangles of R^3, or ``(inf, None, None)``.

    Raises ``NotImplementedError`` when ``p`` lies on the plane of a triangle,
    the three-dimensional analogue of the degenerate branch of the planar
    remark.  That cannot happen when ``Q`` is convex, since an interior point
    is strictly inside every facet half-space.
    """
    vertices_1 = _vertices(first, "the first triangle")
    vertices_2 = _vertices(second, "the second triangle")
    p = _coordinates(point, "point")
    scale = _spread((*vertices_1, *vertices_2, p))

    triangle_1, triangle_2 = _Prepared(vertices_1), _Prepared(vertices_2)
    alpha_1 = _dot(triangle_1.normal, _subtract(p, triangle_1.origin))
    alpha_2 = _dot(triangle_2.normal, _subtract(p, triangle_2.origin))
    if abs(alpha_1) <= tolerance * scale or abs(alpha_2) <= tolerance * scale:
        raise NotImplementedError(
            "the query point lies on the plane of a boundary triangle; "
            "this degenerate branch is not covered by the closed form"
        )

    directions = _interior_directions(triangle_1.normal, triangle_2.normal, alpha_1, alpha_2)
    directions += _edge_directions(triangle_1, triangle_2, p, forward=False)
    directions += _edge_directions(triangle_2, triangle_1, p, forward=True)
    directions += _edge_pair_directions(triangle_1, triangle_2, p, tolerance, scale)

    best = float("inf")
    best_pair: tuple[Vector, Vector] | None = None
    minimum_length = tolerance * scale
    for direction in directions:
        norm = math.sqrt(_dot(direction, direction))
        if norm <= minimum_length:
            continue
        unit = (direction[0] / norm, direction[1] / norm, direction[2] / norm)
        projection_1 = _dot(triangle_1.normal, unit)
        projection_2 = _dot(triangle_2.normal, unit)
        if abs(projection_1) <= tolerance or abs(projection_2) <= tolerance:
            continue
        forward = alpha_1 / projection_1
        backward = -alpha_2 / projection_2
        if forward < 0.0 and backward < 0.0:
            unit = (-unit[0], -unit[1], -unit[2])
            forward, backward = -forward, -backward
        if forward <= minimum_length or backward <= minimum_length:
            continue
        length = forward + backward
        if length >= best:
            continue
        u = (p[0] - forward * unit[0], p[1] - forward * unit[1], p[2] - forward * unit[2])
        v = (p[0] + backward * unit[0], p[1] + backward * unit[1], p[2] + backward * unit[2])
        if triangle_1.contains(u, tolerance) and triangle_2.contains(v, tolerance):
            best, best_pair = length, (u, v)
    if best_pair is None:
        return _INFEASIBLE
    return best, np.array(best_pair[0]), np.array(best_pair[1])


def _interior_directions(
    normal_1: Vector, normal_2: Vector, alpha_1: float, alpha_2: float
) -> list[Vector]:
    """Stratum with both endpoints interior: ``d`` lies in ``span{n_1, n_2}``."""
    axis = _cross(normal_1, normal_2)
    if _dot(axis, axis) <= 1e-24:
        return [normal_1]  # parallel planes: the chord runs along the common normal
    projection = _dot(normal_2, normal_1)
    residual = (
        normal_2[0] - projection * normal_1[0],
        normal_2[1] - projection * normal_1[1],
        normal_2[2] - projection * normal_1[2],
    )
    norm = math.sqrt(_dot(residual, residual))
    second_axis = (residual[0] / norm, residual[1] / norm, residual[2] / norm)
    # In the basis (n_1, second_axis): n_1 . e = (1, 0) and n_2 . e = (projection, norm).
    a, b, c, d = 1.0, 0.0, projection, norm
    # alpha_1 (b - a x)(c + d x)^2 = alpha_2 (d - c x)(a + b x)^2, with x = tan(theta).
    left = _multiply([b, -a], _multiply([c, d], [c, d]))
    right = _multiply([d, -c], _multiply([a, b], [a, b]))
    coefficients = [alpha_1 * left[k] - alpha_2 * right[k] for k in range(4)]
    roots = real_roots(coefficients[3], coefficients[2], coefficients[1], coefficients[0])
    directions = [
        (
            normal_1[0] + root * second_axis[0],
            normal_1[1] + root * second_axis[1],
            normal_1[2] + root * second_axis[2],
        )
        for root in roots
    ]
    directions.append(second_axis)  # the direction at x = infinity
    return directions


def _multiply(left: list[float], right: list[float]) -> list[float]:
    """Polynomial product, coefficients ascending."""
    product = [0.0] * (len(left) + len(right) - 1)
    for i, first in enumerate(left):
        for j, second in enumerate(right):
            product[i + j] += first * second
    return product


def _edge_directions(
    parametrised: _Prepared, other: _Prepared, p: Vector, *, forward: bool
) -> list[Vector]:
    """Stratum with one endpoint on an edge: the planar stationarity cubic.

    Along an edge ``u(s) = a + s A`` the second endpoint is forced onto the
    plane of the other triangle, and the ratio ``kappa(s)`` is again
    ``(gamma + beta s)/(alpha + beta s)``; the stationarity condition is
    therefore the same cubic ``P``.
    """
    directions: list[Vector] = []
    normal = other.normal
    for a, edge in parametrised.edges():
        offset = _subtract(p, a)
        alpha = _dot(normal, offset)
        beta = -_dot(normal, edge)
        gamma = _dot(normal, _subtract(other.origin, a))
        eta_0 = _dot(offset, offset)
        eta_1 = -2.0 * _dot(offset, edge)
        eta_2 = _dot(edge, edge)
        for s in (0.0, 1.0, *stationary_candidates(alpha, beta, gamma, eta_0, eta_1, eta_2)):
            if not math.isfinite(s) or s < 0.0 or s > 1.0:
                continue
            endpoint = (a[0] + s * edge[0], a[1] + s * edge[1], a[2] + s * edge[2])
            directions.append(_subtract(endpoint, p) if forward else _subtract(p, endpoint))
    return directions


def _edge_pair_directions(
    triangle_1: _Prepared, triangle_2: _Prepared, p: Vector, tolerance: float, scale: float
) -> list[Vector]:
    """Stratum with both endpoints on edges: one chord per pair of edges.

    A line through ``p`` meeting an edge ``e`` of the first triangle sweeps the
    plane ``aff({p} u e)``.  In general position that plane meets an edge ``f``
    of the second triangle in a single point, which is the chord this stratum
    contributes.  Two configurations escape it, and neither loses a candidate.

    If the plane misses ``f``, the stratum is empty.  If it contains ``f``, the
    stratum is a curve instead of a point, but that curve is the one already
    parametrised along ``e`` by :func:`_edge_directions`: the plane meets
    ``aff(T_j)`` in a line containing ``f``, so the second endpoint forced by
    ``e`` runs along it.  Its interior stationary points are therefore roots of
    the same cubic, and its endpoints have a third constraint active, hence put
    an endpoint at a vertex, which the parameters ``s = 0`` and ``s = 1`` of
    :func:`_edge_directions` already supply.

    The sweep normal cannot vanish here: that would put ``p`` on ``aff(e)``,
    hence on ``aff(T_i)``, which the caller has excluded.
    """
    directions: list[Vector] = []
    area_threshold = tolerance * scale * scale
    second_edges = triangle_2.edges()
    for a, edge in triangle_1.edges():
        sweep = _cross(_subtract(a, p), edge)
        if math.sqrt(_dot(sweep, sweep)) <= area_threshold:
            continue
        for b, other_edge in second_edges:
            denominator = _dot(sweep, other_edge)
            if abs(denominator) <= area_threshold:
                continue
            parameter = -_dot(sweep, _subtract(b, p)) / denominator
            if parameter < 0.0 or parameter > 1.0:
                continue
            endpoint = (
                b[0] + parameter * other_edge[0],
                b[1] + parameter * other_edge[1],
                b[2] + parameter * other_edge[2],
            )
            directions.append(_subtract(endpoint, p))
    return directions
