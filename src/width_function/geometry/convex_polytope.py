# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Numerical primitives for compact convex polytopes in arbitrary dimensions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import LinearConstraint, linprog, minimize

from width_function.geometry.planar_polygons import ell_segments_2d, point_in_join_2d
from width_function.geometry.triangulated_polyhedra import (
    ell_triangles_3d,
    point_in_join_3d,
    point_triangle_distance,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EllWitness:
    """A shortest feasible segment through the prescribed point."""

    length: float
    start: FloatArray
    end: FloatArray


@dataclass(frozen=True)
class ConvexPolytope:
    """A compact convex set represented as the convex hull of its vertices."""

    vertices: FloatArray

    def __init__(self, vertices: ArrayLike):
        points = np.asarray(vertices, dtype=float)
        if points.ndim != 2 or not len(points):
            raise ValueError("vertices must be a nonempty (n_vertices, dimension) array")
        if points.shape[1] < 1 or not np.all(np.isfinite(points)):
            raise ValueError("vertices must contain finite points in R^d, d >= 1")
        object.__setattr__(self, "vertices", points)

    @property
    def dimension(self) -> int:
        return self.vertices.shape[1]

    def distance(self, point: ArrayLike, *, tolerance: float = 1e-10) -> float:
        """Euclidean distance from a point to this polytope."""
        p = _point(point, self.dimension)
        count = len(self.vertices)
        if count == 3 and self.dimension == 3:
            return point_triangle_distance(self.vertices, p)
        if count == 1:
            return float(np.linalg.norm(p - self.vertices[0]))
        if count == 2:
            start, end = self.vertices
            edge = end - start
            parameter = np.clip(np.dot(p - start, edge) / np.dot(edge, edge), 0.0, 1.0)
            return float(np.linalg.norm(p - (start + parameter * edge)))
        constraint = LinearConstraint(np.ones((1, count)), 1.0, 1.0)

        def objective(weights: FloatArray) -> float:
            residual = weights @ self.vertices - p
            return 0.5 * float(residual @ residual)

        def gradient(weights: FloatArray) -> FloatArray:
            return self.vertices @ (weights @ self.vertices - p)

        seeds = [np.full(count, 1.0 / count), *np.eye(count)]
        candidates = []
        for seed in seeds:
            result = minimize(
                objective,
                seed,
                jac=gradient,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * count,
                constraints=[constraint],
                options={"ftol": max(tolerance**2, 1e-12), "maxiter": 1000},
            )
            if abs(np.sum(result.x) - 1.0) <= tolerance:
                candidates.append(result.x)
        if not candidates:
            raise RuntimeError("point-to-polytope optimization found no feasible solution")
        weights = min(candidates, key=objective)
        return float(np.linalg.norm(weights @ self.vertices - p))


def convex_hull_coordinates(
    point: ArrayLike, polytopes: tuple[ConvexPolytope, ...], *, tolerance: float = 1e-9
) -> FloatArray | None:
    """Return convex weights for a point in the hull of the supplied polytopes."""
    if not polytopes:
        raise ValueError("at least one polytope is required")
    dimension = polytopes[0].dimension
    if any(piece.dimension != dimension for piece in polytopes):
        raise ValueError("all polytopes must have the same ambient dimension")
    p = _point(point, dimension)
    vertices = np.vstack([piece.vertices for piece in polytopes])
    equality = np.vstack([vertices.T, np.ones(len(vertices))])
    target = np.append(p, 1.0)
    result = linprog(
        np.zeros(len(vertices)),
        A_eq=equality,
        b_eq=target,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or np.linalg.norm(equality @ result.x - target, ord=np.inf) > tolerance:
        return None
    return np.asarray(result.x, dtype=float)


def _planar_segments(first: ConvexPolytope, second: ConvexPolytope) -> bool:
    """Whether both pieces are segments of a planar polygon boundary cover."""
    return (
        first.dimension == 2
        and second.dimension == 2
        and len(first.vertices) == 2
        and len(second.vertices) == 2
    )


def _spatial_triangles(first: ConvexPolytope, second: ConvexPolytope) -> bool:
    """Whether both pieces are triangles of a triangulated boundary in R^3."""
    return (
        first.dimension == 3
        and second.dimension == 3
        and len(first.vertices) == 3
        and len(second.vertices) == 3
    )


def contains_in_join(
    first: ConvexPolytope,
    second: ConvexPolytope,
    point: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Test whether point belongs to conv(first union second)."""
    if _planar_segments(first, second):
        return point_in_join_2d(*first.vertices, *second.vertices, point, tolerance=tolerance)
    if _spatial_triangles(first, second):
        return point_in_join_3d(first.vertices, second.vertices, point, tolerance=tolerance)
    return convex_hull_coordinates(point, (first, second), tolerance=tolerance) is not None


def ell_between_polytopes(
    first: ConvexPolytope,
    second: ConvexPolytope,
    point: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> float:
    """Shortest segment through point with endpoints in two convex polytopes.

    Combined barycentric weights ``r, s`` satisfy only linear constraints:
    ``A.T @ r + B.T @ s = point`` and ``sum(r) + sum(s) = 1``.  Normalising
    each group recovers the two endpoints.  Multiple deterministic LP seeds
    make the nonlinear minimisation reliable for low-dimensional boundary
    polytopes used by Algorithm 1.
    """
    return ell_between_polytopes_with_witness(first, second, point, tolerance=tolerance).length


def ell_between_polytopes_with_witness(
    first: ConvexPolytope,
    second: ConvexPolytope,
    point: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> EllWitness:
    """Return the shortest length and its two boundary endpoints.

    Planar segment pairs and spatial triangle pairs use closed forms based on
    a stratification of the boundary whose every stratum is a
    linear or a cubic equation.  Everything else, and the one degenerate
    configuration the spatial closed form leaves out, falls back on the
    dimension-independent nonlinear program below.
    """
    if first.dimension != second.dimension:
        raise ValueError("both polytopes must have the same ambient dimension")
    p = _point(point, first.dimension)
    length = start = end = None
    if _planar_segments(first, second):
        length, start, end = ell_segments_2d(
            *first.vertices, *second.vertices, p, tolerance=tolerance
        )
    elif _spatial_triangles(first, second):
        try:
            length, start, end = ell_triangles_3d(
                first.vertices, second.vertices, p, tolerance=tolerance
            )
        except NotImplementedError:
            return ell_between_polytopes_by_program(first, second, p, tolerance=tolerance)
    else:
        return ell_between_polytopes_by_program(first, second, p, tolerance=tolerance)
    if start is None or end is None:
        raise ValueError("point is not in conv(first union second), or lies on one piece")
    return EllWitness(length=float(length), start=start, end=end)


def ell_between_polytopes_by_program(
    first: ConvexPolytope,
    second: ConvexPolytope,
    point: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> EllWitness:
    """Dimension-independent evaluation of ell by a seeded nonlinear program.

    Combined barycentric weights ``r, s`` satisfy only linear constraints:
    ``A.T @ r + B.T @ s = point`` and ``sum(r) + sum(s) = 1``.  Normalising
    each group recovers the two endpoints.  Multiple deterministic LP seeds
    make the nonlinear minimisation reliable for the low-dimensional boundary
    polytopes used by Algorithm 1.  In the plane the closed form of
    :func:`width_function.geometry.planar_polygons.ell_segments_2d` replaces this, and
    is checked against it in the test suite.
    """
    p = _point(point, first.dimension)
    vertices = np.vstack([first.vertices, second.vertices])
    split = len(first.vertices)
    equality = np.vstack([vertices.T, np.ones(len(vertices))])
    target = np.append(p, 1.0)
    bounds = [(0.0, 1.0)] * len(vertices)
    eps = max(tolerance, 1e-12)

    seeds: list[FloatArray] = []
    objectives = [np.zeros(len(vertices))]
    group_bias = np.r_[np.ones(split), -np.ones(len(vertices) - split)]
    objectives.extend([group_bias, -group_bias])
    for index in range(min(len(vertices), 8)):
        objective = np.zeros(len(vertices))
        objective[index] = 1.0
        objectives.extend([objective, -objective])
    for objective in objectives:
        result = linprog(objective, A_eq=equality, b_eq=target, bounds=bounds, method="highs")
        if result.success:
            r_sum = float(np.sum(result.x[:split]))
            if eps < r_sum < 1.0 - eps:
                seeds.append(np.asarray(result.x, dtype=float))
    if not seeds:
        raise ValueError("point is not in conv(first union second), or lies on one piece")

    def objective(weights: FloatArray) -> float:
        r, s = weights[:split], weights[split:]
        r_sum, s_sum = float(np.sum(r)), float(np.sum(s))
        if r_sum <= eps or s_sum <= eps:
            return 1e30
        u = r @ first.vertices / r_sum
        v = s @ second.vertices / s_sum
        delta = u - v
        return float(delta @ delta)

    linear = LinearConstraint(equality, target, target)
    best = np.inf
    best_weights: FloatArray | None = None
    for seed in seeds:
        result = minimize(
            objective,
            seed,
            method="SLSQP",
            bounds=bounds,
            constraints=[linear],
            options={"ftol": max(tolerance**2, 1e-12), "maxiter": 2000},
        )
        violation = np.linalg.norm(equality @ result.x - target, ord=np.inf)
        if result.success and violation <= 10 * tolerance:
            value = objective(result.x)
            if value < best:
                best = value
                best_weights = np.asarray(result.x, dtype=float)
    if best_weights is None or not np.isfinite(best):
        raise RuntimeError("ell(E_i, E_j; p) optimization failed for every feasible seed")
    r, s = best_weights[:split], best_weights[split:]
    start = r @ first.vertices / np.sum(r)
    end = s @ second.vertices / np.sum(s)
    return EllWitness(
        length=float(np.sqrt(max(0.0, best))),
        start=np.asarray(start, dtype=float),
        end=np.asarray(end, dtype=float),
    )


def _point(point: ArrayLike, dimension: int) -> FloatArray:
    value = np.asarray(point, dtype=float)
    if value.shape != (dimension,) or not np.all(np.isfinite(value)):
        raise ValueError(f"point must be a finite vector in R^{dimension}")
    return value
