# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Finite Voronoi sampling of a simple polygon's medial skeleton."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Voronoi

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SkeletonSample:
    """Voronoi ridge segments and a finite representative point set."""

    points: FloatArray
    segments: FloatArray


def sample_convex_polygon_skeleton(
    polygon: ArrayLike,
    *,
    boundary_samples_per_edge: int = 32,
    max_points: int = 15,
    min_clearance_fraction: float = 0.04,
    tolerance: float = 1e-10,
) -> SkeletonSample:
    """Approximate the medial skeleton using Voronoi ridges of boundary samples."""
    return sample_polygon_skeleton(
        polygon,
        boundary_samples_per_edge=boundary_samples_per_edge,
        max_points=max_points,
        min_clearance_fraction=min_clearance_fraction,
        tolerance=tolerance,
    )


def sample_polygon_skeleton(
    polygon: ArrayLike,
    *,
    boundary_samples_per_edge: int = 32,
    max_points: int = 15,
    min_clearance_fraction: float = 0.04,
    tolerance: float = 1e-10,
) -> SkeletonSample:
    """Approximate a simple polygon's skeleton using interior Voronoi ridges."""
    vertices = np.asarray(polygon, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("polygon must be an (m, 2) array with m >= 3")
    if boundary_samples_per_edge < 2 or max_points < 1:
        raise ValueError("sampling counts must be positive and nontrivial")
    if min_clearance_fraction < 0.0:
        raise ValueError("min_clearance_fraction must be nonnegative")

    boundary_samples = []
    edge_labels = []
    parameters = np.linspace(0.0, 1.0, boundary_samples_per_edge, endpoint=False)
    for edge_index, (start, end) in enumerate(
        zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    ):
        boundary_samples.extend((1.0 - value) * start + value * end for value in parameters)
        edge_labels.extend([edge_index] * boundary_samples_per_edge)
    boundary_samples_array = np.asarray(boundary_samples)
    edge_labels_array = np.asarray(edge_labels)
    voronoi = Voronoi(boundary_samples_array)

    diameter = float(
        np.max(np.linalg.norm(vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :], axis=2))
    )
    minimum_clearance = min_clearance_fraction * diameter
    segments = []
    for ridge_points, ridge_vertices in zip(
        voronoi.ridge_points, voronoi.ridge_vertices, strict=True
    ):
        first_sample, second_sample = ridge_points
        if edge_labels_array[first_sample] == edge_labels_array[second_sample]:
            continue
        if len(ridge_vertices) != 2 or min(ridge_vertices) < 0:
            continue
        start, end = voronoi.vertices[ridge_vertices]
        ridge_parameters = np.linspace(0.0, 1.0, 9)
        interior_checks = (1.0 - ridge_parameters[:, np.newaxis]) * start + ridge_parameters[
            :, np.newaxis
        ] * end
        if not all(_strictly_inside(vertices, point, tolerance) for point in interior_checks):
            continue
        midpoint = (start + end) / 2.0
        if _distance_to_boundary(vertices, midpoint) <= minimum_clearance:
            continue
        segments.append((start, end))
    if not segments:
        raise RuntimeError("Voronoi construction produced no interior skeleton ridge")
    segment_array = np.asarray(segments)

    candidates = np.vstack(
        [segment_array[:, 0], segment_array[:, 1], np.mean(segment_array, axis=1)]
    )
    candidates = np.unique(np.round(candidates, decimals=12), axis=0)
    candidates = np.asarray(
        [
            point
            for point in candidates
            if _distance_to_boundary(vertices, point) > minimum_clearance
        ]
    )
    if not len(candidates):
        raise RuntimeError("no skeleton point satisfies the boundary-clearance threshold")
    selected = _farthest_point_sample(vertices, candidates, min(max_points, len(candidates)))
    return SkeletonSample(points=selected, segments=segment_array)


def _farthest_point_sample(polygon: FloatArray, candidates: FloatArray, count: int) -> FloatArray:
    clearances = np.array([_distance_to_boundary(polygon, point) for point in candidates])
    selected = [int(np.argmax(clearances))]
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(
                candidates[:, np.newaxis, :] - candidates[np.asarray(selected)][np.newaxis, :, :],
                axis=2,
            ),
            axis=1,
        )
        distances[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(distances)))
    return candidates[np.asarray(selected)]


def _strictly_inside(vertices: FloatArray, point: FloatArray, tolerance: float) -> bool:
    if _distance_to_boundary(vertices, point) <= tolerance:
        return False
    x, y = point
    following = np.roll(vertices, -1, axis=0)
    crosses_y = (vertices[:, 1] > y) != (following[:, 1] > y)
    if not np.any(crosses_y):
        return False
    starts = vertices[crosses_y]
    ends = following[crosses_y]
    crossing_x = starts[:, 0] + (y - starts[:, 1]) * (ends[:, 0] - starts[:, 0]) / (
        ends[:, 1] - starts[:, 1]
    )
    return bool(np.count_nonzero(x < crossing_x) % 2)


def _distance_to_boundary(vertices: FloatArray, point: FloatArray) -> float:
    following = np.roll(vertices, -1, axis=0)
    edges = following - vertices
    squared_lengths = np.einsum("ij,ij->i", edges, edges)
    offsets = point - vertices
    parameters = np.divide(
        np.einsum("ij,ij->i", offsets, edges),
        squared_lengths,
        out=np.zeros_like(squared_lengths),
        where=squared_lengths > 0.0,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    projections = vertices + parameters[:, np.newaxis] * edges
    return float(np.min(np.linalg.norm(point - projections, axis=1)))
