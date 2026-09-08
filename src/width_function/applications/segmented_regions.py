# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Polygonization, width measurement, and overlays for segmented image regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from width_function.algorithms import algorithm_1
from width_function.geometry import (
    ConvexPolytope,
    contains_in_join,
    ell_between_polytopes_with_witness,
    sample_polygon_skeleton,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PolygonMeasurement:
    """Geometric objects used to report one segmented-region width."""

    polygon: FloatArray
    query_points: FloatArray
    skeleton_segments: FloatArray
    point_widths: FloatArray
    width: float
    maximizing_point: FloatArray
    segment_start: FloatArray
    segment_end: FloatArray


def polygon_area(polygon: ArrayLike) -> float:
    """Return the unsigned shoelace area of a simple planar polygon."""
    vertices = _clean_polygon(polygon)
    following = np.roll(vertices, -1, axis=0)
    return 0.5 * abs(
        float(np.sum(vertices[:, 0] * following[:, 1] - vertices[:, 1] * following[:, 0]))
    )


def simplify_polygon(
    polygon: ArrayLike,
    *,
    tolerance: float = 2.0,
    max_vertices: int = 48,
) -> FloatArray:
    """Simplify a closed contour while retaining at least three vertices."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if max_vertices < 3:
        raise ValueError("max_vertices must be at least three")
    cv2 = _cv2()
    vertices = _clean_polygon(polygon)
    contour = vertices.astype(np.float32).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = tolerance
    best = vertices
    for _ in range(48):
        approximated = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approximated) >= 3:
            best = _clean_polygon(approximated)
            if len(best) <= max_vertices:
                return best
        epsilon = max(epsilon * 1.5, perimeter * 1e-6, 1e-9)
    indices = np.linspace(0, len(vertices), max_vertices, endpoint=False, dtype=int)
    return _clean_polygon(vertices[indices])


def prediction_polygons(prediction: Any) -> tuple[FloatArray, ...]:
    """Extract valid polygonal instances from one Ultralytics prediction."""
    if prediction.masks is None or not prediction.masks.xy:
        return ()
    polygons = []
    for coordinates in prediction.masks.xy:
        try:
            polygon = _clean_polygon(coordinates)
        except ValueError:
            continue
        if polygon_area(polygon) > 0.0:
            polygons.append(polygon)
    return tuple(polygons)


def merge_detection_polygons(
    polygons: tuple[ArrayLike, ...] | list[ArrayLike],
    *,
    image_shape: tuple[int, int],
    simplify_tolerance: float = 1.5,
    max_vertices: int = 48,
    minimum_area: float = 20.0,
    close_kernel: int = 3,
) -> tuple[FloatArray, ...]:
    """Raster-union detected instances and return their exterior components."""
    cv2 = _cv2()
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        vertices = np.rint(_clean_polygon(polygon)).astype(np.int32)
        cv2.fillPoly(mask, [vertices], 255)
    if close_kernel > 1 and np.any(mask):
        size = close_kernel if close_kernel % 2 else close_kernel + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    components = []
    for contour in contours:
        raw = contour[:, 0, :].astype(float)
        if len(raw) < 3 or polygon_area(raw) < minimum_area:
            continue
        simplified = simplify_polygon(
            raw,
            tolerance=simplify_tolerance,
            max_vertices=max_vertices,
        )
        if polygon_area(simplified) >= minimum_area:
            components.append(simplified)
    return tuple(sorted(components, key=polygon_area, reverse=True))


def points_strictly_inside(points: ArrayLike, polygon: ArrayLike) -> NDArray[np.bool_]:
    """Return whether each query point is strictly inside a polygon."""
    cv2 = _cv2()
    queries = np.asarray(points, dtype=float)
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError("points must be an (n, 2) array")
    contour = _clean_polygon(polygon).astype(np.float32)
    return np.asarray(
        [cv2.pointPolygonTest(contour, tuple(map(float, point)), False) > 0 for point in queries],
        dtype=bool,
    )


def sample_component_skeleton(
    polygons: tuple[ArrayLike, ...] | list[ArrayLike],
    *,
    max_points: int = 30,
    boundary_samples_per_edge: int = 8,
    min_clearance_fraction: float = 0.002,
) -> FloatArray:
    """Sample a finite initial set over all polygonal components."""
    components = [_clean_polygon(polygon) for polygon in polygons]
    if not components:
        raise ValueError("at least one polygonal component is required")
    if max_points < len(components):
        raise ValueError("max_points must be at least the number of components")
    areas = np.asarray([polygon_area(polygon) for polygon in components])
    allocations = np.ones(len(components), dtype=int)
    for _ in range(max_points - len(components)):
        allocations[int(np.argmax(areas / allocations))] += 1
    samples = []
    for polygon, count in zip(components, allocations, strict=True):
        skeleton = _sample_skeleton(
            polygon,
            boundary_samples_per_edge=boundary_samples_per_edge,
            max_points=int(count),
            min_clearance_fraction=min_clearance_fraction,
        )
        samples.append(skeleton.points)
    return np.vstack(samples)


def measure_polygon(
    polygon: ArrayLike,
    *,
    query_points: ArrayLike | None = None,
    simplify_tolerance: float = 2.0,
    max_vertices: int = 48,
    boundary_samples_per_edge: int = 8,
    max_skeleton_points: int = 30,
    min_clearance_fraction: float = 0.005,
) -> PolygonMeasurement:
    """Compute Algorithm 1 and a realizing segment for one polygon."""
    simplified = simplify_polygon(
        polygon,
        tolerance=simplify_tolerance,
        max_vertices=max_vertices,
    )
    if query_points is None:
        skeleton = _sample_skeleton(
            simplified,
            boundary_samples_per_edge=boundary_samples_per_edge,
            max_points=max_skeleton_points,
            min_clearance_fraction=min_clearance_fraction,
        )
        points = skeleton.points
        segments = skeleton.segments
    else:
        points = np.asarray(query_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            raise ValueError("query_points must be a nonempty (n, 2) array")
        if not np.all(points_strictly_inside(points, simplified)):
            raise ValueError("every query point must lie strictly inside the polygon")
        segments = np.empty((0, 2, 2), dtype=float)
    boundary = tuple(
        ConvexPolytope([start, end])
        for start, end in zip(simplified, np.roll(simplified, -1, axis=0), strict=True)
    )
    point_widths = np.asarray(
        [algorithm_1(boundary, point[np.newaxis, :]).width for point in points]
    )
    maximizing_index = int(np.argmax(point_widths))
    maximizing_point = points[maximizing_index]
    segment_start, segment_end = _shortest_segment(boundary, maximizing_point)
    return PolygonMeasurement(
        polygon=simplified,
        query_points=points,
        skeleton_segments=segments,
        point_widths=point_widths,
        width=float(point_widths[maximizing_index]),
        maximizing_point=maximizing_point,
        segment_start=segment_start,
        segment_end=segment_end,
    )


def draw_measurement_overlay(
    image: ArrayLike,
    polygons: tuple[ArrayLike, ...] | list[ArrayLike],
    output_path: Path,
    *,
    query_points: ArrayLike | None = None,
    skeleton_segments: ArrayLike | None = None,
    segment_start: ArrayLike | None = None,
    segment_end: ArrayLike | None = None,
) -> None:
    """Draw polygon boundaries, skeleton geometry, and a maximizing segment."""
    cv2 = _cv2()
    canvas = np.asarray(image).copy()
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("image must be a BGR array with shape (height, width, 3)")
    tint = canvas.copy()
    contours = [_integer_contour(polygon) for polygon in polygons]
    if contours:
        cv2.fillPoly(tint, contours, (0, 140, 255))
        canvas = cv2.addWeighted(tint, 0.22, canvas, 0.78, 0.0)
        cv2.polylines(canvas, contours, True, (0, 140, 255), 2, cv2.LINE_AA)
    if skeleton_segments is not None:
        segments = np.asarray(skeleton_segments, dtype=float).reshape(-1, 2, 2)
        for start, end in segments:
            cv2.line(canvas, _pixel(start), _pixel(end), (255, 255, 255), 1, cv2.LINE_AA)
    if query_points is not None:
        for point in np.asarray(query_points, dtype=float).reshape(-1, 2):
            cv2.circle(canvas, _pixel(point), 2, (0, 220, 0), -1, cv2.LINE_AA)
    if segment_start is not None and segment_end is not None:
        start, end = _pixel(segment_start), _pixel(segment_end)
        cv2.line(canvas, start, end, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(canvas, start, 4, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, end, 4, (255, 255, 0), -1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not write overlay: {output_path}")


def _sample_skeleton(
    polygon: FloatArray,
    *,
    boundary_samples_per_edge: int,
    max_points: int,
    min_clearance_fraction: float,
):
    error: Exception | None = None
    for clearance in dict.fromkeys((min_clearance_fraction, min_clearance_fraction / 2.0, 0.0)):
        try:
            return sample_polygon_skeleton(
                polygon,
                boundary_samples_per_edge=boundary_samples_per_edge,
                max_points=max_points,
                min_clearance_fraction=clearance,
            )
        except (RuntimeError, ValueError) as exc:
            error = exc
    raise RuntimeError(f"skeleton sampling failed: {error}")


def _shortest_segment(
    boundary: tuple[ConvexPolytope, ...],
    point: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    best = None
    for first_index, first in enumerate(boundary):
        for second in boundary[first_index + 1 :]:
            if not contains_in_join(first, second, point):
                continue
            try:
                witness = ell_between_polytopes_with_witness(first, second, point)
            except ValueError:
                continue
            if best is None or witness.length < best.length:
                best = witness
    if best is None:
        raise RuntimeError("no boundary-piece pair realizes the point width")
    return best.start, best.end


def _clean_polygon(polygon: ArrayLike) -> FloatArray:
    vertices = np.asarray(polygon, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("polygon must be an (m, 2) array with m >= 3")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("polygon vertices must be finite")
    if np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    keep = np.r_[True, np.any(np.diff(vertices, axis=0) != 0.0, axis=1)]
    vertices = vertices[keep]
    if len(vertices) < 3:
        raise ValueError("polygon must contain at least three distinct vertices")
    return vertices


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for image applications") from exc
    return cv2


def _integer_contour(polygon: ArrayLike) -> NDArray[np.int32]:
    return np.rint(_clean_polygon(polygon)).astype(np.int32).reshape(-1, 1, 2)


def _pixel(point: ArrayLike) -> tuple[int, int]:
    coordinates = np.rint(np.asarray(point, dtype=float)).astype(int)
    return int(coordinates[0]), int(coordinates[1])
