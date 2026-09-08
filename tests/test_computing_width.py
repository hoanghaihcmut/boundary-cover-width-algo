# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

import numpy as np
import pytest

from width_function.algorithms import (
    adaptive_epsilon_width_2d,
    algorithm_1,
    algorithm_2,
    direction_set,
    heuristic_adaptive_width_2d,
)
from width_function.geometry import (
    ConvexPolytope,
    ell_between_polytopes_with_witness,
    sample_convex_polygon_skeleton,
    sample_polygon_skeleton,
)


def hyperrectangle_boundary(dimension: int) -> list[ConvexPolytope]:
    pieces = []
    for axis in range(dimension):
        other_axes = [index for index in range(dimension) if index != axis]
        for side in (0.0, 1.0):
            vertices = []
            for mask in range(1 << (dimension - 1)):
                point = np.zeros(dimension)
                point[axis] = side
                for bit, other_axis in enumerate(other_axes):
                    point[other_axis] = float((mask >> bit) & 1)
                vertices.append(point)
            pieces.append(ConvexPolytope(vertices))
    return pieces


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_algorithm_is_dimension_independent_on_unit_hypercube(dimension: int) -> None:
    result = algorithm_1(hyperrectangle_boundary(dimension), [np.full(dimension, 0.5)])
    assert result.width == pytest.approx(1.0, abs=1e-7)


def test_square_experiment_matches_reference_formula() -> None:
    n = 8
    a = np.arange(1, n + 1, dtype=float) / (n + 1)
    result = algorithm_1(hyperrectangle_boundary(2), np.column_stack([a, a]))
    assert result.width == pytest.approx(1.0, abs=1e-7)
    assert len(result.w) == n + 1
    assert result.w[0] == pytest.approx(8.0 / 9.0, abs=1e-7)
    assert result.w[-1] == result.width
    assert np.all(np.diff(result.w) >= 0.0)
    # Lazy enumeration only pays for the pairs it reaches, and global pruning
    # stops most points after a single admissible chord.
    assert result.membership_tests < n * 6
    assert result.global_prunes + result.pairwise_prunes == n


def test_rejects_mixed_dimensions() -> None:
    with pytest.raises(ValueError, match="same R\\^d"):
        algorithm_1([ConvexPolytope([[0, 0]]), ConvexPolytope([[0, 0, 0]])], [[0, 0]])


@pytest.mark.parametrize("epsilon", [1e-1, 1e-3, 1e-5])
def test_adaptive_epsilon_approximation_is_certified(epsilon: float) -> None:
    n = 8
    polygon = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)
    a = np.arange(1, n + 1, dtype=float) / (n + 1)
    points = np.column_stack([a, a])
    result = adaptive_epsilon_width_2d(polygon, points, epsilon, theta_0=0.321)

    assert result.lower_bound <= 1.0 <= result.upper_bound
    assert 0.0 <= result.width - 1.0 <= epsilon
    assert result.certified_gap <= epsilon
    assert result.direction_evaluations == sum(1 << (int(level) - 1) for level in result.levels)


def test_algorithm_2_is_the_numbered_adaptive_entry_point() -> None:
    polygon = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    points = np.array([[0.0, 0.0], [0.2, 0.1]])

    numbered = algorithm_2(polygon, points, 1e-2, theta_0=0.321)
    descriptive = adaptive_epsilon_width_2d(polygon, points, 1e-2, theta_0=0.321)

    assert numbered.width == pytest.approx(descriptive.width)
    assert numbered.lower_bound == pytest.approx(descriptive.lower_bound)
    assert numbered.upper_bound == pytest.approx(descriptive.upper_bound)


def test_adaptive_direction_sets_are_nested() -> None:
    assert [len(direction_set(level)) for level in range(1, 6)] == [1, 2, 4, 8, 16]


def test_convex_polygon_skeleton_returns_finite_interior_set() -> None:
    polygon = np.array([[-1.0, -0.5], [0.8, -0.8], [1.0, 0.7], [0.0, 1.0], [-0.9, 0.6]])
    skeleton = sample_convex_polygon_skeleton(polygon, max_points=10)
    assert skeleton.points.shape == (10, 2)
    assert skeleton.segments.ndim == 3
    assert np.all(np.isfinite(skeleton.points))


def test_ell_witness_returns_endpoints_of_segment_through_point() -> None:
    left = ConvexPolytope([[0.0, 0.0], [0.0, 1.0]])
    right = ConvexPolytope([[1.0, 0.0], [1.0, 1.0]])
    point = np.array([0.5, 0.4])
    witness = ell_between_polytopes_with_witness(left, right, point)

    assert witness.length == pytest.approx(1.0, abs=1e-8)
    assert witness.start[0] == pytest.approx(0.0, abs=1e-8)
    assert witness.end[0] == pytest.approx(1.0, abs=1e-8)
    direction = witness.end - witness.start
    offset = point - witness.start
    cross = direction[0] * offset[1] - direction[1] * offset[0]
    assert cross == pytest.approx(0.0, abs=1e-8)


def test_concave_polygon_support_is_explicitly_heuristic() -> None:
    polygon = np.array(
        [
            [-1.0, -1.0],
            [0.0, -0.35],
            [1.0, -1.0],
            [0.35, 0.0],
            [1.0, 1.0],
            [0.0, 0.35],
            [-1.0, 1.0],
            [-0.35, 0.0],
        ]
    )
    with pytest.raises(ValueError, match="requires a convex polygon"):
        adaptive_epsilon_width_2d(polygon, [[0.0, 0.0]], 1e-2)

    result = heuristic_adaptive_width_2d(polygon, [[0.0, 0.0]], 1e-2, theta_0=0.2)
    assert np.isfinite(result.width)
    assert result.certified_gap <= 1e-2
    assert result.converged
    assert result.termination_reason == "epsilon"
    assert result.point_best_directions.shape == (1, 2)

    stopped = heuristic_adaptive_width_2d(
        polygon,
        [[0.0, 0.0]],
        1e-12,
        theta_0=0.2,
        max_runtime_seconds=1e-12,
    )
    assert not stopped.converged
    assert stopped.termination_reason == "time_limit"
    assert stopped.certified_gap > stopped.epsilon

    skeleton = sample_polygon_skeleton(
        polygon, boundary_samples_per_edge=16, max_points=10, min_clearance_fraction=0.01
    )
    assert skeleton.points.shape == (10, 2)
