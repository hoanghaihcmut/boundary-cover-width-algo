# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.random_convex_polyhedra_3d import (
    pointwise_algorithm_1,
    polyhedron_boundary,
    random_convex_polyhedron,
    sample_interior_points,
)
from width_function.algorithms import algorithm_1


def test_random_convex_polyhedra_3d_builds_a_valid_three_dimensional_input() -> None:
    generator = np.random.default_rng(20260840)
    vertices, facets = random_convex_polyhedron(generator, candidate_vertex_count=6)
    query_points = sample_interior_points(vertices, point_count=20, generator=generator)
    boundary = polyhedron_boundary(vertices, facets)

    hull = ConvexHull(vertices)
    signed_offsets = query_points @ hull.equations[:, :3].T + hull.equations[:, 3]

    assert vertices.shape[1] == 3
    assert facets.ndim == 2 and facets.shape[1] == 3
    assert query_points.shape == (20, 3)
    assert np.all(signed_offsets < -1e-10)
    assert len(boundary) == len(facets)
    assert all(piece.dimension == 3 for piece in boundary)


def test_singleton_widths_reconstruct_the_full_g_omega_sequence() -> None:
    generator = np.random.default_rng(20260850)
    vertices, facets = random_convex_polyhedron(generator, candidate_vertex_count=4)
    query_points = sample_interior_points(vertices, point_count=3, generator=generator)
    boundary = polyhedron_boundary(vertices, facets)

    full_g = algorithm_1(boundary, query_points)
    pointwise_widths, _ = pointwise_algorithm_1(boundary, query_points)
    # The incumbents follow the processing order chosen by Algorithm 1.
    processed = pointwise_widths[full_g.order]
    reconstructed = np.concatenate(
        [[full_g.w[0]], np.maximum(full_g.w[0], np.maximum.accumulate(processed))]
    )

    assert full_g.w == pytest.approx(reconstructed, abs=1e-8)
    assert full_g.width == pytest.approx(np.max(pointwise_widths), abs=1e-8)
    assert sorted(full_g.order.tolist()) == list(range(len(query_points)))
