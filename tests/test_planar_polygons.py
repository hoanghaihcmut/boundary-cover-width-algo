# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""The closed-form planar primitives must agree with the general formulation."""

import numpy as np
import pytest

from width_function.geometry import (
    ConvexPolytope,
    ell_between_polytopes_by_program,
    ell_segments_2d,
    point_in_join_2d,
)
from width_function.geometry.convex_polytope import convex_hull_coordinates


def reference_ell(a_1, b_1, a_2, b_2, point, samples=100_001):
    """Brute-force ell by scanning the first edge and intersecting exactly.

    For each ``x = u(s)`` the second endpoint is forced: it is where the line
    through ``x`` and ``p`` meets ``aff(e_j)``.  Scanning ``s`` therefore
    reproduces the minimisation of the Remark without reusing its candidate
    set, which is what the closed form has to be checked against.
    """
    s = np.linspace(0.0, 1.0, samples)[:, np.newaxis]
    x = a_1 + s * (b_1 - a_1)
    ray = point[np.newaxis, :] - x
    edge = b_2 - a_2
    determinant = ray[:, 0] * (-edge[1]) - ray[:, 1] * (-edge[0])
    gap = a_2[np.newaxis, :] - x
    with np.errstate(divide="ignore", invalid="ignore"):
        scaling = (gap[:, 0] * (-edge[1]) - gap[:, 1] * (-edge[0])) / determinant
        crossing = (ray[:, 0] * gap[:, 1] - ray[:, 1] * gap[:, 0]) / determinant
    lengths = scaling * np.linalg.norm(ray, axis=1)
    admissible = np.isfinite(lengths) & (scaling >= 1.0) & (crossing >= 0.0) & (crossing <= 1.0)
    return float(np.min(np.where(admissible, lengths, np.inf)))


def assert_admissible(length, start, end, point, tolerance=1e-9):
    chord = end - start
    offset = point - start
    assert abs(chord[0] * offset[1] - chord[1] * offset[0]) <= tolerance * np.linalg.norm(chord)
    assert -tolerance <= float(offset @ chord) / float(chord @ chord) <= 1.0 + tolerance
    assert np.linalg.norm(chord) == pytest.approx(length, abs=1e-9)


def test_closed_form_matches_the_dimension_independent_program() -> None:
    generator = np.random.default_rng(20260824)
    compared = 0
    for _ in range(400):
        a_1, b_1, a_2, b_2 = generator.uniform(-2.0, 2.0, size=(4, 2))
        point = generator.uniform(-2.0, 2.0, size=2)
        if not point_in_join_2d(a_1, b_1, a_2, b_2, point):
            continue
        length, start, end = ell_segments_2d(a_1, b_1, a_2, b_2, point)
        if not np.isfinite(length):
            continue
        program = ell_between_polytopes_by_program(
            ConvexPolytope([a_1, b_1]), ConvexPolytope([a_2, b_2]), point
        )
        assert length == pytest.approx(program.length, rel=1e-7, abs=1e-9)
        assert_admissible(length, start, end, point)
        compared += 1
    assert compared > 50


def test_cubic_candidate_set_matches_brute_force_sampling() -> None:
    """An independent check that the candidate set of the Remark misses no minimiser."""
    generator = np.random.default_rng(31337)
    compared = 0
    for _ in range(60):
        a_1, b_1, a_2, b_2 = generator.uniform(-2.0, 2.0, size=(4, 2))
        point = generator.uniform(-1.0, 1.0, size=2)
        length, _, _ = ell_segments_2d(a_1, b_1, a_2, b_2, point)
        sampled = reference_ell(a_1, b_1, a_2, b_2, point)
        if not np.isfinite(length) or not np.isfinite(sampled):
            continue
        assert length <= sampled + 1e-9
        assert length == pytest.approx(sampled, rel=1e-3)
        compared += 1
    assert compared > 5


def test_membership_predicate_matches_the_linear_program() -> None:
    generator = np.random.default_rng(4242)
    for _ in range(500):
        a_1, b_1, a_2, b_2 = generator.uniform(-2.0, 2.0, size=(4, 2))
        point = generator.uniform(-2.0, 2.0, size=2)
        pieces = (ConvexPolytope([a_1, b_1]), ConvexPolytope([a_2, b_2]))
        expected = convex_hull_coordinates(point, pieces) is not None
        assert point_in_join_2d(a_1, b_1, a_2, b_2, point) is expected


def test_degenerate_branch_when_the_point_lies_on_the_line_of_an_edge() -> None:
    """p in aff(e_j) but not in aff(e_i): every admissible chord lies in aff(e_j).

    The first edge then contributes its single crossing q of that line, which
    must sit on the ray opposite to e_j, so ell = ||q - p|| + dist(p, e_j).
    """
    generator = np.random.default_rng(99)
    for _ in range(200):
        angle = generator.uniform(0.0, 2.0 * np.pi)
        axis = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-axis[1], axis[0]])
        point = generator.uniform(-2.0, 2.0, size=2)
        near, far = np.sort(generator.uniform(0.3, 3.0, size=2))
        crossing = generator.uniform(0.3, 3.0)
        a_2, b_2 = point + near * axis, point + far * axis
        base = point - crossing * axis
        a_1 = base + generator.uniform(0.2, 2.0) * normal
        b_1 = base - generator.uniform(0.2, 2.0) * normal

        assert point_in_join_2d(a_1, b_1, a_2, b_2, point)
        length, start, end = ell_segments_2d(a_1, b_1, a_2, b_2, point)
        assert length == pytest.approx(crossing + near, rel=1e-9)
        assert_admissible(length, start, end, point)
        assert start == pytest.approx(base, abs=1e-9)
        assert end == pytest.approx(a_2, abs=1e-9)


def test_collinear_edges_reduce_to_the_one_dimensional_formula() -> None:
    # Both edges on the x axis, p between them: ell adds the two clearances.
    length, start, end = ell_segments_2d(
        [-3.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [5.0, 0.0], [0.0, 0.0]
    )
    assert length == pytest.approx(3.0)
    assert start == pytest.approx(np.array([-1.0, 0.0]))
    assert end == pytest.approx(np.array([2.0, 0.0]))

    # The same edges, but p outside their span: no admissible pair.
    assert not np.isfinite(
        ell_segments_2d([-3.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [5.0, 0.0], [6.0, 0.0])[0]
    )


def test_t_shaped_polygon_reaches_the_reflex_vertex_chord() -> None:
    # The chord joining the two reflex vertices of a T is a collinear-edge pair.
    length, start, end = ell_segments_2d([3.5, 4.0], [6.0, 4.0], [0.0, 4.0], [2.5, 4.0], [3.0, 4.0])
    assert length == pytest.approx(1.0)
    assert {tuple(np.round(start, 12)), tuple(np.round(end, 12))} == {(3.5, 4.0), (2.5, 4.0)}
