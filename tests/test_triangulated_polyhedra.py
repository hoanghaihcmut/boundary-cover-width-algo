# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""The closed-form triangle primitives must agree with the general formulation."""

import numpy as np
import pytest

from width_function.geometry import (
    ConvexPolytope,
    ell_between_polytopes_by_program,
    ell_triangles_3d,
    point_in_join_3d,
    point_triangle_distance,
)
from width_function.geometry.convex_polytope import convex_hull_coordinates


def reference_ell(first, second, point, samples=400):
    """Brute-force ell by scanning the first triangle; the second end is forced.

    For each ``u`` the chord is determined: it must leave along the line
    through ``u`` and ``p``, which meets the plane of the second triangle in a
    single point.  This reproduces the minimisation without reusing the
    candidate strata of the closed form.
    """
    normal = np.cross(second[1] - second[0], second[2] - second[0])
    normal = normal / np.linalg.norm(normal)
    grid = np.linspace(0.0, 1.0, samples)
    weight_1, weight_2 = np.meshgrid(grid, grid, indexing="ij")
    inside = (weight_1 + weight_2) <= 1.0
    weight_1, weight_2 = weight_1[inside], weight_2[inside]
    u = (
        first[0]
        + weight_1[:, np.newaxis] * (first[1] - first[0])
        + weight_2[:, np.newaxis] * (first[2] - first[0])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scaling = ((second[0] - u) @ normal) / ((point - u) @ normal)
    v = u + scaling[:, np.newaxis] * (point - u)

    edge_1, edge_2 = second[1] - second[0], second[2] - second[0]
    gram = np.array([[edge_1 @ edge_1, edge_1 @ edge_2], [edge_1 @ edge_2, edge_2 @ edge_2]])
    offset = v - second[0]
    weights = np.stack([offset @ edge_1, offset @ edge_2], axis=1) @ np.linalg.inv(gram).T
    admissible = (
        np.isfinite(scaling)
        & (scaling >= 1.0)
        & (weights[:, 0] >= -1e-12)
        & (weights[:, 1] >= -1e-12)
        & (weights.sum(axis=1) <= 1.0 + 1e-12)
    )
    lengths = scaling * np.linalg.norm(point - u, axis=1)
    return float(np.min(np.where(admissible, lengths, np.inf)))


def assert_admissible(length, u, v, point, tolerance=1e-8):
    chord = v - u
    assert np.linalg.norm(np.cross(chord, point - u)) <= tolerance * np.linalg.norm(chord)
    assert -1e-9 <= float((point - u) @ chord) / float(chord @ chord) <= 1.0 + 1e-9
    assert np.linalg.norm(chord) == pytest.approx(length, abs=1e-9)


def test_closed_form_matches_the_dimension_independent_program() -> None:
    generator = np.random.default_rng(20260824)
    compared = 0
    for _ in range(400):
        first = generator.uniform(-1.0, 1.0, size=(3, 3))
        second = generator.uniform(-1.0, 1.0, size=(3, 3))
        point = generator.uniform(-0.4, 0.4, size=3)
        if not point_in_join_3d(first, second, point):
            continue
        length, u, v = ell_triangles_3d(first, second, point)
        if not np.isfinite(length):
            continue
        program = ell_between_polytopes_by_program(
            ConvexPolytope(first), ConvexPolytope(second), point
        )
        assert length == pytest.approx(program.length, rel=1e-7, abs=1e-9)
        assert_admissible(length, u, v, point)
        compared += 1
    assert compared > 50


def test_strata_enumeration_matches_brute_force_sampling() -> None:
    generator = np.random.default_rng(777)
    compared = 0
    for _ in range(120):
        first = generator.uniform(-1.0, 1.0, size=(3, 3))
        second = generator.uniform(-1.0, 1.0, size=(3, 3))
        point = generator.uniform(-0.4, 0.4, size=3)
        if not point_in_join_3d(first, second, point):
            continue
        length, u, v = ell_triangles_3d(first, second, point)
        sampled = reference_ell(first, second, point)
        if not np.isfinite(length) or not np.isfinite(sampled):
            continue
        # The witness is admissible, so the value is attained and cannot be too
        # small; a dense scan can only overestimate, so it cannot be too large.
        assert_admissible(length, u, v, point)
        assert length <= sampled + 1e-9
        assert length == pytest.approx(sampled, rel=1e-1)
        compared += 1
    assert compared > 10


def test_membership_predicate_matches_the_linear_program() -> None:
    generator = np.random.default_rng(31415)
    for _ in range(400):
        first = generator.uniform(-1.0, 1.0, size=(3, 3))
        second = generator.uniform(-1.0, 1.0, size=(3, 3))
        point = generator.uniform(-1.0, 1.0, size=3)
        pieces = (ConvexPolytope(first), ConvexPolytope(second))
        expected = convex_hull_coordinates(point, pieces) is not None
        assert point_in_join_3d(first, second, point) is expected


def test_membership_handles_triangles_sharing_an_edge() -> None:
    first = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    second = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert point_in_join_3d(first, second, [0.2, 0.2, 0.2])
    assert not point_in_join_3d(first, second, [0.6, 0.6, 0.6])
    # Coplanar triangles: the join is flat, so membership needs the plane test.
    flat = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert point_in_join_3d(first, flat, [0.5, 0.5, 0.0])
    assert not point_in_join_3d(first, flat, [0.5, 0.5, 0.05])


def test_point_triangle_distance_matches_the_polytope_routine() -> None:
    generator = np.random.default_rng(1234)
    for _ in range(200):
        triangle = generator.uniform(-1.0, 1.0, size=(3, 3))
        point = generator.uniform(-2.0, 2.0, size=3)
        expected = ConvexPolytope(triangle).distance(point)
        assert point_triangle_distance(triangle, point) == pytest.approx(expected, abs=1e-8)


def test_degenerate_plane_case_is_reported_rather_than_guessed() -> None:
    first = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    second = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    with pytest.raises(NotImplementedError):
        ell_triangles_3d(first, second, [0.2, 0.2, 0.0])


def test_parallel_facets_give_the_common_normal_chord() -> None:
    first = np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]])
    second = np.array([[-2.0, -2.0, 1.0], [2.0, -2.0, 1.0], [0.0, 2.0, 1.0]])
    length, u, v = ell_triangles_3d(first, second, [0.0, 0.0, 0.25])
    assert length == pytest.approx(1.0)
    assert_admissible(length, u, v, np.array([0.0, 0.0, 0.25]))


def test_edge_pair_stratum_degenerates_without_losing_the_minimum() -> None:
    """The sweep plane may contain a whole edge of the second triangle.

    On the cube the plane through the origin and the edge {(x,1,1)} is {y=z},
    which contains the opposite edge {(x,-1,-1)}.  The edge-pair stratum is then
    a curve rather than a point, and the enumeration has to pick the minimum up
    from the edge stratum instead.
    """
    first = np.array([[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]])
    second = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0]])
    point = np.zeros(3)

    sweep = np.cross(first[0] - point, first[1] - first[0])
    assert sweep @ (second[1] - second[0]) == pytest.approx(0.0, abs=1e-12)
    assert sweep @ (second[0] - point) == pytest.approx(0.0, abs=1e-12)

    length, u, v = ell_triangles_3d(first, second, point)
    expected = ell_between_polytopes_by_program(
        ConvexPolytope(first), ConvexPolytope(second), point
    ).length
    assert length == pytest.approx(expected, abs=1e-9)
    assert length == pytest.approx(2.0, abs=1e-9)
    assert_admissible(length, u, v, point)


def test_degenerate_edge_pairs_match_the_program_in_bulk() -> None:
    """Random configurations built so that the sweep plane contains the edge."""
    generator = np.random.default_rng(2024)
    compared = 0
    for _ in range(300):
        point = generator.uniform(-0.3, 0.3, size=3)
        origin = generator.uniform(-1.0, 1.0, size=3)
        edge = generator.uniform(-1.0, 1.0, size=3)
        radial = origin - point
        normal = np.cross(radial, edge)
        if np.linalg.norm(normal) < 1e-3:
            continue
        normal = normal / np.linalg.norm(normal)
        base = point + generator.uniform(-1.5, -0.4) * radial + generator.uniform(-1.0, 1.0) * edge
        other = generator.uniform(-1.0, 1.0) * radial + generator.uniform(-1.0, 1.0) * edge
        first = np.array(
            [origin, origin + edge, origin + 0.5 * edge + generator.uniform(0.4, 1.5) * normal]
        )
        second = np.array(
            [base, base + other, base + 0.5 * other - generator.uniform(0.4, 1.5) * normal]
        )
        if not point_in_join_3d(first, second, point):
            continue
        try:
            length, u, v = ell_triangles_3d(first, second, point)
        except NotImplementedError:
            continue
        if not np.isfinite(length):
            continue
        expected = ell_between_polytopes_by_program(
            ConvexPolytope(first), ConvexPolytope(second), point
        )
        assert length == pytest.approx(expected.length, rel=1e-7, abs=1e-9)
        assert_admissible(length, u, v, point)
        compared += 1
    assert compared > 40
