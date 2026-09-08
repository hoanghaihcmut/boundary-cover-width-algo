# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Dimension-independent geometric primitives."""

from width_function.geometry.convex_polytope import (
    ConvexPolytope,
    EllWitness,
    contains_in_join,
    ell_between_polytopes,
    ell_between_polytopes_by_program,
    ell_between_polytopes_with_witness,
)
from width_function.geometry.numerical_medial_skeleton import (
    SkeletonSample,
    sample_convex_polygon_skeleton,
    sample_polygon_skeleton,
)
from width_function.geometry.planar_polygons import ell_segments_2d, point_in_join_2d
from width_function.geometry.triangulated_polyhedra import (
    ell_triangles_3d,
    point_in_join_3d,
    point_triangle_distance,
)

__all__ = [
    "ConvexPolytope",
    "EllWitness",
    "SkeletonSample",
    "contains_in_join",
    "ell_between_polytopes",
    "ell_between_polytopes_by_program",
    "ell_between_polytopes_with_witness",
    "ell_segments_2d",
    "ell_triangles_3d",
    "point_in_join_2d",
    "point_in_join_3d",
    "point_triangle_distance",
    "sample_convex_polygon_skeleton",
    "sample_polygon_skeleton",
]
