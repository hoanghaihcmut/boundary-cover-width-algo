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

pytest.importorskip("cv2")

from width_function.applications import (
    measure_polygon,
    merge_detection_polygons,
    points_strictly_inside,
)


def test_rectangle_measurement_returns_realizing_segment() -> None:
    rectangle = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])

    result = measure_polygon(
        rectangle,
        query_points=[[2.0, 1.0]],
        simplify_tolerance=0.0,
        max_vertices=4,
    )

    assert result.width == pytest.approx(2.0, abs=1e-7)
    assert np.linalg.norm(result.segment_end - result.segment_start) == pytest.approx(result.width)


def test_detection_polygons_are_unioned_before_measurement() -> None:
    left = np.array([[2, 4], [12, 4], [12, 8], [2, 8]], dtype=float)
    right = np.array([[10, 4], [22, 4], [22, 8], [10, 8]], dtype=float)

    components = merge_detection_polygons(
        [left, right],
        image_shape=(14, 28),
        minimum_area=5.0,
        close_kernel=1,
        simplify_tolerance=0.0,
        max_vertices=20,
    )

    assert len(components) == 1
    assert points_strictly_inside(
        [[5.0, 6.0], [18.0, 6.0], [25.0, 6.0]], components[0]
    ).tolist() == [True, True, False]
