# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Algorithms for computing the width function."""

from width_function.algorithms.adaptive_epsilon_approximation import (
    AdaptiveApproximationResult,
    adaptive_epsilon_width_2d,
    algorithm_2,
    direction_set,
    directional_widths_polygon,
    heuristic_adaptive_width_2d,
)
from width_function.algorithms.computing_width import WidthResult, algorithm_1

__all__ = [
    "AdaptiveApproximationResult",
    "WidthResult",
    "adaptive_epsilon_width_2d",
    "algorithm_1",
    "algorithm_2",
    "direction_set",
    "directional_widths_polygon",
    "heuristic_adaptive_width_2d",
]
