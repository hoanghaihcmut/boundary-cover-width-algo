# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Root finding for the cubics that the exact evaluation of ell reduces to.

Both the planar primitive and its three-dimensional counterpart end at the same
stationarity condition,

    P(s) = 2 beta (alpha - gamma) eta(s) + (gamma + beta s)(alpha + beta s) eta'(s),

a polynomial of degree at most three, so the solver lives here and is shared.
"""

from __future__ import annotations

import math

_RELATIVE_ZERO = 1e-14


def real_roots(cubic: float, quadratic: float, linear: float, constant: float) -> list[float]:
    """Real roots of a polynomial of degree at most three, coefficients descending."""
    magnitude = max(abs(cubic), abs(quadratic), abs(linear), abs(constant))
    if magnitude == 0.0:
        return []
    threshold = _RELATIVE_ZERO * magnitude
    if abs(cubic) <= threshold:
        if abs(quadratic) <= threshold:
            return [] if abs(linear) <= threshold else [-constant / linear]
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant < 0.0:
            return []
        root = math.sqrt(discriminant)
        return [(-linear + root) / (2.0 * quadratic), (-linear - root) / (2.0 * quadratic)]

    a = quadratic / cubic
    b = linear / cubic
    c = constant / cubic
    shift = a / 3.0
    depressed_linear = b - a * a / 3.0
    depressed_constant = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c
    half = depressed_constant / 2.0
    third = depressed_linear / 3.0
    discriminant = half * half + third * third * third

    if discriminant > 0.0:
        root = math.sqrt(discriminant)
        roots = [math.cbrt(-half + root) + math.cbrt(-half - root) - shift]
    elif discriminant == 0.0:
        if depressed_linear == 0.0:
            roots = [-shift]
        else:
            roots = [
                3.0 * depressed_constant / depressed_linear - shift,
                -1.5 * depressed_constant / depressed_linear - shift,
            ]
    else:
        radius = 2.0 * math.sqrt(-third)
        cosine = 3.0 * depressed_constant / (depressed_linear * radius)
        angle = math.acos(max(-1.0, min(1.0, cosine)))
        roots = [
            radius * math.cos((angle - 2.0 * math.pi * index) / 3.0) - shift for index in range(3)
        ]
    return [_polish(root, cubic, quadratic, linear, constant) for root in roots]


def _polish(root: float, cubic: float, quadratic: float, linear: float, constant: float) -> float:
    """One Newton step against the original coefficients."""
    value = ((cubic * root + quadratic) * root + linear) * root + constant
    slope = (3.0 * cubic * root + 2.0 * quadratic) * root + linear
    if slope == 0.0 or not math.isfinite(value / slope):
        return root
    return root - value / slope


def stationary_candidates(
    alpha: float, beta: float, gamma: float, eta_0: float, eta_1: float, eta_2: float
) -> list[float]:
    """Real roots of the stationarity cubic ``P``.

    ``eta(s) = eta_0 + eta_1 s + eta_2 s^2`` is the squared distance from the
    query point to the parametrised first piece. The coefficients ``alpha``,
    ``beta``, and ``gamma`` occur both for a planar edge pair and for an edge
    of one triangle tested against the plane of another.
    """
    leading = 2.0 * beta * (alpha - gamma)
    return real_roots(
        2.0 * beta * beta * eta_2,
        leading * eta_2 + 2.0 * beta * (alpha + gamma) * eta_2 + beta * beta * eta_1,
        leading * eta_1 + 2.0 * alpha * gamma * eta_2 + beta * (alpha + gamma) * eta_1,
        leading * eta_0 + alpha * gamma * eta_1,
    )
