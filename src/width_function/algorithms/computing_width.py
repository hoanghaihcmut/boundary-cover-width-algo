# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Computing W(Q; G) in R^d."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush

import numpy as np
from numpy.typing import ArrayLike, NDArray

from width_function.geometry.convex_polytope import (
    ConvexPolytope,
    contains_in_join,
    ell_between_polytopes,
)


@dataclass(frozen=True)
class WidthResult:
    """Result and counters useful for checking the two pruning rules."""

    width: float
    point_widths: NDArray[np.float64]
    w: NDArray[np.float64]
    order: NDArray[np.int64]
    feasible_pairs: int
    membership_tests: int
    ell_evaluations: int
    pairwise_prunes: int
    global_prunes: int

    @property
    def omega(self) -> NDArray[np.float64]:
        """Alias for the incumbent sequence produced by Algorithm 1."""
        return self.w


def algorithm_1(
    boundary_pieces: list[ConvexPolytope] | tuple[ConvexPolytope, ...],
    query_points: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> WidthResult:
    """Compute ``max_q W(Q; q)`` for a convex-polytope boundary cover in R^d.

    The ambient dimension is inferred from the pieces; it is never hard-coded.

    Query points are processed in nonincreasing order of their clearance
    ``dist(q, boundary)``, so that the incumbent rises early and the global
    pruning rule fires often; ``result.order`` records that permutation and
    ``result.w`` holds ``[w_0, w_1, ..., w_n]`` along it, so it has ``n + 1``
    entries.  Any order returns the same width -- the reordering is a
    heuristic.  Candidate pairs are emitted lazily, in nondecreasing order of
    ``dist(q, E_i) + dist(q, E_j)``, by a heap holding one pair per sequence;
    the membership test is therefore paid only for the pairs actually reached,
    not for all ``binom(N, 2)`` of them.

    ``point_widths[k]`` refers to ``query_points[k]`` and is NaN when global
    pruning proved that point cannot increase the returned maximum without
    computing its exact local width.
    """
    pieces = tuple(boundary_pieces)
    if len(pieces) < 2:
        raise ValueError("the boundary cover must contain at least two pieces")
    dimension = pieces[0].dimension
    if any(piece.dimension != dimension for piece in pieces):
        raise ValueError("all boundary pieces must lie in the same R^d")
    points = np.asarray(query_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != dimension or not len(points):
        raise ValueError(f"query_points must be a nonempty (n, {dimension}) array")
    if not np.all(np.isfinite(points)):
        raise ValueError("query_points must be finite")

    distances = np.array([[piece.distance(q) for piece in pieces] for q in points])
    clearances = np.min(distances, axis=1)
    order = np.argsort(-clearances, kind="stable")
    omega_values = [2.0 * float(clearances[order[0]])]
    local_widths = np.full(len(points), np.nan)
    feasible_count = membership_count = ell_count = pair_prunes = global_prunes = 0

    for index in order:
        upper = np.inf
        exact = True
        for i, j, bound in _pairs_by_increasing_bound(distances[index]):
            if bound >= upper - tolerance:
                pair_prunes += 1
                break
            membership_count += 1
            if not contains_in_join(pieces[i], pieces[j], points[index], tolerance=tolerance):
                continue
            feasible_count += 1
            length = ell_between_polytopes(pieces[i], pieces[j], points[index], tolerance=tolerance)
            ell_count += 1
            if length <= omega_values[-1] + tolerance:
                upper = length
                exact = False
                global_prunes += 1
                break
            upper = min(upper, length)
        if not np.isfinite(upper):
            raise ValueError(f"no feasible boundary-piece pair for query point {index}")
        if exact:
            local_widths[index] = upper
        omega_values.append(max(omega_values[-1], upper if exact else 0.0))

    return WidthResult(
        width=omega_values[-1],
        point_widths=local_widths,
        w=np.asarray(omega_values),
        order=np.asarray(order, dtype=np.int64),
        feasible_pairs=feasible_count,
        membership_tests=membership_count,
        ell_evaluations=ell_count,
        pairwise_prunes=pair_prunes,
        global_prunes=global_prunes,
    )


def _pairs_by_increasing_bound(distances: NDArray[np.float64]):
    """Emit ``(i, j, d_i + d_j)`` for ``i < j``, by nondecreasing sum.

    Sorting the pieces by distance splits the pairs into ``N - 1`` sequences
    with nondecreasing key; a heap holding the first unemitted pair of each
    needs ``O(N)`` memory and ``O(log N)`` time per pair, and at any break
    every unemitted pair has key at least the last emitted one.
    """
    count = len(distances)
    sigma = np.argsort(distances, kind="stable")
    sorted_distances = distances[sigma]
    heap = [
        (float(sorted_distances[rank] + sorted_distances[rank + 1]), rank, rank + 1)
        for rank in range(count - 1)
    ]
    heapify(heap)
    while heap:
        bound, rank, follower = heappop(heap)
        if follower + 1 < count:
            heappush(
                heap,
                (
                    float(sorted_distances[rank] + sorted_distances[follower + 1]),
                    rank,
                    follower + 1,
                ),
            )
        first, second = int(sigma[rank]), int(sigma[follower])
        yield (first, second, bound) if first < second else (second, first, bound)
