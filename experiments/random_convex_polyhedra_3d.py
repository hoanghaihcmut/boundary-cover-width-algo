# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Random convex polyhedra in R^3."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError

from width_function.algorithms import WidthResult, algorithm_1
from width_function.geometry import (
    ConvexPolytope,
    EllWitness,
    contains_in_join,
    ell_between_polytopes_with_witness,
)


def median_runtime(call: Callable[[], object], repeats: int) -> float:
    """Return the median duration of ``repeats`` executions."""
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        durations.append(perf_counter() - start)
    return float(np.median(durations))


def random_convex_polyhedron(
    generator: np.random.Generator,
    candidate_vertex_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and triangular boundary facets of a convex 3-polytope."""
    if candidate_vertex_count < 4:
        raise ValueError("a three-dimensional polyhedron requires at least four vertices")

    for _ in range(100):
        directions = generator.normal(size=(candidate_vertex_count, 3))
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms == 0.0):
            continue
        directions /= norms[:, np.newaxis]

        # An anisotropic linear map avoids making every case nearly spherical.
        axis_scales = generator.uniform(0.72, 1.28, size=3)
        rotation_seed = generator.normal(size=(3, 3))
        rotation, _ = np.linalg.qr(rotation_seed)
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 0] *= -1.0
        candidates = (directions * axis_scales) @ rotation.T

        try:
            hull = ConvexHull(candidates)
        except QhullError:
            continue
        if hull.volume <= 1e-8 or len(hull.vertices) < 4:
            continue

        vertex_indices = np.asarray(hull.vertices, dtype=int)
        remap = {int(old): new for new, old in enumerate(vertex_indices)}
        vertices = candidates[vertex_indices]
        facets = np.asarray(
            [[remap[int(index)] for index in simplex] for simplex in hull.simplices],
            dtype=int,
        )
        return vertices, facets
    raise RuntimeError("failed to generate a full-dimensional convex polyhedron")


def sample_interior_points(
    vertices: np.ndarray,
    point_count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Sample reproducible strict convex combinations and include the centroid."""
    if point_count < 1:
        raise ValueError("point_count must be positive")
    points = []
    for _ in range(point_count - 1):
        weights = generator.dirichlet(np.full(len(vertices), 2.0))
        points.append(weights @ vertices)
    points.append(np.mean(vertices, axis=0))
    return np.asarray(points, dtype=float)


def polyhedron_boundary(vertices: np.ndarray, facets: np.ndarray) -> list[ConvexPolytope]:
    """Represent the boundary cover by compact convex triangular facets."""
    return [ConvexPolytope(vertices[facet]) for facet in facets]


def pointwise_algorithm_1(
    boundary: list[ConvexPolytope], query_points: np.ndarray
) -> tuple[np.ndarray, int]:
    """Compute every local width by running Algorithm 1 with ``G = {p_k}``."""
    results = [algorithm_1(boundary, point[np.newaxis, :]) for point in query_points]
    widths = np.asarray([result.width for result in results], dtype=float)
    ell_evaluations = sum(result.ell_evaluations for result in results)
    return widths, ell_evaluations


def exact_chord_3d(
    boundary: list[ConvexPolytope], point: np.ndarray
) -> tuple[int, int, EllWitness]:
    """Recover a shortest boundary-to-boundary segment through one point."""
    best: tuple[int, int, EllWitness] | None = None
    for first_index, second_index in combinations(range(len(boundary)), 2):
        first, second = boundary[first_index], boundary[second_index]
        if not contains_in_join(first, second, point):
            continue
        witness = ell_between_polytopes_with_witness(first, second, point)
        if best is None or witness.length < best[2].length:
            best = (first_index, second_index, witness)
    if best is None:
        raise RuntimeError("no boundary-facet chord was found for the selected point")
    return best


def maximizer_index(result: WidthResult) -> int | None:
    """Return a maximizer whose local width was evaluated without global pruning."""
    candidates = np.flatnonzero(
        np.isfinite(result.point_widths)
        & np.isclose(result.point_widths, result.width, rtol=1e-7, atol=1e-8)
    )
    return int(candidates[0]) if len(candidates) else None


def recover_maximizer_and_chord(
    boundary: list[ConvexPolytope],
    query_points: np.ndarray,
    result: WidthResult,
) -> tuple[int, int, int, EllWitness]:
    """Find a visual witness, including the rare all-globally-pruned case."""
    selected = maximizer_index(result)
    if selected is not None:
        first, second, witness = exact_chord_3d(boundary, query_points[selected])
        if np.isclose(witness.length, result.width, rtol=1e-7, atol=1e-8):
            return selected, first, second, witness

    for point_index, point in enumerate(query_points):
        first, second, witness = exact_chord_3d(boundary, point)
        if np.isclose(witness.length, result.width, rtol=1e-7, atol=1e-8):
            return point_index, first, second, witness
    raise AssertionError("no chord witness agrees with the width returned by Algorithm 1")


def _set_equal_3d_limits(axis: object, vertices: np.ndarray) -> None:
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.55 * float(np.max(upper - lower))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def crop_white_margins(path: Path, padding_fraction: float = 0.04) -> None:
    """Crop an image to its nonwhite content while retaining a small border."""
    image = plt.imread(path)
    content_mask = np.min(image[..., :3], axis=2) < 0.985
    rows, columns = np.nonzero(content_mask)
    if len(rows) == 0:
        return
    padding = max(4, round(padding_fraction * max(np.ptp(rows), np.ptp(columns))))
    top = max(0, int(rows.min()) - padding)
    bottom = min(image.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(image.shape[1], int(columns.max()) + padding + 1)
    plt.imsave(path, image[top:bottom, left:right])


def save_maximizer_figure(
    path: Path,
    vertices: np.ndarray,
    facets: np.ndarray,
    query_points: np.ndarray,
    point_index: int,
    witness: EllWitness,
    *,
    azimuth: float = -58.0,
) -> None:
    """Draw the 3D boundary cover, query set, maximizer, and exact chord."""
    figure = plt.figure(figsize=(5.2, 4.6))
    axis = figure.add_subplot(111, projection="3d")
    triangles = [vertices[facet] for facet in facets]
    collection = Poly3DCollection(
        triangles,
        facecolor="0.72",
        edgecolor="0.25",
        linewidth=0.65,
        alpha=0.28,
    )
    axis.add_collection3d(collection)
    axis.scatter(
        query_points[:, 0],
        query_points[:, 1],
        query_points[:, 2],
        color="0.4",
        s=15,
        depthshade=False,
    )
    axis.plot(
        [witness.start[0], witness.end[0]],
        [witness.start[1], witness.end[1]],
        [witness.start[2], witness.end[2]],
        color="tab:blue",
        linewidth=3.0,
        zorder=5,
    )
    axis.scatter(
        [witness.start[0], witness.end[0]],
        [witness.start[1], witness.end[1]],
        [witness.start[2], witness.end[2]],
        color="tab:blue",
        marker="s",
        s=28,
        depthshade=False,
    )
    selected = query_points[point_index]
    axis.scatter(
        selected[0],
        selected[1],
        selected[2],
        color="red",
        s=52,
        depthshade=False,
        zorder=6,
    )
    _set_equal_3d_limits(axis, vertices)
    axis.view_init(elev=22.0, azim=azimuth)
    axis.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)
    crop_white_margins(path)


def save_sequence_figure(
    path: Path,
    point_widths: np.ndarray,
    omega: np.ndarray,
) -> None:
    """Plot the two sequences without repeated titles, labels, or legends."""
    point_indices = np.arange(1, len(point_widths) + 1)
    omega_indices = np.arange(len(omega))
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.plot(
        point_indices,
        point_widths,
        "o-",
        color="tab:orange",
        linewidth=1.8,
        markersize=4,
    )
    axis.plot(
        omega_indices,
        omega,
        "o-",
        color="tab:blue",
        linewidth=2.0,
        markersize=4,
    )
    tick_step = max(1, len(point_widths) // 4)
    axis.set_xticks(np.arange(0, len(omega), tick_step))
    axis.tick_params(labelsize=8)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def save_geometry_sequence_panel(path: Path, geometry_path: Path, sequence_path: Path) -> None:
    """Place one polyhedron and its sequence plot side by side without labels."""
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    for axis, image_path in zip(axes, (geometry_path, sequence_path), strict=True):
        axis.imshow(plt.imread(image_path))
        axis.axis("off")
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.005)
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def save_panel_grid(paths: list[Path], output_path: Path) -> None:
    """Arrange six geometry-and-sequence panels in three rows of two."""
    column_count = min(2, len(paths))
    row_count = (len(paths) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(10.6 * column_count, 4.3 * row_count),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, image_path in zip(flat_axes, paths, strict=False):
        axis.imshow(plt.imread(image_path))
        axis.axis("off")
    for axis in flat_axes[len(paths) :]:
        axis.axis("off")
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.005, hspace=0.005)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def _write_points(path: Path, points: np.ndarray, prefix: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([f"{prefix}_index", "x", "y", "z"])
        for index, point in enumerate(points, start=1):
            writer.writerow([index, *point])


def run_case(
    candidate_vertex_count: int,
    query_point_count: int,
    output_dir: Path,
    random_seed: int,
    timing_repeats: int,
    *,
    azimuth: float = -58.0,
) -> dict[str, object]:
    """Run exact Algorithm 1 on one random convex polyhedron in R^3."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(random_seed)
    vertices, facets = random_convex_polyhedron(generator, candidate_vertex_count)
    query_points = sample_interior_points(vertices, query_point_count, generator)
    boundary = polyhedron_boundary(vertices, facets)

    _write_points(
        output_dir / "random_convex_polyhedra_3d_convex_polyhedron_Q.csv", vertices, "vertex"
    )
    _write_points(output_dir / "random_convex_polyhedra_3d_query_set_G.csv", query_points, "p")
    with (output_dir / "random_convex_polyhedra_3d_boundary_facets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["facet", "vertex_1", "vertex_2", "vertex_3"])
        for index, facet in enumerate(facets, start=1):
            writer.writerow([index, *(facet + 1)])

    result = algorithm_1(boundary, query_points)
    full_g_elapsed = median_runtime(lambda: algorithm_1(boundary, query_points), timing_repeats)
    pointwise_widths, pointwise_ell_evaluations = pointwise_algorithm_1(boundary, query_points)
    pointwise_elapsed = median_runtime(
        lambda: pointwise_algorithm_1(boundary, query_points), timing_repeats
    )
    point_index, first_facet, second_facet, witness = recover_maximizer_and_chord(
        boundary, query_points, result
    )
    if not np.isclose(witness.length, result.width, rtol=1e-7, atol=1e-8):
        raise AssertionError("the 3D chord witness does not match Algorithm 1")
    if result.w.shape != (len(query_points) + 1,) or np.any(np.diff(result.w) < -1e-9):
        raise AssertionError("Algorithm 1 returned an invalid incumbent sequence")
    reconstructed_omega = np.concatenate(
        [
            [result.w[0]],
            np.maximum(result.w[0], np.maximum.accumulate(pointwise_widths[result.order])),
        ]
    )
    if not np.allclose(result.w, reconstructed_omega, rtol=1e-7, atol=1e-8):
        raise AssertionError("singleton widths do not reconstruct Algorithm 1's omega sequence")
    if not np.isclose(np.max(pointwise_widths), result.width, rtol=1e-7, atol=1e-8):
        raise AssertionError("singleton and full-G implementations disagree on W(Q;G)")

    with (output_dir / "random_convex_polyhedra_3d_algorithm_1_sequence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["k", "omega_k"])
        writer.writerows(enumerate(result.w))

    with (output_dir / "random_convex_polyhedra_3d_width_sequences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["position", "p_k", "W(Q;p_k)_singleton", "omega_k_full_G"])
        for position, index in enumerate(result.order, start=1):
            writer.writerow([position, int(index) + 1, pointwise_widths[index], result.w[position]])

    figure_path = output_dir / "random_convex_polyhedra_3d_maximizer.png"
    save_maximizer_figure(
        figure_path,
        vertices,
        facets,
        query_points,
        point_index,
        witness,
        azimuth=azimuth,
    )
    sequence_figure_path = output_dir / "random_convex_polyhedra_3d_sequences.png"
    save_sequence_figure(
        sequence_figure_path,
        pointwise_widths[result.order],
        result.w,
    )

    summary: dict[str, object] = {
        "dimension": 3,
        "polyhedron_type": "random convex polyhedron with triangular boundary facets",
        "random_seed": random_seed,
        "number_of_candidate_vertices": candidate_vertex_count,
        "number_of_polyhedron_vertices": len(vertices),
        "number_of_boundary_facets": len(facets),
        "number_of_query_points": len(query_points),
        "algorithm_1_width": result.width,
        "algorithm_1_time_seconds": full_g_elapsed,
        "algorithm_1_full_G_time_seconds": full_g_elapsed,
        "pointwise_singletons_time_seconds": pointwise_elapsed,
        "full_G_speedup_vs_singletons": pointwise_elapsed / full_g_elapsed,
        "algorithm_1_maximizer_p_k": point_index + 1,
        "maximizing_facet_pair": [first_facet + 1, second_facet + 1],
        "maximizing_point": query_points[point_index].tolist(),
        "chord_start": witness.start.tolist(),
        "chord_end": witness.end.tolist(),
        "chord_length": witness.length,
        "initial_lower_bound_omega_0": float(result.w[0]),
        "w_sequence": result.w.tolist(),
        "pointwise_widths": pointwise_widths.tolist(),
        "processing_order": (result.order + 1).tolist(),
        "membership_tests": result.membership_tests,
        "feasible_pairs": result.feasible_pairs,
        "ell_evaluations": result.ell_evaluations,
        "pointwise_ell_evaluations": pointwise_ell_evaluations,
        "pairwise_prunes": result.pairwise_prunes,
        "global_prunes": result.global_prunes,
    }
    (output_dir / "random_convex_polyhedra_3d_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_suite(
    candidate_vertex_count: int,
    query_point_count: int,
    polyhedron_count: int,
    output_dir: Path,
    random_seed: int = 20260840,
    timing_repeats: int = 3,
) -> dict[str, object]:
    """Run the random convex polyhedron experiment in R^3."""
    output_dir.mkdir(parents=True, exist_ok=True)
    case_summaries = []
    result_rows = []
    panel_paths = []

    for polyhedron_index in range(1, polyhedron_count + 1):
        seed = random_seed + polyhedron_index - 1
        case_dir = output_dir / f"polyhedron_{polyhedron_index}"
        summary = run_case(
            candidate_vertex_count,
            query_point_count,
            case_dir,
            seed,
            timing_repeats,
            azimuth=-62.0 + 9.0 * (polyhedron_index - 1),
        )
        summary["polyhedron_index"] = polyhedron_index
        case_summaries.append(summary)
        result_rows.append(
            {
                "polyhedron": f"Q_{polyhedron_index}",
                "random_seed": seed,
                "dimension": summary["dimension"],
                "vertices": summary["number_of_polyhedron_vertices"],
                "boundary_facets": summary["number_of_boundary_facets"],
                "query_points": summary["number_of_query_points"],
                "maximizer_p_k": summary["algorithm_1_maximizer_p_k"],
                "W(Q;G)": summary["algorithm_1_width"],
                "omega_0": summary["initial_lower_bound_omega_0"],
                "algorithm_1_full_G_time_seconds": summary["algorithm_1_full_G_time_seconds"],
                "pointwise_singletons_time_seconds": summary["pointwise_singletons_time_seconds"],
                "full_G_speedup_vs_singletons": summary["full_G_speedup_vs_singletons"],
                "membership_tests": summary["membership_tests"],
                "feasible_pairs": summary["feasible_pairs"],
                "full_G_ell_evaluations": summary["ell_evaluations"],
                "pointwise_ell_evaluations": summary["pointwise_ell_evaluations"],
                "pairwise_prunes": summary["pairwise_prunes"],
                "global_prunes": summary["global_prunes"],
            }
        )

        geometry_path = case_dir / "random_convex_polyhedra_3d_maximizer.png"
        sequence_path = case_dir / "random_convex_polyhedra_3d_sequences.png"
        panel_path = case_dir / "random_convex_polyhedra_3d_geometry_and_sequences.png"
        save_geometry_sequence_panel(panel_path, geometry_path, sequence_path)
        panel_paths.append(panel_path)

    with (output_dir / "random_convex_polyhedra_3d.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_rows)

    combined_figure = output_dir / "random_convex_polyhedra_3d.png"
    save_panel_grid(panel_paths, combined_figure)

    suite_summary: dict[str, object] = {
        "dimension": 3,
        "polyhedron_type": "random convex polyhedra with triangular boundary facets",
        "number_of_polyhedra": polyhedron_count,
        "candidate_vertices_per_polyhedron": candidate_vertex_count,
        "query_points_per_polyhedron": query_point_count,
        "base_random_seed": random_seed,
        "timing_repeats": timing_repeats,
        "cases": case_summaries,
        "result_rows": result_rows,
    }
    (output_dir / "random_convex_polyhedra_3d.json").write_text(
        json.dumps(suite_summary, indent=2) + "\n", encoding="utf-8"
    )
    return suite_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices", type=int, default=6, help="candidate vertices per case")
    parser.add_argument("--query-points", type=int, default=20)
    parser.add_argument("--polyhedron-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260840)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/random_convex_polyhedra_3d")
    )
    args = parser.parse_args()
    if args.vertices < 4:
        parser.error("--vertices must be at least 4")
    if args.query_points < 1:
        parser.error("--query-points must be positive")
    if args.polyhedron_count < 1:
        parser.error("--polyhedron-count must be positive")
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be positive")

    suite = run_suite(
        args.vertices,
        args.query_points,
        args.polyhedron_count,
        args.output_dir,
        args.seed,
        args.timing_repeats,
    )
    report = {
        "dimension": suite["dimension"],
        "number_of_polyhedra": suite["number_of_polyhedra"],
        "results_file": str(args.output_dir / "random_convex_polyhedra_3d.csv"),
        "figure_file": str(args.output_dir / "random_convex_polyhedra_3d.png"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
