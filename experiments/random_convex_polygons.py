# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Random convex polygons."""

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
from scipy.spatial import ConvexHull

from width_function.algorithms import (
    algorithm_1,
    algorithm_2,
    direction_set,
    directional_widths_polygon,
)
from width_function.geometry import (
    ConvexPolytope,
    contains_in_join,
    ell_between_polytopes_with_witness,
    sample_convex_polygon_skeleton,
)

EPSILONS = tuple(10.0**-power for power in range(1, 6))


def polygon_boundary(vertices: np.ndarray) -> list[ConvexPolytope]:
    return [
        ConvexPolytope([start, end])
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    ]


def median_runtime(call: Callable[[], object], repeats: int) -> float:
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        durations.append(perf_counter() - start)
    return float(np.median(durations))


def exact_maximizer_index(
    boundary: list[ConvexPolytope],
    query_points: np.ndarray,
    exact_width: float,
    point_widths: np.ndarray,
) -> int:
    """Return a p_k whose exact local width equals W(Q; G)."""
    candidates = np.flatnonzero(
        np.isfinite(point_widths) & np.isclose(point_widths, exact_width, rtol=1e-8, atol=1e-9)
    )
    if len(candidates):
        return int(candidates[0])
    local_widths = np.array(
        [algorithm_1(boundary, point[np.newaxis, :]).width for point in query_points]
    )
    return int(np.argmax(local_widths))


def exact_chord(
    boundary: list[ConvexPolytope], point: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover an endpoint witness for Algorithm 1 at one query point."""
    best = None
    for first_index, second_index in combinations(range(len(boundary)), 2):
        first, second = boundary[first_index], boundary[second_index]
        if not contains_in_join(first, second, point):
            continue
        witness = ell_between_polytopes_with_witness(first, second, point)
        if best is None or witness.length < best.length:
            best = witness
    if best is None:
        raise RuntimeError("no exact chord witness found for the selected point")
    return best.start, best.end, best.length


def directional_chord(
    polygon: np.ndarray, point: np.ndarray, direction: np.ndarray, tolerance: float = 1e-10
) -> tuple[np.ndarray, np.ndarray, float]:
    """Intersect the line p + t*direction with the polygon."""
    unit_direction = direction / np.linalg.norm(direction)
    parameters = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = end - start
        matrix = np.column_stack([unit_direction, -edge])
        if abs(np.linalg.det(matrix)) <= tolerance:
            continue
        line_parameter, edge_parameter = np.linalg.solve(matrix, start - point)
        if -tolerance <= edge_parameter <= 1.0 + tolerance:
            parameters.append(float(line_parameter))
    if len(parameters) < 2:
        raise RuntimeError("the selected direction does not produce a polygon chord")
    negative = [value for value in parameters if value < -tolerance]
    positive = [value for value in parameters if value > tolerance]
    if not negative or not positive:
        raise RuntimeError("the query point must lie strictly inside the polygon")
    lower, upper = max(negative), min(positive)
    return (
        point + lower * unit_direction,
        point + upper * unit_direction,
        upper - lower,
    )


def save_maximizer_figure(
    path: Path,
    polygon: np.ndarray,
    skeleton_segments: np.ndarray,
    query_points: np.ndarray,
    point_index: int,
    chord_start: np.ndarray,
    chord_end: np.ndarray,
) -> None:
    """Draw a label-free illustration of Q, G, the maximizer, and its chord."""
    closed_polygon = np.vstack([polygon, polygon[0]])
    figure, axis = plt.subplots(figsize=(3.2, 3.2))
    axis.plot(closed_polygon[:, 0], closed_polygon[:, 1], "k-", linewidth=2)
    for start, end in skeleton_segments:
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="tab:orange",
            linewidth=0.55,
            alpha=0.75,
        )
    axis.scatter(
        query_points[:, 0],
        query_points[:, 1],
        color="0.55",
        s=10,
        zorder=3,
    )
    axis.plot(
        [chord_start[0], chord_end[0]],
        [chord_start[1], chord_end[1]],
        color="tab:blue",
        linewidth=2.6,
        zorder=4,
    )
    axis.scatter(
        [chord_start[0], chord_end[0]],
        [chord_start[1], chord_end[1]],
        color="tab:blue",
        marker="s",
        s=22,
        zorder=5,
    )
    selected = query_points[point_index]
    axis.scatter(selected[0], selected[1], color="red", s=45, zorder=6)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout(pad=0.02)
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def save_illustration_grid(rows: list[list[Path]], path: Path) -> None:
    """Combine all label-free illustrations into rows of six images."""
    figure, axes = plt.subplots(len(rows), 6, figsize=(18, 3 * len(rows)), squeeze=False)
    for row_axes, image_paths in zip(axes, rows, strict=True):
        if len(image_paths) != 6:
            raise ValueError("each illustration row must contain exactly six images")
        for axis, image_path in zip(row_axes, image_paths, strict=True):
            axis.imshow(plt.imread(image_path))
            axis.axis("off")
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.015, hspace=0.015)
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def _write_points(path: Path, points: np.ndarray, prefix: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([f"{prefix}_index", "x", "y"])
        for index, point in enumerate(points, start=1):
            writer.writerow([index, *point])


def run(
    number_of_random_points: int,
    skeleton_point_count: int,
    output_dir: Path,
    random_seed: int = 20260820,
    timing_repeats: int = 3,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(random_seed)
    random_points = generator.uniform(-1.0, 1.0, size=(number_of_random_points, 2))
    hull = ConvexHull(random_points)
    polygon = random_points[hull.vertices]
    skeleton = sample_convex_polygon_skeleton(
        polygon,
        boundary_samples_per_edge=64,
        max_points=skeleton_point_count,
        min_clearance_fraction=0.04,
    )
    query_points = skeleton.points
    boundary = polygon_boundary(polygon)

    _write_points(output_dir / "random_convex_polygons_random_points.csv", random_points, "point")
    _write_points(output_dir / "random_convex_polygons_convex_polygon_Q.csv", polygon, "vertex")
    _write_points(output_dir / "random_convex_polygons_skeleton_G.csv", query_points, "p")
    with (output_dir / "random_convex_polygons_skeleton_segments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["segment", "x_1", "y_1", "x_2", "y_2"])
        for index, (start, end) in enumerate(skeleton.segments, start=1):
            writer.writerow([index, *start, *end])

    exact = algorithm_1(boundary, query_points)
    exact_time = median_runtime(lambda: algorithm_1(boundary, query_points), timing_repeats)

    theta_0 = float(generator.uniform(0.0, np.pi))
    adaptive_results = [
        algorithm_2(polygon, query_points, epsilon, theta_0=theta_0) for epsilon in EPSILONS
    ]
    adaptive_times = [
        median_runtime(
            lambda epsilon=epsilon: algorithm_2(polygon, query_points, epsilon, theta_0=theta_0),
            timing_repeats,
        )
        for epsilon in EPSILONS
    ]

    exact_point_index = exact_maximizer_index(
        boundary, query_points, exact.width, exact.point_widths
    )
    exact_start, exact_end, exact_chord_width = exact_chord(
        boundary, query_points[exact_point_index]
    )
    if not np.isclose(exact_chord_width, exact.width, rtol=1e-7, atol=1e-8):
        raise AssertionError("the exact visual witness does not match Algorithm 1")

    adaptive_visuals = []
    for result in adaptive_results:
        point_index = int(np.argmax(result.point_upper_bounds))
        level = int(result.levels[point_index])
        directions = direction_set(level, result.theta_0)
        widths = directional_widths_polygon(polygon, query_points[point_index], directions)
        direction_index = int(np.argmin(widths))
        direction = directions[direction_index]
        start, end, chord_width = directional_chord(polygon, query_points[point_index], direction)
        if not np.isclose(chord_width, result.width, rtol=1e-8, atol=1e-9):
            raise AssertionError("the adaptive visual witness does not match its upper bound")
        adaptive_visuals.append(
            {
                "point_index": point_index,
                "level": level,
                "direction": direction,
                "direction_angle_rad": float(np.mod(np.arctan2(direction[1], direction[0]), np.pi)),
                "start": start,
                "end": end,
                "width": chord_width,
            }
        )

    comparison_rows = []
    for result, elapsed, visual in zip(
        adaptive_results, adaptive_times, adaptive_visuals, strict=True
    ):
        actual_error = result.width - exact.width
        if actual_error < -1e-9 or actual_error > result.epsilon + 1e-9:
            raise AssertionError("adaptive result violates its epsilon guarantee")
        if result.certified_gap > result.epsilon + 1e-9:
            raise AssertionError("adaptive result stopped without a valid certificate")
        comparison_rows.append(
            {
                "epsilon": result.epsilon,
                "algorithm_1_W(Q;G)": exact.width,
                "adaptive_estimate": result.width,
                "actual_error": actual_error,
                "certified_lower_bound": result.lower_bound,
                "certified_upper_bound": result.upper_bound,
                "certified_gap": result.certified_gap,
                "maximizer_p_k": visual["point_index"] + 1,
                "maximizer_level": visual["level"],
                "maximizing_direction_rad": visual["direction_angle_rad"],
                "max_level": int(np.max(result.levels)),
                "levels_by_point": ";".join(str(level) for level in result.levels),
                "refinements": result.refinements,
                "direction_evaluations": result.direction_evaluations,
                "time_seconds": elapsed,
                "speedup_vs_algorithm_1": exact_time / elapsed,
            }
        )

    with (output_dir / "random_convex_polygons_adaptive_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison_rows)

    maximizer_rows = [
        {
            "method": "Algorithm 1 (exact)",
            "epsilon": "",
            "p_k": exact_point_index + 1,
            "x": query_points[exact_point_index, 0],
            "y": query_points[exact_point_index, 1],
            "width": exact_chord_width,
            "actual_error": 0.0,
            "certified_gap": 0.0,
            "time_seconds": exact_time,
            "membership_tests": exact.membership_tests,
            "ell_evaluations": exact.ell_evaluations,
            "direction_evaluations": "",
            "speedup_vs_algorithm_1": 1.0,
            "level": "",
            "direction_angle_rad": float(
                np.mod(np.arctan2(*(exact_end - exact_start)[::-1]), np.pi)
            ),
            "chord_x_1": exact_start[0],
            "chord_y_1": exact_start[1],
            "chord_x_2": exact_end[0],
            "chord_y_2": exact_end[1],
        }
    ]
    for result, elapsed, visual in zip(
        adaptive_results, adaptive_times, adaptive_visuals, strict=True
    ):
        point_index = int(visual["point_index"])
        maximizer_rows.append(
            {
                "method": "Adaptive epsilon-approximation",
                "epsilon": result.epsilon,
                "p_k": point_index + 1,
                "x": query_points[point_index, 0],
                "y": query_points[point_index, 1],
                "width": visual["width"],
                "actual_error": result.width - exact.width,
                "certified_gap": result.certified_gap,
                "time_seconds": elapsed,
                "membership_tests": "",
                "ell_evaluations": "",
                "direction_evaluations": result.direction_evaluations,
                "speedup_vs_algorithm_1": exact_time / elapsed,
                "level": visual["level"],
                "direction_angle_rad": visual["direction_angle_rad"],
                "chord_x_1": visual["start"][0],
                "chord_y_1": visual["start"][1],
                "chord_x_2": visual["end"][0],
                "chord_y_2": visual["end"][1],
            }
        )
    with (output_dir / "random_convex_polygons_maximizers.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(maximizer_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(maximizer_rows)

    with (output_dir / "random_convex_polygons_adaptive_point_bounds.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["epsilon", "p_k", "level", "lower_bound", "upper_bound"])
        for result in adaptive_results:
            for index in range(len(query_points)):
                writer.writerow(
                    [
                        result.epsilon,
                        index + 1,
                        result.levels[index],
                        result.point_lower_bounds[index],
                        result.point_upper_bounds[index],
                    ]
                )

    with (output_dir / "random_convex_polygons_algorithm_1_sequence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["k", "w_k"])
        writer.writerows(enumerate(exact.w))

    closed_polygon = np.vstack([polygon, polygon[0]])
    geometry_figure, geometry_axis = plt.subplots(figsize=(6, 6))
    geometry_axis.plot(closed_polygon[:, 0], closed_polygon[:, 1], "k-", linewidth=2)
    for start, end in skeleton.segments:
        geometry_axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="tab:orange",
            linewidth=0.7,
        )
    geometry_axis.scatter(
        query_points[:, 0],
        query_points[:, 1],
        color="red",
        s=28,
        zorder=3,
    )
    geometry_axis.set_aspect("equal")
    geometry_axis.axis("off")
    geometry_figure.tight_layout(pad=0.02)
    geometry_figure.savefig(
        output_dir / "random_convex_polygons_geometry.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(geometry_figure)

    save_maximizer_figure(
        output_dir / "random_convex_polygons_maximizer_exact.png",
        polygon,
        skeleton.segments,
        query_points,
        exact_point_index,
        exact_start,
        exact_end,
    )
    for result, visual in zip(adaptive_results, adaptive_visuals, strict=True):
        exponent = round(-np.log10(result.epsilon))
        save_maximizer_figure(
            output_dir / f"random_convex_polygons_maximizer_epsilon_1e-{exponent:02d}.png",
            polygon,
            skeleton.segments,
            query_points,
            int(visual["point_index"]),
            visual["start"],
            visual["end"],
        )

    epsilon_values = np.array([row["epsilon"] for row in comparison_rows], dtype=float)
    estimates = np.array([row["adaptive_estimate"] for row in comparison_rows], dtype=float)
    actual_errors = np.array([row["actual_error"] for row in comparison_rows], dtype=float)
    gaps = np.array([row["certified_gap"] for row in comparison_rows], dtype=float)
    times = np.array([row["time_seconds"] for row in comparison_rows], dtype=float)
    evaluations = np.array([row["direction_evaluations"] for row in comparison_rows], dtype=float)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].axhline(exact.width, color="black", label="Algorithm 1 (exact)")
    axes[0].semilogx(epsilon_values, estimates, "o--", label="Adaptive estimate")
    axes[0].invert_xaxis()
    axes[0].set(title="Width estimate", xlabel="epsilon", ylabel="W(Q;G)")
    axes[0].legend()

    axes[1].loglog(epsilon_values, epsilon_values, "k:", label="requested epsilon")
    axes[1].loglog(epsilon_values, gaps, "s--", label="certified gap")
    axes[1].loglog(epsilon_values, np.maximum(actual_errors, 1e-16), "o-", label="actual error")
    axes[1].invert_xaxis()
    axes[1].set(title="Error certificate", xlabel="epsilon", ylabel="error")
    axes[1].legend()

    axes[2].loglog(epsilon_values, times, "o-", color="tab:blue", label="time")
    axes[2].set_xlabel("epsilon")
    axes[2].set_ylabel("seconds", color="tab:blue")
    axes[2].tick_params(axis="y", labelcolor="tab:blue")
    axes[2].invert_xaxis()
    evaluation_axis = axes[2].twinx()
    evaluation_axis.loglog(
        epsilon_values, evaluations, "s--", color="tab:orange", label="directions"
    )
    evaluation_axis.set_ylabel("direction evaluations", color="tab:orange")
    evaluation_axis.tick_params(axis="y", labelcolor="tab:orange")
    axes[2].set_title("Computational cost")
    lines = axes[2].lines + evaluation_axis.lines
    axes[2].legend(lines, [line.get_label() for line in lines], loc="upper left")
    figure.tight_layout()
    figure.savefig(output_dir / "random_convex_polygons_adaptive_comparison.png", dpi=180)
    plt.close(figure)

    summary: dict[str, object] = {
        "dimension": 2,
        "random_seed": random_seed,
        "number_of_random_points": number_of_random_points,
        "number_of_polygon_vertices": len(polygon),
        "number_of_skeleton_points": len(query_points),
        "theta_0_rad": theta_0,
        "algorithm_1_width": exact.width,
        "algorithm_1_time_seconds": exact_time,
        "algorithm_1_maximizer_p_k": exact_point_index + 1,
        "w_sequence": exact.w.tolist(),
        "adaptive_results": comparison_rows,
        "result_rows": maximizer_rows,
    }
    (output_dir / "random_convex_polygons_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_suite(
    number_of_random_points: int,
    skeleton_point_count: int,
    polygon_count: int,
    output_dir: Path,
    random_seed: int = 20260820,
    timing_repeats: int = 3,
) -> dict[str, object]:
    """Run the random convex polygon experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    case_summaries = []
    aggregate_rows = []
    illustration_rows = []

    for polygon_index in range(1, polygon_count + 1):
        seed = random_seed + polygon_index - 1
        case_dir = output_dir / f"polygon_{polygon_index}"
        summary = run(
            number_of_random_points,
            skeleton_point_count,
            case_dir,
            seed,
            timing_repeats,
        )
        summary["polygon_index"] = polygon_index
        case_summaries.append(summary)

        for row in summary["result_rows"]:
            aggregate_rows.append(
                {
                    "polygon": f"Q_{polygon_index}",
                    "random_seed": seed,
                    "polygon_vertices": summary["number_of_polygon_vertices"],
                    "skeleton_points": summary["number_of_skeleton_points"],
                    **row,
                }
            )

        source_names = [
            "random_convex_polygons_maximizer_exact.png",
            *(
                f"random_convex_polygons_maximizer_epsilon_1e-{power:02d}.png"
                for power in range(1, 6)
            ),
        ]
        illustration_rows.append([case_dir / source_name for source_name in source_names])

    with (output_dir / "random_convex_polygons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate_rows)

    combined_figure = output_dir / "random_convex_polygons.png"
    save_illustration_grid(illustration_rows, combined_figure)

    suite_summary: dict[str, object] = {
        "dimension": 2,
        "number_of_polygons": polygon_count,
        "number_of_random_points_per_polygon": number_of_random_points,
        "number_of_skeleton_points_per_polygon": skeleton_point_count,
        "base_random_seed": random_seed,
        "timing_repeats": timing_repeats,
        "cases": case_summaries,
        "result_rows": aggregate_rows,
    }
    (output_dir / "random_convex_polygons.json").write_text(
        json.dumps(suite_summary, indent=2) + "\n", encoding="utf-8"
    )
    return suite_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of random planar points")
    parser.add_argument("--skeleton-points", type=int, default=50)
    parser.add_argument("--polygon-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/random_convex_polygons"))
    args = parser.parse_args()
    if args.n < 3:
        parser.error("--n must be at least 3")
    if args.skeleton_points < 1:
        parser.error("--skeleton-points must be positive")
    if args.polygon_count < 1:
        parser.error("--polygon-count must be positive")
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be positive")
    suite = run_suite(
        args.n,
        args.skeleton_points,
        args.polygon_count,
        args.output_dir,
        args.seed,
        args.timing_repeats,
    )
    report = {
        "number_of_polygons": suite["number_of_polygons"],
        "number_of_skeleton_points_per_polygon": suite["number_of_skeleton_points_per_polygon"],
        "results_file": str(args.output_dir / "random_convex_polygons.csv"),
        "figure_file": str(args.output_dir / "random_convex_polygons.png"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
