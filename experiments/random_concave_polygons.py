# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Random concave polygons.

All polygons are generated reproducibly by the same random construction.  The
adaptive method is used only as an exploratory heuristic because its error bound
requires convexity; its returned values are compared a posteriori with the exact
values from Algorithm 1.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from random_convex_polygons import (
    directional_chord,
    exact_chord,
    exact_maximizer_index,
    median_runtime,
    polygon_boundary,
    save_illustration_grid,
    save_maximizer_figure,
)

from width_function.algorithms import (
    algorithm_1,
    heuristic_adaptive_width_2d,
)
from width_function.geometry import sample_polygon_skeleton

EPSILONS = tuple(10.0**-power for power in range(1, 6))


@dataclass(frozen=True)
class CaseSpec:
    """One polygon, its query set, and the initial angles to refine from."""

    name: str
    polygon: np.ndarray
    query_points: np.ndarray
    initial_angles: dict[str, float]
    random_seed: int
    description: str


def random_concave_polygon(generator: np.random.Generator, vertex_count: int) -> np.ndarray:
    """Generate a reproducible simple star-shaped polygon with reflex vertices."""
    if vertex_count < 6:
        raise ValueError("a random concave polygon requires at least six vertices")
    for _ in range(100):
        rotation = generator.uniform(0.0, 2.0 * np.pi)
        jitter = generator.uniform(-0.24, 0.24, size=vertex_count)
        angles = rotation + 2.0 * np.pi * (np.arange(vertex_count) + jitter) / vertex_count
        radii = generator.uniform(0.78, 1.0, size=vertex_count)
        dent_count = max(3, vertex_count // 4)
        dent_indices = generator.choice(vertex_count, size=dent_count, replace=False)
        radii[dent_indices] = generator.uniform(0.28, 0.48, size=dent_count)
        polygon = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        polygon *= generator.uniform(0.82, 1.08, size=2)
        if _is_simple_polygon(polygon) and _is_concave_polygon(polygon):
            return polygon
    raise RuntimeError("failed to generate a simple concave polygon")


def _is_concave_polygon(polygon: np.ndarray, tolerance: float = 1e-10) -> bool:
    first = np.roll(polygon, -1, axis=0) - polygon
    second = np.roll(polygon, -2, axis=0) - np.roll(polygon, -1, axis=0)
    crosses = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    return bool(np.any(crosses > tolerance) and np.any(crosses < -tolerance))


def _orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    left = second - first
    right = third - first
    return float(left[0] * right[1] - left[1] * right[0])


def _segments_cross(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float = 1e-10,
) -> bool:
    values = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    return values[0] * values[1] < -tolerance and values[2] * values[3] < -tolerance


def _is_simple_polygon(polygon: np.ndarray) -> bool:
    count = len(polygon)
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first in (second, second_next) or first_next in (second, second_next):
                continue
            if _segments_cross(
                polygon[first],
                polygon[first_next],
                polygon[second],
                polygon[second_next],
            ):
                return False
    return True


def _write_points(path: Path, points: np.ndarray, prefix: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([f"{prefix}_index", "x", "y"])
        for index, point in enumerate(points, start=1):
            writer.writerow([index, *point])


def build_case(
    polygon_index: int,
    vertex_count: int,
    skeleton_point_count: int,
    random_seed: int,
) -> tuple[CaseSpec, np.ndarray]:
    """Assemble one polygon, its skeleton sample, and its initial angles."""
    generator = np.random.default_rng(random_seed)
    polygon = random_concave_polygon(generator, vertex_count)
    angles = {"seeded": float(generator.uniform(0.0, np.pi))}

    skeleton = sample_polygon_skeleton(
        polygon,
        boundary_samples_per_edge=64,
        max_points=skeleton_point_count,
        min_clearance_fraction=0.02,
    )
    return CaseSpec(
        name=f"Q_{polygon_index}",
        polygon=polygon,
        query_points=skeleton.points,
        initial_angles=angles,
        random_seed=random_seed,
        description="random simple concave polygon",
    ), skeleton.segments


def run_case(
    spec: CaseSpec,
    skeleton_segments: np.ndarray,
    output_dir: Path,
    timing_repeats: int,
) -> dict[str, object]:
    """Run the exact algorithm and the exploratory refinement on one polygon."""
    output_dir.mkdir(parents=True, exist_ok=True)
    polygon = spec.polygon
    query_points = spec.query_points
    boundary = polygon_boundary(polygon)

    _write_points(output_dir / "random_concave_polygons_concave_polygon_Q.csv", polygon, "vertex")
    _write_points(output_dir / "random_concave_polygons_skeleton_G.csv", query_points, "p")
    with (output_dir / "random_concave_polygons_skeleton_segments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["segment", "x_1", "y_1", "x_2", "y_2"])
        for index, (start, end) in enumerate(skeleton_segments, start=1):
            writer.writerow([index, *start, *end])

    exact = algorithm_1(boundary, query_points)
    exact_time = median_runtime(lambda: algorithm_1(boundary, query_points), timing_repeats)
    exact_point_index = exact_maximizer_index(
        boundary, query_points, exact.width, exact.point_widths
    )
    exact_start, exact_end, exact_chord_width = exact_chord(
        boundary, query_points[exact_point_index]
    )
    if not np.isclose(exact_chord_width, exact.width, rtol=1e-7, atol=1e-8):
        raise AssertionError("the exact visual witness does not match Algorithm 1")

    result_rows = [
        {
            "theta_0_label": "",
            "theta_0_rad": "",
            "method": "Algorithm 1 (exact)",
            "epsilon": "",
            "p_k": exact_point_index + 1,
            "width": exact.width,
            "actual_error": 0.0,
            "surrogate_gap": "",
            "within_epsilon": True,
            "converged": True,
            "termination_reason": "exact",
            "time_seconds": exact_time,
            "speedup_vs_algorithm_1": 1.0,
            "membership_tests": exact.membership_tests,
            "ell_evaluations": exact.ell_evaluations,
            "direction_evaluations": "",
        }
    ]
    save_maximizer_figure(
        output_dir / "random_concave_polygons_maximizer_exact.png",
        polygon,
        skeleton_segments,
        query_points,
        exact_point_index,
        exact_start,
        exact_end,
    )

    for label, theta_0 in spec.initial_angles.items():
        adaptive_results = []
        adaptive_times = []
        for epsilon in EPSILONS:

            def adaptive_call(epsilon: float = epsilon, angle: float = theta_0) -> object:
                return heuristic_adaptive_width_2d(polygon, query_points, epsilon, theta_0=angle)

            adaptive_call()
            timed = []
            for _ in range(timing_repeats):
                start_time = perf_counter()
                result = adaptive_call()
                timed.append((perf_counter() - start_time, result))
            elapsed, result = sorted(timed, key=lambda item: item[0])[len(timed) // 2]
            adaptive_times.append(elapsed)
            adaptive_results.append(result)

        visuals = []
        for result in adaptive_results:
            point_index = int(np.argmax(result.point_upper_bounds))
            direction = result.point_best_directions[point_index]
            start, end, chord_width = directional_chord(
                polygon, query_points[point_index], direction
            )
            if not np.isclose(chord_width, result.width, rtol=1e-8, atol=1e-9):
                raise AssertionError("the adaptive witness does not match its upper estimate")
            visuals.append(
                {
                    "point_index": point_index,
                    "level": int(result.levels[point_index]),
                    "direction_angle_rad": float(
                        np.mod(np.arctan2(direction[1], direction[0]), np.pi)
                    ),
                    "start": start,
                    "end": end,
                    "width": chord_width,
                }
            )

        for result, elapsed, visual in zip(adaptive_results, adaptive_times, visuals, strict=True):
            actual_error = result.width - exact.width
            if actual_error < -1e-8:
                raise AssertionError("a sampled-direction estimate fell below the exact width")
            result_rows.append(
                {
                    "theta_0_label": label,
                    "theta_0_rad": theta_0,
                    "method": "Adaptive heuristic",
                    "epsilon": result.epsilon,
                    "p_k": int(visual["point_index"]) + 1,
                    "width": result.width,
                    "actual_error": actual_error,
                    "surrogate_gap": result.certified_gap,
                    "within_epsilon": bool(actual_error <= result.epsilon + 1e-9),
                    "converged": result.converged,
                    "termination_reason": result.termination_reason,
                    "time_seconds": elapsed,
                    "speedup_vs_algorithm_1": exact_time / elapsed,
                    "membership_tests": "",
                    "ell_evaluations": "",
                    "direction_evaluations": result.direction_evaluations,
                }
            )
            exponent = round(-np.log10(result.epsilon))
            save_maximizer_figure(
                output_dir / f"random_concave_polygons_maximizer_{label}_1e-{exponent:02d}.png",
                polygon,
                skeleton_segments,
                query_points,
                int(visual["point_index"]),
                visual["start"],
                visual["end"],
            )

    with (output_dir / "random_concave_polygons_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_rows)

    with (output_dir / "random_concave_polygons_algorithm_1_sequence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["k", "w_k"])
        writer.writerows(enumerate(exact.w))

    summary: dict[str, object] = {
        "dimension": 2,
        "polygon_type": spec.description,
        "random_seed": spec.random_seed,
        "number_of_polygon_vertices": len(polygon),
        "number_of_skeleton_points": len(query_points),
        "prescribed_query_points": 0,
        "initial_angles_rad": spec.initial_angles,
        "algorithm_1_width": exact.width,
        "algorithm_1_time_seconds": exact_time,
        "algorithm_1_maximizer_p_k": exact_point_index + 1,
        "algorithm_1_ell_evaluations": exact.ell_evaluations,
        "w_sequence": exact.w.tolist(),
        "result_rows": result_rows,
    }
    (output_dir / "random_concave_polygons_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return summary


def run_suite(
    vertex_count: int,
    skeleton_point_count: int,
    polygon_count: int,
    output_dir: Path,
    random_seed: int,
    timing_repeats: int,
) -> dict[str, object]:
    """Run the random concave polygon experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    case_summaries = []
    aggregate_rows = []
    illustration_rows = []

    for polygon_index in range(1, polygon_count + 1):
        seed = random_seed + polygon_index - 1
        case_dir = output_dir / f"polygon_{polygon_index}"
        spec, segments = build_case(polygon_index, vertex_count, skeleton_point_count, seed)
        summary = run_case(spec, segments, case_dir, timing_repeats)
        summary["polygon_index"] = polygon_index
        case_summaries.append(summary)
        for row in summary["result_rows"]:
            aggregate_rows.append(
                {
                    "polygon": spec.name,
                    "random_seed": seed,
                    "polygon_vertices": summary["number_of_polygon_vertices"],
                    "skeleton_points": summary["number_of_skeleton_points"],
                    **row,
                }
            )

        illustrated = next(iter(spec.initial_angles))
        source_names = [
            "random_concave_polygons_maximizer_exact.png",
            *(
                f"random_concave_polygons_maximizer_{illustrated}_1e-{power:02d}.png"
                for power in range(1, 6)
            ),
        ]
        illustration_rows.append([case_dir / source_name for source_name in source_names])

    with (output_dir / "random_concave_polygons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate_rows)

    combined_figure = output_dir / "random_concave_polygons.png"
    save_illustration_grid(illustration_rows, combined_figure)

    suite_summary: dict[str, object] = {
        "dimension": 2,
        "polygon_type": "random simple concave polygons",
        "number_of_polygons": polygon_count,
        "number_of_vertices_per_random_polygon": vertex_count,
        "number_of_skeleton_points_per_polygon": skeleton_point_count,
        "base_random_seed": random_seed,
        "timing_repeats": timing_repeats,
        "cases": case_summaries,
        "result_rows": aggregate_rows,
    }
    (output_dir / "random_concave_polygons.json").write_text(
        json.dumps(suite_summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return suite_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices", type=int, default=12)
    parser.add_argument("--skeleton-points", type=int, default=50)
    parser.add_argument("--polygon-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/random_concave_polygons"))
    args = parser.parse_args()
    if args.vertices < 6:
        parser.error("--vertices must be at least 6")
    if args.skeleton_points < 2:
        parser.error("--skeleton-points must be at least two")
    if args.polygon_count < 1:
        parser.error("--polygon-count must be positive")
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be positive")

    suite = run_suite(
        args.vertices,
        args.skeleton_points,
        args.polygon_count,
        args.output_dir,
        args.seed,
        args.timing_repeats,
    )
    print(
        json.dumps(
            {
                "number_of_polygons": suite["number_of_polygons"],
                "polygon_type": suite["polygon_type"],
                "results_file": str(args.output_dir / "random_concave_polygons.csv"),
                "figure_file": str(args.output_dir / "random_concave_polygons.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
