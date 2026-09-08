# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Measure a registered concrete-crack self-healing sequence with Algorithm 1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from width_function.applications import (
    draw_measurement_overlay,
    measure_polygon,
    merge_detection_polygons,
    points_strictly_inside,
    prediction_polygons,
    sample_component_skeleton,
)

OBSERVATION_DAYS = (0, 2, 4, 7, 14, 28)
MICROMETRES_PER_PIXEL_6400_DPI = 3.96875


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure W(Q_i; G_i) across registered concrete-crack images while keeping "
            "the initial skeleton set fixed."
        )
    )
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="one multi-page TIFF stack or a registered sequence of image files",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/concrete_crack_segmentation_yolov8n.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/concrete_crack"))
    parser.add_argument("--days", type=int, nargs="+")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--fallback-confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simplify-tolerance", type=float, default=1.5)
    parser.add_argument("--max-polygon-vertices", type=int, default=48)
    parser.add_argument("--minimum-component-area", type=float, default=20.0)
    parser.add_argument("--merge-close-kernel", type=int, default=3)
    parser.add_argument("--boundary-samples-per-edge", type=int, default=8)
    parser.add_argument("--skeleton-points", type=int, default=30)
    parser.add_argument("--min-clearance-fraction", type=float, default=0.002)
    parser.add_argument(
        "--micrometres-per-pixel",
        type=float,
        default=MICROMETRES_PER_PIXEL_6400_DPI,
    )
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if not 0.0 <= args.fallback_confidence <= args.confidence <= 1.0:
        parser.error("confidence values must satisfy 0 <= fallback <= confidence <= 1")
    frames, labels = _load_registered_frames(args.images)
    days = _observation_days(len(frames), args.days, parser)
    model = YOLO(str(args.model))

    initial_components = _segment_components(model, frames[0], args, args.confidence)
    initial_confidence = args.confidence
    if not initial_components:
        initial_components = _segment_components(
            model,
            frames[0],
            args,
            args.fallback_confidence,
        )
        initial_confidence = args.fallback_confidence
    if not initial_components:
        raise RuntimeError("no crack component was detected in the initial image")
    initial_points = sample_component_skeleton(
        initial_components,
        max_points=args.skeleton_points,
        boundary_samples_per_edge=args.boundary_samples_per_edge,
        min_clearance_fraction=args.min_clearance_fraction,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    initial_width = None
    for index, (frame, label, day) in enumerate(zip(frames, labels, days, strict=True)):
        confidence = initial_confidence if index == 0 else args.confidence
        components = (
            initial_components
            if index == 0
            else _segment_components(model, frame, args, confidence)
        )
        selected_measurement = None
        surviving_mask = np.zeros(len(initial_points), dtype=bool)
        for component in components:
            inside = points_strictly_inside(initial_points, component)
            surviving_mask |= inside
            if not np.any(inside):
                continue
            measurement = measure_polygon(
                component,
                query_points=initial_points[inside],
                simplify_tolerance=0.0,
                max_vertices=args.max_polygon_vertices,
            )
            if selected_measurement is None or measurement.width > selected_measurement.width:
                selected_measurement = measurement
        width_pixels = 0.0 if selected_measurement is None else selected_measurement.width
        if initial_width is None:
            initial_width = width_pixels
        relative_width = 0.0 if initial_width == 0.0 else width_pixels / initial_width
        status = "ok" if selected_measurement is not None else "no_surviving_initial_point"
        overlay_path = args.output_dir / f"stage_{index + 1:02d}_day_{day}_overlay.png"
        draw_measurement_overlay(
            frame,
            list(components),
            overlay_path,
            query_points=initial_points[surviving_mask],
            segment_start=(
                None if selected_measurement is None else selected_measurement.segment_start
            ),
            segment_end=None if selected_measurement is None else selected_measurement.segment_end,
        )
        rows.append(
            {
                "stage": index + 1,
                "day": day,
                "image": label,
                "status": status,
                "confidence": confidence,
                "polygon_components": len(components),
                "initial_skeleton_points": len(initial_points),
                "surviving_skeleton_points": int(np.count_nonzero(surviving_mask)),
                "width_pixels": width_pixels,
                "width_micrometres": width_pixels * args.micrometres_per_pixel,
                "relative_width": relative_width,
                "overlay": str(overlay_path),
            }
        )

    csv_path = args.output_dir / "concrete_crack_self_healing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "application": "Concrete crack self-healing assessment",
        "model": str(args.model),
        "fixed_initial_set": "G_i = G_1 intersect int(Q_i)",
        "micrometres_per_pixel": args.micrometres_per_pixel,
        "measurements": rows,
        "csv": str(csv_path),
    }
    json_path = args.output_dir / "concrete_crack_self_healing.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _segment_components(model: YOLO, image: np.ndarray, args, confidence: float):
    prediction = model.predict(
        source=image,
        conf=confidence,
        imgsz=args.image_size,
        device=args.device,
        verbose=False,
    )[0]
    return merge_detection_polygons(
        list(prediction_polygons(prediction)),
        image_shape=image.shape[:2],
        simplify_tolerance=args.simplify_tolerance,
        max_vertices=args.max_polygon_vertices,
        minimum_area=args.minimum_component_area,
        close_kernel=args.merge_close_kernel,
    )


def _load_registered_frames(paths: list[Path]) -> tuple[list[np.ndarray], list[str]]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"image does not exist: {path}")
    if len(paths) == 1 and paths[0].suffix.casefold() in {".tif", ".tiff"}:
        loaded, frames = cv2.imreadmulti(str(paths[0]), flags=cv2.IMREAD_COLOR)
        if loaded and frames:
            return list(frames), [f"{paths[0]}#{index + 1}" for index in range(len(frames))]
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(frame is None for frame in frames):
        raise ValueError("one or more input images could not be decoded")
    return frames, [str(path) for path in paths]


def _observation_days(count: int, supplied: list[int] | None, parser) -> tuple[int, ...]:
    if supplied is not None:
        if len(supplied) != count:
            parser.error("--days must provide one value per observation")
        return tuple(supplied)
    if count == len(OBSERVATION_DAYS):
        return OBSERVATION_DAYS
    return tuple(range(count))


if __name__ == "__main__":
    main()
