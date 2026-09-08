# ==============================================================================
# Copyright (c) Institute of Mathematical and Computational Sciences (IMACS),
# Ho Chi Minh City University of Technology (HCMUT),
# 268 Ly Thuong Kiet, Ward 14, District 10, Ho Chi Minh City, Vietnam.
# All rights reserved.
#
# For usage, inquiries, or feedback, please contact:
# - Prof. Phan Thanh An: thanhan@hcmut.edu.vn
# ==============================================================================

"""Measure a YOLO-segmented Lugol-negative region with Algorithm 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from width_function.applications import (
    draw_measurement_overlay,
    measure_polygon,
    polygon_area,
    prediction_polygons,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure W_max for the largest Lugol-negative region in a cervical image."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/lugol_negative_region_segmentation_yolov8m.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/lugol_negative_region"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--device", default=None)
    parser.add_argument("--simplify-tolerance", type=float, default=2.0)
    parser.add_argument("--max-polygon-vertices", type=int, default=48)
    parser.add_argument("--boundary-samples-per-edge", type=int, default=8)
    parser.add_argument("--skeleton-points", type=int, default=30)
    parser.add_argument("--min-clearance-fraction", type=float, default=0.005)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"could not read image: {args.image}")
    prediction = YOLO(str(args.model)).predict(
        source=image,
        conf=args.confidence,
        imgsz=args.image_size,
        device=args.device,
        verbose=False,
    )[0]
    polygons = prediction_polygons(prediction)
    if not polygons:
        raise RuntimeError("no Lugol-negative region was detected")
    polygon = max(polygons, key=polygon_area)
    measurement = measure_polygon(
        polygon,
        simplify_tolerance=args.simplify_tolerance,
        max_vertices=args.max_polygon_vertices,
        boundary_samples_per_edge=args.boundary_samples_per_edge,
        max_skeleton_points=args.skeleton_points,
        min_clearance_fraction=args.min_clearance_fraction,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / f"{args.image.stem}_width_overlay.png"
    draw_measurement_overlay(
        image,
        [measurement.polygon],
        overlay_path,
        query_points=measurement.query_points,
        skeleton_segments=measurement.skeleton_segments,
        segment_start=measurement.segment_start,
        segment_end=measurement.segment_end,
    )
    result = {
        "application": "Lugol-negative region width measurement",
        "image": str(args.image),
        "model": str(args.model),
        "detections": len(polygons),
        "selected_polygon_vertices": len(measurement.polygon),
        "skeleton_points": len(measurement.query_points),
        "W_max_pixels": measurement.width,
        "maximizing_point": measurement.maximizing_point.tolist(),
        "segment_start": measurement.segment_start.tolist(),
        "segment_end": measurement.segment_end.tolist(),
        "overlay": str(overlay_path),
    }
    result_path = args.output_dir / f"{args.image.stem}_width.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
