# Real-World Applications

The two executable workflows implement the same four phases:

1. Extraction and Polygonization: a YOLO segmentation mask is converted into
   one or more polygons in the image plane.
2. Selection of Set $G$: finite points are sampled from the approximate
   topological skeleton.
3. Width Computation: polygon edges form the compact convex boundary cover and
   Algorithm 1 computes $W(Q;G)$.
4. Calibration and Overlay: the result is converted when a physical scale is
   available and the geometric measurement is drawn on the source image.

## Installation

```bash
python -m pip install -e ".[deployment]"
mkdir -p models
```

Use these checkpoint filenames:

- `models/concrete_crack_segmentation_yolov8n.pt`
- `models/lugol_negative_region_segmentation_yolov8m.pt`

Both checkpoints are included in the repository and are the default model
paths used by the application programs.

## Concrete Crack Self-Healing Assessment

The program accepts either one multi-page TIFF stack or an ordered list of
spatially registered observations. For a six-stage sequence, the default
observation days are 0, 2, 4, 7, 14, and 28.

```bash
python applications/concrete_crack_self_healing.py registered_stack.tiff
```

```bash
python applications/concrete_crack_self_healing.py \
  day_0.png day_2.png day_4.png day_7.png day_14.png day_28.png
```

The initial topological-skeleton sample $G_1$ is fixed. At stage $i$, the
program retains the points of $G_1$ lying strictly inside $Q_i$ and computes
$W(Q_i;G_i)$. The default 6400-dpi calibration is 3.96875 micrometres per
pixel. Use `--micrometres-per-pixel` for another calibrated acquisition.

The output directory contains a CSV table, a JSON summary, and one overlay per
observation. Each overlay shows retained polygon boundaries, surviving points
from the initial set, and the segment realizing the reported maximum width.

## Lugol-Negative Regions in Cervical Images

The program selects the largest YOLO instance, converts it into a polygon $Q$,
samples $G$ from its approximate topological skeleton, and reports
$W_{\max}=W(Q;G)$ in pixels.

```bash
python applications/lugol_negative_region_width.py cervical_image.png
```

The output directory contains a JSON measurement and an overlay showing the
segmented region, approximate skeleton, sampled points, and segment realizing
$W_{\max}$. Pixel lengths should be converted to physical units only when an
image-specific calibration is available.
