
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import cv2

from src.detect import (
    detect_boxes,
    create_debug_view
)

from src.evaluate import load_yolo_boxes, match
from src.visualize import draw_check


IMG_DIR = Path("data/raw/images")
LBL_DIR = Path("data/raw/labels")
OUT_DIR = Path("results/batch")

OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 1280


# Optional water polygons for individual images.
#
# Coordinates refer to the resized 1280 px image.
# Leave an image out of this dictionary to process
# the entire frame.
#
# Example:
#
# WATER_POLYGONS = {
#     "GOPR1767": [
#         (250, 0),
#         (1280, 0),
#         (1280, 750),
#         (250, 750)
#     ]
# }

WATER_POLYGONS = {}


total_tp = 0
total_fp = 0
total_fn = 0

images = sorted(
    p for p in IMG_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

for img_path in images:

    img = cv2.imread(str(img_path))

    if img is None:
        print(f"Could not load: {img_path}")
        continue

    # Resize image
    img = cv2.resize(
        img,
        (
            WIDTH,
            int(img.shape[0] * WIDTH / img.shape[1])
        )
    )

    h, w = img.shape[:2]

    # Get optional water polygon
    water_polygon = WATER_POLYGONS.get(
        img_path.stem
    )

    # Run detector with diagnostic information
    det, debug = detect_boxes(
        img,
        water_polygon=water_polygon,
        return_debug=True
    )

    # Load ground truth
    gt = load_yolo_boxes(
        LBL_DIR / (img_path.stem + ".txt"),
        w,
        h
    )

    # Evaluate
    tp, fp, fn = match(det, gt)

    total_tp += tp
    total_fp += fp
    total_fn += fn

    # Save original evaluation visualization
    check = draw_check(img, gt, det)

    cv2.imwrite(
        str(OUT_DIR / f"{img_path.stem}_check.jpg"),
        check
    )

    # Save processing stages and shape analysis
    debug_view = create_debug_view(
        img,
        det,
        debug
    )

    cv2.imwrite(
        str(OUT_DIR / f"{img_path.stem}_debug.jpg"),
        debug_view
    )

    # Save individual masks
    for mask_name in [
        "white_mask",
        "color_mask",
        "valid_mask",
        "candidate_mask",
        "grouped_mask"
    ]:

        cv2.imwrite(
            str(
                OUT_DIR /
                f"{img_path.stem}_{mask_name}.jpg"
            ),
            debug[mask_name]
        )

    print(
        f"{img_path.stem}: "
        f"GT={len(gt)} "
        f"det={len(det)} "
        f"TP={tp} "
        f"FP={fp} "
        f"FN={fn}"
    )


precision = total_tp / max(
    1,
    total_tp + total_fp
)

recall = total_tp / max(
    1,
    total_tp + total_fn
)

print("\n========== RESULTS ==========")

print(f"TP: {total_tp}")
print(f"FP: {total_fp}")
print(f"FN: {total_fn}")

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

print("=============================")
