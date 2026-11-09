import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from src.detect import detect_boxes
from src.evaluate import load_yolo_boxes, match
from src.visualize import draw_check

IMG_DIR = Path("data/raw/images")
LBL_DIR = Path("data/raw/labels")
OUT_DIR = Path("results/batch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 1280

total_tp = total_fp = total_fn = 0
images = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
for img_path in images:
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (WIDTH, int(img.shape[0] * WIDTH / img.shape[1])))
    h, w = img.shape[:2]

    det = detect_boxes(img)
    gt = load_yolo_boxes(LBL_DIR / (img_path.stem + ".txt"), w, h)
    tp, fp, fn = match(det, gt)
    total_tp += tp; total_fp += fp; total_fn += fn

    cv2.imwrite(str(OUT_DIR / (img_path.stem + "_check.jpg")), draw_check(img, gt, det))
    print(f"{img_path.stem}: GT={len(gt)} det={len(det)} TP={tp} FP={fp} FN={fn}")

prec = total_tp / max(1, total_tp + total_fp)
rec = total_tp / max(1, total_tp + total_fn)
print(f"\nTOTAL: TP={total_tp} FP={total_fp} FN={total_fn} | precisão={prec:.2f} recall={rec:.2f}")
