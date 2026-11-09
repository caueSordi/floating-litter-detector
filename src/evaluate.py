from pathlib import Path


def load_yolo_boxes(path, w, h):
    """Lê um .txt YOLO e devolve caixas [x, y, w, h] em pixels."""
    path = Path(path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        _, xc, yc, bw, bh = map(float, line.split())
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
    return boxes


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0


def match(det, gt, iou_thresh=0.5):
    """verify matches compared to the ground truth. and return (tp, fp, fn), not enough ground truth cases to point accuracy yet"""
    matched = set()
    tp = 0
    for d in det:
        best, best_iou = -1, 0.0
        for i, g in enumerate(gt):
            v = iou(d, g)
            if v > best_iou:
                best, best_iou = i, v
        if best_iou >= iou_thresh and best not in matched:
            matched.add(best)
            tp += 1
    return tp, len(det) - tp, len(gt) - tp
