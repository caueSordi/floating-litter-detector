import cv2
import numpy as np


def detect_boxes(img, v_min=180, s_max=60,
                 min_area=100, max_area_frac=0.10, max_aspect=4.0):
    """detects trash floating by segmentation HSV + borders.
    return a list of boxes with positions [x, y, w, h]."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, v_min]), np.array([180, s_max, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_area = img.shape[0] * img.shape[1]
    boxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(1, min(w, h))
        touches = x <= 0 or y <= 0 or x + w >= img.shape[1] or y + h >= img.shape[0]
        if area < min_area or area > max_area_frac * img_area:
            continue
        if aspect > max_aspect or touches:
            continue
        boxes.append([x, y, w, h])

    return _merge_contained(boxes)


def _merge_contained(boxes):
    keep = []
    for b in boxes:
        inside = any(
            b is not o and b[0] >= o[0] and b[1] >= o[1]
            and b[0] + b[2] <= o[0] + o[2] and b[1] + b[3] <= o[1] + o[3]
            for o in boxes
        )
        if not inside:
            keep.append(b)
    return keep
