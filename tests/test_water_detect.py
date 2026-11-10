import numpy as np
import cv2

from src.water import segment_water
from src.detect import detect_boxes


def _synthetic():
    img = np.full((480, 640, 3), (60, 90, 40), np.uint8)      # greenish water
    img[400:, :] = (200, 210, 220)                            # bright dock (bottom)
    cv2.rectangle(img, (300, 250), (360, 300), (240, 240, 240), -1)  # white litter
    return img


def test_manual_polygon_excludes_dock_and_finds_litter():
    img = _synthetic()
    polygon = [(0, 0), (639, 0), (639, 395), (0, 395)]
    mask, dbg = segment_water(img, polygon=polygon)
    assert dbg["mode"] == "manual"
    assert mask[450, 100] == 0 and mask[100, 100] == 255

    boxes = detect_boxes(img, water_mask=mask)
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert abs(x - 300) < 10 and abs(y - 250) < 10


def test_automatic_mode_runs():
    img = _synthetic()
    mask, dbg = segment_water(img)
    assert dbg["mode"] == "automatic"
    assert mask.shape == img.shape[:2]


def test_debug_view_and_auto_water():
    from src.detect import create_debug_view

    img = _synthetic()
    boxes, dbg = detect_boxes(img, auto_water=True, return_debug=True)
    view = create_debug_view(img, boxes, dbg)
    assert view.shape == (960, 1920, 3)


def test_color_method_still_available():
    img = _synthetic()
    polygon = [(0, 0), (639, 0), (639, 395), (0, 395)]
    mask, _ = segment_water(img, polygon=polygon)
    boxes = detect_boxes(img, water_mask=mask, method="color")
    assert len(boxes) >= 1


def test_anomaly_ignores_water_color_and_finds_litter():
    # saturated blue water (the old color rules would flag all of it)
    img = np.full((480, 640, 3), (180, 90, 30), np.uint8)
    cv2.rectangle(img, (300, 250), (360, 300), (235, 235, 235), -1)
    polygon = [(0, 0), (639, 0), (639, 479), (0, 479)]
    mask, _ = segment_water(img, polygon=polygon)
    boxes = detect_boxes(img, water_mask=mask, method="anomaly")
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert abs(x - 300) < 10 and abs(y - 250) < 10
