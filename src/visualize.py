import cv2
"""" here the boxes are drawn on the image """

def draw_check(img, gt, det):
    vis = img.copy()
    for x, y, w, h in gt:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for x, y, w, h in det:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return vis
