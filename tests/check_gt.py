import cv2

img = cv2.imread("data/raw/images/GOPR1708.jpg")
h, w = img.shape[:2]

with open("data/raw/labels/GOPR1708.txt") as f:
    for line in f:
        cls, xc, yc, bw, bh = map(float, line.split())
        x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
        x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

img = cv2.resize(img, (1280, int(h * 1280 / w)))
cv2.imwrite("results/gt_check.jpg", img)
print("salvo em results/gt_check.jpg")
