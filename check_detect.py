import cv2
import numpy as np

img = cv2.imread("data/raw/images/GOPR1708.jpg")
img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# branco/claro = saturação baixa + valor alto (o saco é branco, as folhas são marrons)
mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_area = img.shape[0] * img.shape[1]
for c in contours:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    aspect = max(w, h) / max(1, min(w, h))
    touches_border = x <= 0 or y <= 0 or x + w >= img.shape[1] or y + h >= img.shape[0]

    if area < 100:               # ruído pequeno
        continue
    if area > 0.10 * img_area:   # grande demais pra ser lixo
        continue
    if aspect > 4:               # largo/fino demais
        continue
    if touches_border:           # estrutura (doca, céu)
        continue

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


cv2.imwrite("results/mask.jpg", mask)
cv2.imwrite("results/detect.jpg", img)
