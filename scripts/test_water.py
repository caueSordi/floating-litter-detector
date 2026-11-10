
import sys
import numpy as np
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import cv2
import numpy as np

from src.water import (
    segment_water,
    create_water_debug_view
)


# ==========================================================
# CONFIGURATION
# ==========================================================

IMG_DIR = Path("data/raw/images")

OUT_DIR = Path("results/water_segmentation")

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

WIDTH = 1280

# Set to None to process all images.
# For initial testing, use one image.
TEST_IMAGE = "GOPR1708"

# Enable or disable interactive polygon selection.
MANUAL_SELECTION = True


# ==========================================================
# MANUAL WATER SELECTION
# ==========================================================

def select_water_polygon(img):
    """
    Interactive water-region selection.

    Controls:
        Left click: add vertex
        ENTER: confirm polygon
        R: reset polygon
        ESC: cancel and use automatic segmentation
    """

    points = []
    window_name = "Select Water Region"

    def mouse_callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point added: ({x}, {y})")

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print("\n=== WATER SELECTION ===")
    print("Left click: Add point")
    print("ENTER: Confirm polygon")
    print("R: Reset")
    print("ESC: Automatic segmentation")

    while True:

        display = img.copy()

        # Draw vertices
        for point in points:

            cv2.circle(
                display,
                point,
                5,
                (0, 255, 0),
                -1
            )

        # Draw polygon lines
        if len(points) > 1:

            cv2.polylines(
                display,
                [np.array(points, dtype=np.int32)],
                False,
                (0, 255, 0),
                2
            )

        # Polygon preview
        if len(points) >= 3:

            cv2.polylines(
                display,
                [np.array(points, dtype=np.int32)],
                True,
                (0, 255, 255),
                2
            )

        cv2.putText(
            display,
            f"Selected points: {len(points)}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "CLICK: Add | ENTER: Confirm | R: Reset | ESC: Auto",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # ENTER (Linux and Windows)
        if key in (10, 13):

            if len(points) >= 3:

                print("\nPolygon confirmed!")
                print(f"Total vertices: {len(points)}")

                cv2.destroyWindow(window_name)
                cv2.waitKey(1)

                return points

            print("You need at least 3 points.")

        # Reset
        elif key == ord("r"):

            points.clear()
            print("Polygon reset.")

        # Automatic mode
        elif key == 27:

            print("Automatic segmentation selected.")

            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

            return None


# ==========================================================
# IMAGE DISCOVERY
# ==========================================================

images = sorted(
    p for p in IMG_DIR.iterdir()
    if p.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png"
    }
)

if TEST_IMAGE is not None:

    images = [
        p for p in images
        if p.stem == TEST_IMAGE
    ]

    if not images:

        raise FileNotFoundError(
            f"Image {TEST_IMAGE} was not found in {IMG_DIR}"
        )


# ==========================================================
# PROCESSING
# ==========================================================

for img_path in images:

    print(f"\nProcessing: {img_path.name}")

    img = cv2.imread(str(img_path))

    if img is None:

        print(f"Could not load: {img_path}")

        continue

    # ------------------------------------------------------
    # Resize image
    # ------------------------------------------------------

    img = cv2.resize(
        img,
        (
            WIDTH,
            int(img.shape[0] * WIDTH / img.shape[1])
        )
    )

    # ------------------------------------------------------
    # Manual polygon selection
    # ------------------------------------------------------

    if MANUAL_SELECTION:

        polygon = select_water_polygon(img)

    else:

        polygon = None

    # ------------------------------------------------------
    # Water segmentation
    # ------------------------------------------------------

    water_mask, debug = segment_water(
        img,
        polygon=polygon,
        iterations=5
    )

    # ------------------------------------------------------
    # Diagnostic dashboard
    # ------------------------------------------------------

    debug_view = create_water_debug_view(
        img,
        water_mask,
        debug
    )

    # ------------------------------------------------------
    # Water percentage
    # ------------------------------------------------------

    water_percentage = (
        cv2.countNonZero(water_mask) /
        water_mask.size
    ) * 100

    # ------------------------------------------------------
    # Save final binary mask
    # ------------------------------------------------------

    mask_path = (
        OUT_DIR /
        f"{img_path.stem}_water_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        water_mask
    )

    # ------------------------------------------------------
    # Save diagnostic dashboard
    # ------------------------------------------------------

    debug_path = (
        OUT_DIR /
        f"{img_path.stem}_water_debug.jpg"
    )

    cv2.imwrite(
        str(debug_path),
        debug_view
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print(
        f"Image: {img_path.stem}\n"
        f"Mode: {debug['mode']}\n"
        f"Water coverage: {water_percentage:.2f}%\n"
        f"Mask saved: {mask_path}\n"
        f"Debug saved: {debug_path}"
    )


print("\nWater segmentation finished.")
