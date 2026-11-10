
import cv2
import numpy as np


# ==========================================================
# 1. COLOR SEGMENTATION
# ==========================================================

def create_candidate_mask(
    img,
    v_min=180,
    s_max=70,
    color_s_min=90,
    color_v_min=70
):
    """
    Combines multiple HSV masks to detect potential litter.

    Categories:
        - Bright, low-saturation objects
        - Blue and cyan objects
        - Yellow and orange objects
        - Red objects
        - Purple and magenta objects

    Returns:
        Binary mask containing all candidate pixels.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # A. Bright neutral objects
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, v_min]),
        np.array([179, s_max, 255])
    )

    # B. Saturated artificial-looking colors
    color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    color_ranges = [
        # Red
        ((0, color_s_min, color_v_min),
         (10, 255, 255)),

        ((170, color_s_min, color_v_min),
         (179, 255, 255)),

        # Orange and yellow
        ((10, color_s_min, color_v_min),
         (38, 255, 255)),

        # Cyan and blue
        ((80, color_s_min, color_v_min),
         (135, 255, 255)),

        # Purple and magenta
        ((135, color_s_min, color_v_min),
         (169, 255, 255)),
    ]

    for lower, upper in color_ranges:
        current = cv2.inRange(
            hsv,
            np.array(lower),
            np.array(upper)
        )

        color_mask = cv2.bitwise_or(
            color_mask,
            current
        )

    # C. Combine all candidate colors
    combined = cv2.bitwise_or(
        white_mask,
        color_mask
    )

    return combined, white_mask, color_mask


# ==========================================================
# 2. WATER / VALID REGION MASK
# ==========================================================

def create_valid_mask(img, water_polygon=None):
    """
    Creates the region in which litter detection is allowed.

    water_polygon:
        Optional list of (x, y) points describing the
        approximate water boundary.

        If None, the entire image is considered valid.

    IMPORTANT:
        Polygon coordinates must refer to the resized image.
    """

    height, width = img.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if water_polygon is None:
        mask[:] = 255
        return mask

    points = np.array(
        water_polygon,
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [points],
        255
    )

    return mask


# ==========================================================
# 3. SHAPE ANALYSIS
# ==========================================================

def analyze_shape(contour):
    """
    Calculates geometric descriptors for a contour.

    Returns a dictionary with:
        area, aspect_ratio, extent, solidity,
        circularity and perimeter.
    """

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = max(w, h) / max(
        1,
        min(w, h)
    )

    bounding_area = max(1, w * h)

    extent = area / bounding_area

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = area / max(1, hull_area)

    circularity = (
        4 * np.pi * area / max(1, perimeter ** 2)
    )

    return {
        "area": area,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "perimeter": perimeter,
        "box": [x, y, w, h]
    }


# ==========================================================
# 4. BOUNDING BOX FILTERS
# ==========================================================

def touches_border(box, width, height):
    x, y, w, h = box

    return (
        x <= 0 or
        y <= 0 or
        x + w >= width or
        y + h >= height
    )


def is_valid_candidate(
    shape,
    img_area,
    min_area=40,
    max_area_frac=0.10,
    max_aspect=6.0,
    min_extent=0.08,
    min_solidity=0.20
):
    """
    Applies geometric filters.

    Shape descriptors are used as soft geometric
    constraints, not as a strict object classifier.
    """

    if shape["area"] < min_area:
        return False

    if shape["area"] > max_area_frac * img_area:
        return False

    if shape["aspect_ratio"] > max_aspect:
        return False

    if shape["extent"] < min_extent:
        return False

    if shape["solidity"] < min_solidity:
        return False

    return True


# ==========================================================
# 5. CONTAINED BOX MERGING
# ==========================================================

def _merge_contained(boxes):
    """
    Removes boxes fully contained in another box.
    """

    keep = []

    for i, box in enumerate(boxes):

        x, y, w, h = box

        inside = False

        for j, other in enumerate(boxes):

            if i == j:
                continue

            ox, oy, ow, oh = other

            if (
                x >= ox and
                y >= oy and
                x + w <= ox + ow and
                y + h <= oy + oh
            ):
                inside = True
                break

        if not inside:
            keep.append(box)

    return keep


# ==========================================================
# 6. MAIN DETECTOR
# ==========================================================

def detect_boxes(
    img,
    v_min=180,
    s_max=70,
    color_s_min=90,
    color_v_min=70,
    min_area=40,
    max_area_frac=0.10,
    max_aspect=6.0,
    min_extent=0.08,
    min_solidity=0.20,
    group_size=5,
    water_polygon=None,
    return_debug=False
):
    """
    Multi-color floating litter detector.

    Returns:
        boxes

    If return_debug=True:
        boxes, debug_data
    """

    height, width = img.shape[:2]
    img_area = height * width

    # --------------------------------------------------
    # A. Color segmentation
    # --------------------------------------------------

    candidate_mask, white_mask, color_mask = (
        create_candidate_mask(
            img,
            v_min=v_min,
            s_max=s_max,
            color_s_min=color_s_min,
            color_v_min=color_v_min
        )
    )

    # --------------------------------------------------
    # B. Restrict detection to valid region
    # --------------------------------------------------

    valid_mask = create_valid_mask(
        img,
        water_polygon
    )

    candidate_mask = cv2.bitwise_and(
        candidate_mask,
        valid_mask
    )

    # --------------------------------------------------
    # C. Remove isolated noise
    # --------------------------------------------------

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    clean_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    # --------------------------------------------------
    # D. Connect small fragments
    # --------------------------------------------------

    group_size = max(1, int(group_size))

    if group_size % 2 == 0:
        group_size += 1

    group_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (group_size, group_size)
    )

    grouped_mask = cv2.morphologyEx(
        clean_mask,
        cv2.MORPH_CLOSE,
        group_kernel
    )

    # Do not allow morphology to cross the valid-region
    # boundary.
    grouped_mask = cv2.bitwise_and(
        grouped_mask,
        valid_mask
    )

    # --------------------------------------------------
    # E. Extract external contours
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        grouped_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    shape_data = []

    for contour in contours:

        shape = analyze_shape(contour)

        if not is_valid_candidate(
            shape,
            img_area,
            min_area=min_area,
            max_area_frac=max_area_frac,
            max_aspect=max_aspect,
            min_extent=min_extent,
            min_solidity=min_solidity
        ):
            continue

        box = shape["box"]

        if touches_border(box, width, height):
            continue

        boxes.append(box)
        shape_data.append((contour, shape))

    boxes = _merge_contained(boxes)

    if not return_debug:
        return boxes

    debug = {
        "white_mask": white_mask,
        "color_mask": color_mask,
        "valid_mask": valid_mask,
        "candidate_mask": candidate_mask,
        "clean_mask": clean_mask,
        "grouped_mask": grouped_mask,
        "contours": contours,
        "shape_data": shape_data
    }

    return boxes, debug


# ==========================================================
# 7. DEBUG VISUALIZATION
# ==========================================================

def create_debug_view(img, boxes, debug):
    """
    Generates a 2x3 diagnostic dashboard.

    1. Final detections
    2. Valid region / water ROI
    3. Bright-object mask
    4. Saturated-color mask
    5. Combined and grouped candidate mask
    6. Contour and shape analysis
    """

    original = img.copy()
    shape_panel = img.copy()

    # Final bounding boxes
    for i, box in enumerate(boxes):

        x, y, w, h = box

        cv2.rectangle(
            original,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        cv2.putText(
            original,
            f"ID {i + 1}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    # Draw contour and shape information
    for contour, shape in debug["shape_data"]:

        x, y, w, h = shape["box"]

        cv2.drawContours(
            shape_panel,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            shape_panel,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            shape_panel,
            f"S:{shape['solidity']:.2f} "
            f"C:{shape['circularity']:.2f}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

    def panel(image, title):

        if len(image.shape) == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        image = image.copy()

        cv2.putText(
            image,
            title,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return cv2.resize(image, (640, 480))

    panels = [
        panel(original, "1. Final detections"),
        panel(debug["valid_mask"], "2. Valid region"),
        panel(debug["white_mask"], "3. Bright objects"),
        panel(debug["color_mask"], "4. Saturated colors"),
        panel(debug["grouped_mask"], "5. Grouped candidates"),
        panel(shape_panel, "6. Shape analysis")
    ]

    row1 = cv2.hconcat(panels[0:3])
    row2 = cv2.hconcat(panels[3:6])

    return cv2.vconcat([row1, row2])
