import cv2
import numpy as np


# =====================================================
# 1. MANUAL WATER REGION
# =====================================================

def create_polygon_mask(image_shape, polygon):
    """
    Creates a binary mask from a manually defined water polygon.

    White (255): valid water region.
    Black (0): excluded region.

    Polygon coordinates must use the dimensions of the
    image being processed.
    """

    height, width = image_shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    # np.array (not asarray) makes a copy, so the caller's
    # polygon is never modified by the clipping below.
    points = np.round(np.array(polygon, dtype=np.float64)).astype(np.int32)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Polygon must contain (x, y) coordinate pairs.")

    if len(points) < 3:
        raise ValueError("A water polygon requires at least 3 points.")

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], 255)

    return mask


# =====================================================
# 2. AUTOMATIC SEGMENTATION INITIALIZATION
# =====================================================

def create_grabcut_initial_mask(
    img,
    corner_distance_threshold=30,
    min_corner_distance=15
):
    """
    Creates initial GrabCut labels.

    GrabCut labels:
        GC_BGD     = definite background
        GC_FGD     = definite foreground
        GC_PR_BGD  = probable background
        GC_PR_FGD  = probable foreground

    The central region is initially assumed to contain water.
    Corner regions are examined to find likely external
    structures.

    Parameters:
        corner_distance_threshold:
            Corners whose LAB distance to the center is above this
            value are marked as definite background.

        min_corner_distance:
            The most different corner is only marked as background
            if its distance is also above this (lower) value. This
            avoids labeling a water-like corner as background.

    This is a heuristic and must be inspected visually.
    """

    height, width = img.shape[:2]

    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)

    # A. Probable water region

    x1 = int(width * 0.12)
    y1 = int(height * 0.12)
    x2 = int(width * 0.88)
    y2 = int(height * 0.88)

    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    # B. Definite foreground seed

    cx = width // 2
    cy = height // 2

    core_w = max(1, int(width * 0.12))
    core_h = max(1, int(height * 0.12))

    mask[
        cy - core_h:cy + core_h,
        cx - core_w:cx + core_w
    ] = cv2.GC_FGD

    # C. Examine image corners

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    patch_w = max(1, int(width * 0.08))
    patch_h = max(1, int(height * 0.08))

    center_patch = lab[
        cy - core_h:cy + core_h,
        cx - core_w:cx + core_w
    ]

    center_color = np.median(center_patch.reshape(-1, 3), axis=0)

    corners = {
        "top_left": (0, 0),
        "top_right": (width - patch_w, 0),
        "bottom_left": (0, height - patch_h),
        "bottom_right": (width - patch_w, height - patch_h)
    }

    corner_distances = {}

    for name, (x, y) in corners.items():

        patch = lab[y:y + patch_h, x:x + patch_w]

        corner_color = np.median(patch.reshape(-1, 3), axis=0)

        corner_distances[name] = np.linalg.norm(
            corner_color.astype(float) - center_color.astype(float)
        )

    # The corner most different from the center is a background
    # seed, as long as it is different enough to be meaningful.

    most_different = max(corner_distances, key=corner_distances.get)

    for name, (x, y) in corners.items():

        distance = corner_distances[name]

        is_most_different = (
            name == most_different and distance > min_corner_distance
        )

        if is_most_different or distance > corner_distance_threshold:
            mask[y:y + patch_h, x:x + patch_w] = cv2.GC_BGD

    return mask


# =====================================================
# 3. POST-PROCESSING
# =====================================================

def clean_water_mask(mask, min_component_fraction=0.005):
    """
    Cleans a binary water mask.

    Input:
        Binary mask:
            255 = water
            0 = background

    Output:
        Clean binary mask containing only the largest valid
        connected component.
    """

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)

    height, width = binary.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if count <= 1:
        return binary

    min_area = int(height * width * min_component_fraction)

    valid_components = []

    for i in range(1, count):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            valid_components.append((area, i))

    if not valid_components:
        return np.zeros_like(binary)

    largest_component = max(valid_components, key=lambda item: item[0])[1]

    result = np.zeros_like(binary)
    result[labels == largest_component] = 255

    return result


# =====================================================
# 4. MAIN WATER SEGMENTATION
# =====================================================

def _ensure_bgr(img):
    """Returns a 3-channel 8-bit BGR image (GrabCut requirement)."""

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if img.dtype != np.uint8:
        img = cv2.normalize(
            img, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    return img


def segment_water(
    img,
    polygon=None,
    iterations=5
):
    """
    Hybrid water segmentation.

    Parameters:
        img:
            Input image (BGR, grayscale or BGRA).

        polygon:
            Optional manually defined water polygon.
            When provided, automatic segmentation is skipped.

        iterations:
            Number of GrabCut iterations.

    Returns:
        water_mask:
            Binary mask where white represents water.

        debug:
            Intermediate segmentation results.
    """

    if img is None or img.size == 0:
        raise ValueError("Input image is empty.")

    img = _ensure_bgr(img)

    # ---- Manual mode ----------------------------------------

    if polygon is not None:

        water_mask = create_polygon_mask(img.shape, polygon)

        debug = {
            "mode": "manual",
            "initial_mask": None,
            "grabcut_mask": None,
            "water_mask": water_mask
        }

        print(
            f"[WATER] Manual segmentation completed: "
            f"{cv2.countNonZero(water_mask)} pixels"
        )

        return water_mask, debug

    # ---- Automatic mode -------------------------------------

    # A. Generate initial GrabCut labels

    initial_mask = create_grabcut_initial_mask(img)

    # B. Run GrabCut

    grabcut_mask = initial_mask.copy()

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        img,
        grabcut_mask,
        None,
        bgd_model,
        fgd_model,
        iterations,
        cv2.GC_INIT_WITH_MASK
    )

    # C. Convert GrabCut labels into binary mask

    raw_mask = np.where(
        np.isin(grabcut_mask, [cv2.GC_FGD, cv2.GC_PR_FGD]),
        255,
        0
    ).astype(np.uint8)

    # D. Clean segmentation

    water_mask = clean_water_mask(raw_mask)

    print(
        f"[WATER] Automatic segmentation completed: "
        f"{cv2.countNonZero(water_mask)} pixels"
    )

    debug = {
        "mode": "automatic",
        "initial_mask": initial_mask,
        "grabcut_mask": raw_mask,
        "water_mask": water_mask
    }

    return water_mask, debug


# =====================================================
# 5. DEBUG VISUALIZATION
# =====================================================

def labels_to_visible(mask):
    """
    Maps GrabCut labels (0-3) to visible gray levels.
    """

    lut = np.zeros(256, dtype=np.uint8)
    lut[cv2.GC_BGD] = 0
    lut[cv2.GC_PR_BGD] = 85
    lut[cv2.GC_PR_FGD] = 170
    lut[cv2.GC_FGD] = 255

    return lut[mask]


def create_water_debug_view(img, water_mask, debug):
    """
    Creates a 2x2 diagnostic dashboard.

    1. Original image with water overlay
    2. Initial GrabCut labels
    3. Raw GrabCut segmentation
    4. Final water mask
    """

    img = _ensure_bgr(img)

    # Green overlay indicates detected water
    overlay = img.copy()
    overlay[water_mask > 0] = (0, 180, 0)

    original = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)

    def make_panel(image, title):

        if image is None:
            image = np.zeros(water_mask.shape, dtype=np.uint8)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image = image.copy()

        target_width = 640
        target_height = int(image.shape[0] * target_width / image.shape[1])

        image = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Draw text after resizing so it has a consistent size
        cv2.putText(
            image,
            title,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return image

    initial_vis = (
        labels_to_visible(debug["initial_mask"])
        if debug["initial_mask"] is not None
        else None
    )

    panels = [
        make_panel(original, f"1. Water overlay ({debug['mode']})"),
        make_panel(initial_vis, "2. Initial labels"),
        make_panel(debug["grabcut_mask"], "3. GrabCut result"),
        make_panel(water_mask, "4. Final water mask")
    ]

    top = cv2.hconcat(panels[:2])
    bottom = cv2.hconcat(panels[2:])

    return cv2.vconcat([top, bottom])


# =====================================================
# EXAMPLE USAGE
# =====================================================

if __name__ == "__main__":

    image = cv2.imread("input.jpg")

    # Automatic mode:
    mask, dbg = segment_water(image)

    # Manual mode (example polygon, (x, y) in image coordinates):
    # polygon = [(100, 200), (500, 180), (550, 400), (80, 420)]
    # mask, dbg = segment_water(image, polygon=polygon)

    view = create_water_debug_view(image, mask, dbg)

    cv2.imwrite("water_mask.png", mask)
    cv2.imwrite("water_debug.png", view)