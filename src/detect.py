
import cv2
import numpy as np

from src.water import segment_water, create_polygon_mask


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

def fill_small_holes(water_mask, max_hole_fraction=0.05):
    """
    Returns (region, holes).

    region: water mask with small internal holes filled.
    holes:  the internal holes. Litter often appears here, because
            segmentation labels it "not water" and it ends up surrounded
            by water. Holes larger than max_hole_fraction of the image
            are ignored (probably land/structures).
    """

    h, w = water_mask.shape

    contours, hierarchy = cv2.findContours(
        water_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    region = water_mask.copy()
    holes = np.zeros_like(water_mask)

    if hierarchy is None:
        return region, holes

    for c, hier in zip(contours, hierarchy[0]):

        is_hole = hier[3] != -1  # has a parent -> inner contour

        if is_hole and cv2.contourArea(c) <= max_hole_fraction * h * w:
            cv2.drawContours(region, [c], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(holes, [c], -1, 255, thickness=cv2.FILLED)

    return region, holes


def create_valid_mask(img, water_polygon=None, water_mask=None):
    """
    Creates the region in which litter detection is allowed.

    Priority:
        1. water_mask    (from segment_water): small holes are filled so
                         litter floating inside the water stays valid.
        2. water_polygon (manual, coordinates of the resized image).
        3. None: the entire image is valid.

    Returns:
        valid_mask, holes
        (holes is all zeros unless water_mask is used)
    """

    height, width = img.shape[:2]

    holes = np.zeros((height, width), dtype=np.uint8)

    if water_mask is not None:
        valid, holes = fill_small_holes(water_mask)
        return valid, holes

    mask = np.zeros((height, width), dtype=np.uint8)

    if water_polygon is None:
        mask[:] = 255
        return mask, holes

    points = np.array(water_polygon, dtype=np.int32)

    cv2.fillPoly(mask, [points], 255)

    return mask, holes


# ==========================================================
# 2.5 WATER-RELATIVE ANOMALY (replaces absolute color rules)
# ==========================================================

def create_anomaly_map(
    img,
    water_mask,
    sigma_frac=0.04,
    edge_margin_frac=0.008,
    l_weight=0.7,
    l_weight_dark=0.25,
    k_mad=6.0,
    min_dist=16.0,
    shadow_frac=0.5,
    n_iter=2
):
    """
    Finds pixels whose color differs from the surrounding WATER.

    Absolute rules ("saturated" or "bright") fail because the water
    itself is saturated (blue/green/brown) and changes with the sky
    reflection. Here the expected water color is estimated locally
    from the water pixels only, and litter is whatever deviates from it.

    Steps:
        1. LAB image; reliable background = water, away from its edge
           (dock shadows / ripples) and away from segmentation holes.
        2. Local background = normalized Gaussian blur over those pixels.
        3. dist = weighted LAB distance to that background.
        4. Threshold = median + k_mad * MAD of dist over the water,
           never below min_dist. Pixels above it are removed from the
           background estimate and it is recomputed (n_iter times), so
           litter does not contaminate its own background.
        5. Darker-than-water pixels count less (shadows are not litter),
           and very dark bands (shade over the water) are ignored.

    Returns:
        dist_map (float32), anomaly_mask (uint8 0/255), threshold
    """

    height, width = img.shape[:2]

    # work at 1/4 resolution: the background is smooth by design
    scale = 0.25
    sw, sh = max(1, int(width * scale)), max(1, int(height * scale))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_s = cv2.resize(lab, (sw, sh), interpolation=cv2.INTER_AREA)

    water_s = cv2.resize(
        water_mask, (sw, sh), interpolation=cv2.INTER_NEAREST
    )

    margin = max(1, int(width * edge_margin_frac * scale))
    reliable = cv2.erode(
        water_s,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
        )
    ) > 0

    if reliable.sum() < 50:
        reliable = water_s > 0

    # Shadow band (dock/wall shade that segmentation labeled as water):
    # much darker than the typical water. It is not a valid background
    # model and the lit/shade transition would otherwise look like an
    # "anomalous" strip, so it is excluded from both background and
    # candidates.
    l_smooth = cv2.GaussianBlur(lab_s[..., 0], (0, 0), 2.0)
    l_med = np.median(l_smooth[reliable]) if reliable.any() else 0.0
    shadow_s = (l_smooth < shadow_frac * l_med).astype(np.uint8)
    shadow_s = cv2.dilate(shadow_s, np.ones((5, 5), np.uint8))
    reliable = reliable & (shadow_s == 0)

    if reliable.sum() < 50:
        reliable = water_s > 0
        shadow_s[:] = 0

    sigma = max(1.0, sigma_frac * width * scale)
    def lab_distance(lab_img, bg_img):
        """
        Weighted LAB distance. Pixels DARKER than the water get a much
        smaller lightness weight, so shadows (dock edges, shade) do not
        count as litter, while brighter or differently colored objects do.
        """
        diff = lab_img - bg_img
        dl = diff[..., 0]
        dl = np.where(dl >= 0, dl * l_weight, dl * l_weight_dark)
        return np.sqrt(dl ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2)

    use = reliable.copy()

    for _ in range(max(1, n_iter)):

        w = use.astype(np.float32)

        num = cv2.GaussianBlur(lab_s * w[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(w, (0, 0), sigma)

        bg = num / np.maximum(den[..., None], 1e-4)

        # where no reliable water is nearby, fall back to global median
        far = den < 1e-3
        if far.any() and use.any():
            bg[far] = np.median(lab_s[use], axis=0)

        d_s = lab_distance(lab_s, bg)

        vals = d_s[reliable]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med)) * 1.4826

        thr = max(min_dist, med + k_mad * mad)

        # exclude anomalous pixels (and a halo) from the background
        bad = cv2.dilate((d_s > thr).astype(np.uint8), np.ones((3, 3), np.uint8))
        use = reliable & (bad == 0)

        if use.sum() < 50:
            use = reliable

    # full-resolution distance against the (smooth) upsampled background
    bg_full = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LINEAR)
    dist = lab_distance(lab, bg_full)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.5)

    shadow_full = cv2.resize(
        shadow_s, (width, height), interpolation=cv2.INTER_NEAREST
    )
    dist[shadow_full > 0] = 0

    anomaly = np.where(dist > thr, 255, 0).astype(np.uint8)

    return dist.astype(np.float32), anomaly, float(thr)


# ==========================================================
# 2.6 SPLITTING OF TOUCHING OBJECTS
# ==========================================================

def split_touching_objects(
    mask,
    img,
    min_area=400,
    solidity_max=0.85,
    peak_frac=0.35
):
    """
    Splits blobs made of several touching objects (e.g. a ball next to a
    bag) with a distance-transform watershed.

    Only blobs that are large and clearly non-convex (low solidity) are
    considered, so single compact objects are never cut. The cut lines
    are removed from the mask, so findContours returns separate blobs.
    """

    out = mask.copy()

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)

        cnts, _ = cv2.findContours(
            comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        c = max(cnts, key=cv2.contourArea)

        solidity = cv2.contourArea(c) / max(
            1.0, cv2.contourArea(cv2.convexHull(c))
        )

        if solidity > solidity_max:
            continue

        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)

        peaks = (dt > peak_frac * dt.max()).astype(np.uint8)

        n_peaks, markers = cv2.connectedComponents(peaks)

        if n_peaks <= 2:  # label 0 is background -> fewer than 2 peaks
            continue

        markers = markers.astype(np.int32)
        markers[comp == 0] = n_peaks  # outside-blob marker

        crop = np.ascontiguousarray(img[y:y + h, x:x + w])

        cv2.watershed(crop, markers)

        lines = ((markers == -1) & (comp > 0)).astype(np.uint8)
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))

        region = out[y:y + h, x:x + w]
        region[lines > 0] = 0

    return out


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

    # Aspect ratio of the minimum-area ROTATED rectangle: a thin slanted
    # strip (shadow edge, stick) has a huge value even when its
    # axis-aligned box looks almost square.
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    rot_aspect = max(rw, rh) / max(1.0, min(rw, rh))

    return {
        "area": area,
        "aspect_ratio": aspect_ratio,
        "rot_aspect": rot_aspect,
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
    min_solidity=0.20,
    max_rot_aspect=6.0
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

    if shape.get("rot_aspect", 1.0) > max_rot_aspect:
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
    min_area=None,
    max_area_frac=0.10,
    max_aspect=6.0,
    min_extent=0.08,
    min_solidity=0.20,
    max_rot_aspect=6.0,
    group_size=5,
    split_touching=True,
    water_polygon=None,
    water_mask=None,
    auto_water=False,
    method="anomaly",
    anomaly_k=6.0,
    anomaly_min_dist=16.0,
    edge_margin_frac=0.008,
    return_debug=False
):
    """
    Multi-color floating litter detector.

    Water region (where litter is allowed), by priority:
        water_mask    -> binary mask from segment_water()
        water_polygon -> manual polygon (resized-image coordinates)
        auto_water    -> True computes segment_water(img) automatically
        otherwise     -> entire image

    method:
        "anomaly" (default): litter = what differs from the local water
                 color, plus holes in the water mask. Needs a water
                 region: if none is given, water_mask is computed
                 automatically.
        "color":  original absolute HSV rules (bright + saturated).

    Returns:
        boxes

    If return_debug=True:
        boxes, debug_data
    """

    height, width = img.shape[:2]
    img_area = height * width

    # Anomaly masks are cleaner than raw color rules, so tiny blobs
    # (ripples, leaves, glints) can be rejected more aggressively.
    if min_area is None:
        min_area = 120 if method == "anomaly" else 40

    if method == "anomaly" and water_mask is None:
        if water_polygon is not None:
            water_mask = create_polygon_mask(img.shape, water_polygon)
        else:
            water_mask, _ = segment_water(img)
    elif water_mask is None and water_polygon is None and auto_water:
        water_mask, _ = segment_water(img)

    # --------------------------------------------------
    # A/B. Candidate pixels inside the valid region
    # --------------------------------------------------

    valid_mask, holes = create_valid_mask(
        img,
        water_polygon=water_polygon,
        water_mask=water_mask
    )

    dist_map = None
    anomaly_thr = None

    if method == "anomaly":

        dist_map, anomaly_mask, anomaly_thr = create_anomaly_map(
            img,
            water_mask,
            k_mad=anomaly_k,
            min_dist=anomaly_min_dist,
            edge_margin_frac=edge_margin_frac
        )

        # keep the anomaly away from the water edge (dock shadows,
        # ripples); holes are always kept
        margin = max(1, int(width * edge_margin_frac))
        inner = cv2.erode(
            valid_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
            )
        )

        anomaly_mask = cv2.bitwise_and(anomaly_mask, inner)

        candidate_mask = cv2.bitwise_or(anomaly_mask, holes)

        white_mask = anomaly_mask
        color_mask = holes

    else:

        candidate_mask, white_mask, color_mask = (
            create_candidate_mask(
                img,
                v_min=v_min,
                s_max=s_max,
                color_s_min=color_s_min,
                color_v_min=color_v_min
            )
        )

        # Holes in the water mask are litter candidates too
        candidate_mask = cv2.bitwise_or(candidate_mask, holes)

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

    # Separate touching objects (ball next to a bag, etc.)
    if split_touching:
        grouped_mask = split_touching_objects(grouped_mask, img)

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
            min_solidity=min_solidity,
            max_rot_aspect=max_rot_aspect
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
        "holes_mask": holes,
        "dist_map": dist_map,
        "anomaly_threshold": anomaly_thr,
        "method": method,
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

    if debug.get("method") == "anomaly":
        t3, t4 = "3. Water anomaly", "4. Holes in water mask"
    else:
        t3, t4 = "3. Bright objects", "4. Saturated colors"

    panels = [
        panel(original, "1. Final detections"),
        panel(debug["valid_mask"], "2. Valid region"),
        panel(debug["white_mask"], t3),
        panel(debug["color_mask"], t4),
        panel(debug["grouped_mask"], "5. Grouped candidates"),
        panel(shape_panel, "6. Shape analysis")
    ]

    row1 = cv2.hconcat(panels[0:3])
    row2 = cv2.hconcat(panels[3:6])

    return cv2.vconcat([row1, row2])
