
from pathlib import Path


def load_yolo_labels(label_path):
    """
    Load YOLO-format annotations from a text file.

    Each annotation contains:
    class_id, x_center, y_center, width, height
    """

    annotations = []

    label_path = Path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with label_path.open("r") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"Invalid annotation at line {line_number}"
                )

            class_id = int(values[0])

            x_center, y_center, width, height = map(
                float, values[1:]
            )

            annotations.append(
                (class_id, x_center, y_center, width, height)
            )

    return annotations


def yolo_to_xyxy(annotation, image_width, image_height):
    """
    Convert normalized YOLO coordinates into pixel coordinates.

    Returns:
        x1, y1, x2, y2
    """

    class_id, x_center, y_center, width, height = annotation

    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    x1 = round(x_center - width / 2)
    y1 = round(y_center - height / 2)
    x2 = round(x_center + width / 2)
    y2 = round(y_center + height / 2)

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    return x1, y1, x2, y2
