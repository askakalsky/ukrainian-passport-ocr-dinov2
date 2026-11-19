"""Select an expansion without removing pixels from the proposal."""
from itertools import product
import math

import cv2

MARGINS = (0., .04, .08)
VERTICAL_MARGINS = (0., .06)


def expanded_recrop(image, box, reader, *, vertical=False):
    """Compare candidate windows using the geometric mean of CRNN character scores."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    x, y, r, b = box
    width, height = r - x, b - y
    vertical_margins = VERTICAL_MARGINS if vertical else (0.,)
    seen = set()
    best = None
    for left, right, top, bottom in sorted(product(MARGINS, MARGINS, vertical_margins, vertical_margins), key=sum):
        candidate = (max(0, round(x-left*width)), max(0, round(y-top*height)),
                     min(gray.shape[1], round(r+right*width)), min(gray.shape[0], round(b+bottom*height)))
        if candidate in seen:
            continue
        seen.add(candidate)
        x0, y0, x1, y1 = candidate
        prediction = reader.recognize_crop(gray[y0:y1, x0:x1])
        confidence = (sum(math.log(max(1e-7, float(value))) for value in prediction.character_scores) /
                      len(prediction.character_scores)) if prediction.character_scores else float("-inf")
        if best is None or confidence > best[0]:
            best = confidence, candidate, prediction
    _, chosen, prediction = best
    return list(chosen), prediction
