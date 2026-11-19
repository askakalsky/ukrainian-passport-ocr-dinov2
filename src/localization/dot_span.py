"""Check a passport crop against six rows of perforation dots on the source page."""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .crnn_recrop import expanded_recrop
from .perforation_sides import fit_dot_rows
from ..recognition.perforation import PerforationResult


@dataclass
class SpanResult:
    box: list
    prediction: PerforationResult
    status: str
    baseline_box: list
    baseline_prediction: PerforationResult


def supported_span(gray, proposal, percentile):
    """Return the dot-centre extent and row pitch, or None if support is weak."""
    x, y, r, b = proposal
    width, height = r - x, b - y
    if width < 30 or height < 8 or height > width:
        return None
    left = max(0, math.floor(x - .48 * width))
    right = min(gray.shape[1], math.ceil(r + .48 * width))
    top = max(0, math.floor(y - .28 * height))
    bottom = min(gray.shape[0], math.ceil(b + .28 * height))
    scale = 320 / width
    if (bottom-top)*scale < 6:
        return None
    region = cv2.resize(gray[top:bottom, left:right], None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    contrast = cv2.morphologyEx(region, cv2.MORPH_BLACKHAT, kernel)
    mask = (contrast >= max(15, np.percentile(contrast, percentile))).astype(np.uint8)
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    w, h, area = stats[1:, 2:].T
    valid = (area >= 3) & (w >= 2) & (h >= 2) & (w <= 13) & (h <= 13)
    centers = centers[1:][valid & (w <= 2.4*h) & (h <= 2.4*w)]
    if not 50 <= len(centers) <= 520:
        return None
    start, end = (x - left) * scale, (r - left) * scale
    fit = fit_dot_rows(centers, height * scale, start, end)
    if fit is None:
        return None
    aligned, rows = fit
    points, rows = centers[aligned], rows[aligned]
    order = np.argsort(points[:, 0])
    groups = np.split(order, np.flatnonzero(np.diff(points[order, 0]) > 35) + 1)
    inside = (points[:, 0] >= start) & (points[:, 0] < end)
    group = max(groups, key=lambda indices: np.count_nonzero(inside[indices]))
    points, rows = points[group], rows[group]
    labels, counts = np.unique(rows, return_counts=True)
    if len(labels) != 6 or counts.min() < 4 or np.ptp(points[:, 0]) < .8 * (end - start):
        return None
    design = np.column_stack([points[:, 0], rows, np.ones(len(points))])
    _, pitch, _ = np.linalg.lstsq(design, points[:, 1], rcond=None)[0]
    if pitch <= 0:
        return None
    # A lone mark in the background must not pull an edge away from the field.
    for distance in (points[:, 0] - points[:, 0].min(), points[:, 0].max() - points[:, 0]):
        if len(np.unique(rows[distance <= 1.5 * pitch])) < 3:
            return None
    low = points.min(axis=0)/scale + [left, top]
    high = points.max(axis=0)/scale + [left, top]
    return (*low, *high), float(pitch / scale)


def dot_span_recrop(image, proposal, reader):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    base, prediction = expanded_recrop(gray, proposal, reader, vertical=True)
    span = supported_span(gray, proposal, 94)
    if span is None:
        return SpanResult(base, prediction, "dot_rows_unverified", base, prediction)
    (left, top, right, bottom), pitch = span
    x, y, r, b = base
    if max(x-left, y-top, right-r, bottom-b) < pitch/2:
        return SpanResult(base, prediction, "no_clear_dot_overrun", base, prediction)
    # Background specks occasionally line up with the text. Require the same
    # boundaries at a higher contrast threshold before admitting extra pixels.
    confirmation = supported_span(gray, proposal, 96)
    if confirmation is None or np.max(np.abs(np.subtract(span[0], confirmation[0]))) > .65 * pitch:
        return SpanResult(base, prediction, "disputed_dot_span", base, prediction)
    margin = .65 * pitch
    corrected = [max(0, min(x, math.floor(left-margin))), max(0, min(y, math.floor(top-margin))),
                 min(gray.shape[1], max(r, math.ceil(right+margin)+1)),
                 min(gray.shape[0], max(b, math.ceil(bottom+margin)+1))]
    x, y, r, b = corrected
    return SpanResult(corrected, reader.recognize_crop(gray[y:b, x:r]), "recovered_dot_span", base, prediction)
