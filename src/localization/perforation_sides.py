"""Refine the horizontal extent of an approximately localized dot-matrix field."""

import cv2
import numpy as np


def dot_components(gray, detector_height):
    # Local contrast tolerates the gradual background changes on passport paper.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    threshold = max(8, float(np.percentile(response, 92)))
    mask = np.uint8(response >= threshold)
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    w, h, area = stats[:, 2:].T
    valid = (w >= 2) & (h >= 2) & (w <= 8) & (h <= 8) & (area >= 3)
    valid &= (w <= 2*h) & (h <= 2*w) & (h <= .15*detector_height)
    valid[0] = False
    # Use a looser detection pass when vetoing shrinkage. A candidate end can
    # miss faint or elongated dots, even when the interior rows fit well.
    weak = np.uint8(response >= max(5, threshold*.52))
    _, _, weak_stats, _ = cv2.connectedComponentsWithStats(weak)
    ww, wh, wa = weak_stats[:, 2:].T
    plausible = (ww >= 2) & (wh >= 2) & (ww <= 16) & (wh <= 12) & (wa >= 4)
    plausible &= (ww <= 4*wh) & (wh <= 3*ww)
    plausible[0] = False
    return stats[valid], centers[valid], weak_stats[plausible]


def fit_dot_rows(centers, detector_height, left, right):
    """Find the common slope and row pitch using the phase of dot centers."""
    x, y = centers.T
    detector_width = right-left
    dx = x - (left+right)/2
    core = (x >= left) & (x < right)
    if core.sum() < 38:
        return None
    slopes = np.linspace(-.28, .28, 71)
    pitches = np.linspace(max(4., .065*detector_height), .205*detector_height, 40)
    if pitches[-1] <= pitches[0]:
        return None
    corrected = y[None, core] - slopes[:, None]*dx[None, core]
    phase = corrected[:, None, :] / pitches[None, :, None]
    coherence = np.exp(2j*np.pi*phase).mean(axis=-1)
    slope_idx, pitch_idx = np.unravel_index(np.abs(coherence).argmax(), coherence.shape)
    peak = coherence[slope_idx, pitch_idx]
    if abs(peak) < .34:
        return None
    slope, pitch = slopes[slope_idx], pitches[pitch_idx]
    rows = (y - slope*dx)/pitch - np.angle(peak)/(2*np.pi)
    nearest = np.rint(rows).astype(int)
    aligned = np.abs(rows-nearest) <= .21
    # A true field covers several adjacent rows and most of the proposal width.
    if aligned.sum() < 35:
        return None
    labels = np.arange(nearest[aligned].min(), nearest[aligned].max()+1)
    counts = np.array([np.count_nonzero(core & aligned & (nearest == row)) for row in labels])
    if len(counts) < 4:
        return None
    if len(counts) > 6:
        start = int(np.convolve(counts, np.ones(6), "valid").argmax())
        labels = labels[start:start+6]
    good = aligned & np.isin(nearest, labels)
    hist = np.histogram(x[good], bins=8, range=(left, right))[0]
    if np.count_nonzero(hist >= 3) < 7 or np.ptp(x[good]) < .78*detector_width:
        return None
    return good, nearest


def refine_perforation_sides(image, box):
    """Return a conservative horizontal recrop or the detector's original box."""
    x, y, r, b = box
    width, height = r-x, b-y
    if width < 30 or height < 8:
        return box, "small_proposal"
    h, w = image.shape[:2]
    search_left = max(0, x-round(.16*width))
    search_right = min(w, r+round(.16*width))
    crop = image[y:b, search_left:search_right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = 480/width
    gray = cv2.resize(gray, (round(gray.shape[1]*scale), max(1, round(height*scale))),
                      interpolation=cv2.INTER_AREA if scale <= 1 else cv2.INTER_LINEAR)
    stats, centers, possible_dots = dot_components(gray, height*scale)
    if not 38 <= len(stats) <= 520:
        return box, "ambiguous_dot_components"
    left, right = (x-search_left)*scale, (r-search_left)*scale
    fit = fit_dot_rows(centers, height*scale, left, right)
    if fit is None:
        return box, "no_coherent_dot_rows"
    good, rows = fit
    aligned = stats[good]
    centers = centers[good, 0]
    rows = rows[good]
    order = centers.argsort()
    centers, aligned, rows = centers[order], aligned[order], rows[order]
    # A gap beyond several dot pitches separates unrelated edge clutter.
    splits = np.flatnonzero(np.diff(centers) > 33) + 1
    bands = np.split(np.arange(len(aligned)), splits)
    band = max(bands, key=len)
    if len(band) < 35 or np.ptp(centers[band]) < .78*(right-left):
        return box, "disconnected_dot_rows"
    margin = max(4, float(np.median(aligned[band, 2]))*1.25)
    new_left = int(np.floor(search_left + (aligned[band, 0].min()-margin)/scale))
    new_right = int(np.ceil(search_left + (np.max(aligned[band, 0]+aligned[band, 2])+margin)/scale))
    for side, edge in (("left", centers[band].min()), ("right", centers[band].max())):
        support = band[np.abs(centers[band]-edge) <= 18]
        # Each end needs support from more than one row.
        if len(np.unique(rows[support])) < 2:
            if side == "left":
                new_left = x
            else:
                new_right = r
    new_left, new_right = max(0, new_left), min(w, new_right)
    dot_left = possible_dots[:, 0]
    dot_right = dot_left + possible_dots[:, 2]
    if new_left > x and np.any((dot_right > left) & (dot_left < (new_left-search_left)*scale)):
        new_left = x
    if new_right < r and np.any((dot_left < right) & (dot_right > (new_right-search_left)*scale)):
        new_right = r
    if abs(new_left-x) > .13*width or abs(new_right-r) > .13*width:
        return box, "edge_shift_too_large"
    if (new_right-new_left) < .83*width:
        return box, "field_too_narrow"
    result = [new_left, y, new_right, b]
    return (result, "refined_dot_sides") if result != list(box) else (box, "unchanged_dot_sides")
