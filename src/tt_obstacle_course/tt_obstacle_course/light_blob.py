"""Colour-disc detection for the start light (pure OpenCV, testable without ROS)."""
import math

import cv2
import numpy as np

# Red wraps around hue 0 in OpenCV's 0-180 hue scale.
RED_RANGES = (([0, 100, 60], [10, 255, 255]), ([170, 100, 60], [180, 255, 255]))


def colour_mask(hsv, ranges):
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def disc_centres(mask, min_area, min_circularity=0.6, min_disc_fraction=0.45):
    """Centres (u, v) of blobs that are a disc, largest first.

    A blob counts if it is round (4*pi*A/P^2) or, failing that (a disc on a same-colour
    stick, a slightly oval or ragged print), if its largest inscribed circle still
    covers min_disc_fraction of it; the centre is then that circle's centre.
    """
    out = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < min_area:
            break
        perim = cv2.arcLength(c, True)
        if perim > 0 and 4.0 * math.pi * area / (perim * perim) >= min_circularity:
            m = cv2.moments(c)
            out.append((m['m10'] / m['m00'], m['m01'] / m['m00']))
            continue
        x, y, w, h = cv2.boundingRect(c)
        blob = np.zeros((h + 2, w + 2), np.uint8)
        cv2.drawContours(blob, [c - [x - 1, y - 1]], -1, 255, -1)
        dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
        _, r_in, _, (cu, cv_) = cv2.minMaxLoc(dist)
        if math.pi * r_in * r_in >= min_disc_fraction * area and math.pi * r_in * r_in >= min_area / 2:
            out.append((x - 1 + cu, y - 1 + cv_))
    return out
