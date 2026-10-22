"""Line segmentation used to build the CRNN training data (notebook 01, Step 3).
The service must cut lines exactly like this so inference matches training."""
import math
import cv2
import numpy as np


def binarize(g):
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)


def deskew(g):
    b = binarize(g); h, w = g.shape
    segs = cv2.HoughLinesP(b, 1, np.pi / 720, 150, minLineLength=int(.2 * w), maxLineGap=25)
    if segs is None: return g
    a = [math.degrees(math.atan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4)]
    a = [x for x in a if abs(x) < 10]
    if not a: return g
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(np.median(a)), 1)
    return cv2.warpAffine(g, M, (w, h), borderValue=255)


def remove_rules(b):
    h, w = b.shape
    hor = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1)))
    ver = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 8)))
    return cv2.subtract(b, cv2.dilate(hor | ver, np.ones((3, 3), np.uint8))), ver


def column_bounds(ver, w):
    cov = (ver > 0).mean(0); xs, x = [], 0
    while x < w:
        if cov[x] > 0.3:
            s = x
            while x < w and cov[x] > 0.1: x += 1
            xs.append((s + x) // 2)
        x += 1
    cuts = [0] + [x for x in xs if 0.03 * w < x < 0.97 * w] + [w]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] > 40]


def line_boxes(clean, x0, x1, min_h=15, pad=8):
    strip = clean[:, x0:x1]
    prof = np.convolve((strip > 0).sum(1), np.ones(9) / 9, 'same')
    if prof.max() == 0: return []
    on, out, y, H = prof > max(3, 0.08 * prof.max()), [], 0, len(prof)
    while y < H:
        if on[y]:
            s = y
            while y < H and on[y]: y += 1
            if y - s >= min_h:
                ys, ye = max(0, s - pad), min(H, y + pad)
                cols = np.where(strip[ys:ye].sum(0) > 0)[0]
                if len(cols) > 20:
                    out.append((x0 + max(0, cols[0] - pad), ys, x0 + min(x1 - x0, cols[-1] + pad), ye))
        y += 1
    return out


def segment_page(gray):
    """Full page -> (deskewed page, [(column_index, (x0, y0, x1, y1)), ...]) in reading order."""
    g = deskew(gray)
    clean, ver = remove_rules(binarize(g))
    boxes = []
    for ci, (x0, x1) in enumerate(column_bounds(ver, g.shape[1])):
        for box in line_boxes(clean, x0, x1):
            boxes.append((ci, box))
    return g, boxes