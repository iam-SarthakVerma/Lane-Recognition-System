import cv2
import numpy as np
import matplotlib.pyplot as plt


# ── helpers ────────────────────────────────────────────────────────────────────

def build_roi_mask(image, vertices):
    """Return a binary mask that keeps only the polygon defined by vertices."""
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def slope_filter(lines, min_slope=0.4):
    """
    Split raw Hough lines into left / right lane candidates by slope.
    Lines with |slope| < min_slope are discarded (nearly horizontal → noise).
    """
    left, right = [], []
    if lines is None:
        return left, right
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:          # avoid division-by-zero for vertical lines
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < min_slope:
            continue
        if slope < 0:          # negative slope → left lane (image y grows downward)
            left.append(line[0])
        else:
            right.append(line[0])
    return left, right


def fit_lane_line(points, y_top, y_bottom):
    """
    Fit a single line through a list of (x1,y1,x2,y2) segments and
    return the (x1,y1,x2,y2) endpoints clipped to [y_top, y_bottom].
    Returns None when fewer than 2 points are available.
    """
    if len(points) < 2:
        return None
    xs = [p[0] for p in points] + [p[2] for p in points]
    ys = [p[1] for p in points] + [p[3] for p in points]
    poly = np.polyfit(ys, xs, 1)          # x = f(y) — more stable near-vertical
    x_bottom = int(np.polyval(poly, y_bottom))
    x_top    = int(np.polyval(poly, y_top))
    return (x_bottom, y_bottom, x_top, y_top)


def draw_lane_lines(image, left_pts, right_pts, y_top, y_bottom):
    """Draw smooth averaged lane lines onto a copy of image."""
    overlay = image.copy()
    left_line  = fit_lane_line(left_pts,  y_top, y_bottom)
    right_line = fit_lane_line(right_pts, y_top, y_bottom)

    for lane in (left_line, right_line):
        if lane:
            x1, y1, x2, y2 = lane
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 6)

    # Optionally shade the lane area
    if left_line and right_line:
        pts = np.array([[left_line[0],  left_line[1]],
                        [left_line[2],  left_line[3]],
                        [right_line[2], right_line[3]],
                        [right_line[0], right_line[1]]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 0))

    return cv2.addWeighted(image, 0.8, overlay, 0.2, 0)


# ── main pipeline ───────────────────────────────────────────────────────────────

def process(img):
    h, w = img.shape[:2]

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. Gaussian blur to suppress noise before edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Adaptive Canny — thresholds derived from median pixel value
    median   = np.median(blur)
    sigma    = 0.33
    low_thr  = int(max(0,   (1.0 - sigma) * median))
    high_thr = int(min(255, (1.0 + sigma) * median))
    canny    = cv2.Canny(blur, low_thr, high_thr)

    # 4. ROI — proportional trapezoid, works for any resolution
    roi_vertices = np.array([[
        (int(0.10 * w), h),
        (int(0.45 * w), int(0.60 * h)),
        (int(0.55 * w), int(0.60 * h)),
        (int(0.95 * w), h),
    ]], dtype=np.int32)
    roi_img = build_roi_mask(canny, roi_vertices)

    # 5. Hough transform
    lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180,
                            threshold=50,
                            minLineLength=80,
                            maxLineGap=150)

    # 6. Filter by slope and average into one line per side
    y_top    = int(0.60 * h)
    y_bottom = h
    left_pts, right_pts = slope_filter(lines)
    result = draw_lane_lines(img, left_pts, right_pts, y_top, y_bottom)
    return result


# ── entry point ─────────────────────────────────────────────────────────────────

img = cv2.imread("./Data/test_img.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = process(img)

plt.figure(figsize=(12, 6))
plt.imshow(result)
plt.title("Lane Detection — Image")
plt.axis("off")
plt.tight_layout()
plt.show()
