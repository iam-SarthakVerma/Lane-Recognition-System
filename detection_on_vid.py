import cv2
import numpy as np
from collections import deque


# ── helpers ────────────────────────────────────────────────────────────────────

def build_roi_mask(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def slope_filter(lines, min_slope=0.4):
    left, right = [], []
    if lines is None:
        return left, right
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < min_slope:
            continue
        (left if slope < 0 else right).append(line[0])
    return left, right


def fit_lane_line(points, y_top, y_bottom):
    if len(points) < 2:
        return None
    xs = [p[0] for p in points] + [p[2] for p in points]
    ys = [p[1] for p in points] + [p[3] for p in points]
    try:
        poly = np.polyfit(ys, xs, 1)
    except np.RankWarning:
        return None
    x_bottom = int(np.polyval(poly, y_bottom))
    x_top    = int(np.polyval(poly, y_top))
    return (x_bottom, y_bottom, x_top, y_top)


class LaneSmoother:
    """
    Temporal smoother — keeps a rolling window of recent lane endpoints
    and returns their average, eliminating per-frame jitter.
    """
    def __init__(self, window=12):
        self.left_buf  = deque(maxlen=window)
        self.right_buf = deque(maxlen=window)

    def update(self, left, right):
        if left:
            self.left_buf.append(left)
        if right:
            self.right_buf.append(right)

    def get_smooth(self):
        def avg(buf):
            if not buf:
                return None
            return tuple(int(np.mean([b[i] for b in buf])) for i in range(4))
        return avg(self.left_buf), avg(self.right_buf)


def draw_lane_lines(image, left_line, right_line):
    overlay = image.copy()
    for lane in (left_line, right_line):
        if lane:
            x1, y1, x2, y2 = lane
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 6)
    if left_line and right_line:
        pts = np.array([[left_line[0],  left_line[1]],
                        [left_line[2],  left_line[3]],
                        [right_line[2], right_line[3]],
                        [right_line[0], right_line[1]]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
    return cv2.addWeighted(image, 0.8, overlay, 0.2, 0)


# ── per-frame pipeline ─────────────────────────────────────────────────────────

smoother = LaneSmoother(window=12)


def process(img):
    h, w = img.shape[:2]
    y_top    = int(0.60 * h)
    y_bottom = h

    # Grayscale + Gaussian blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Canny
    median   = np.median(blur)
    sigma    = 0.33
    canny    = cv2.Canny(blur,
                         int(max(0,   (1 - sigma) * median)),
                         int(min(255, (1 + sigma) * median)))

    # Proportional ROI trapezoid
    roi_vertices = np.array([[
        (int(0.05 * w), h),
        (int(0.44 * w), y_top),
        (int(0.56 * w), y_top),
        (int(0.95 * w), h),
    ]], dtype=np.int32)
    roi_img = build_roi_mask(canny, roi_vertices)

    # Hough
    lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180,
                            threshold=30,
                            minLineLength=40,
                            maxLineGap=120)

    # Average lines per side
    left_pts, right_pts = slope_filter(lines)
    left_line  = fit_lane_line(left_pts,  y_top, y_bottom)
    right_line = fit_lane_line(right_pts, y_top, y_bottom)

    # Temporal smoothing
    smoother.update(left_line, right_line)
    left_smooth, right_smooth = smoother.get_smooth()

    return draw_lane_lines(img, left_smooth, right_smooth)


# ── video I/O ──────────────────────────────────────────────────────────────────

cap         = cv2.VideoCapture("./Data/lane_vid2.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc      = cv2.VideoWriter_fourcc(*"XVID")
out         = cv2.VideoWriter("lane_detection.avi", fourcc, fps,
                              (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    try:
        frame = process(frame)
        out.write(frame)
        cv2.imshow("Lane Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit early
            break
    except Exception as e:
        print(f"[WARN] Skipped frame: {e}")

cap.release()
out.release()
cv2.destroyAllWindows()
