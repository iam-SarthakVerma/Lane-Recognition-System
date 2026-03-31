# 🚗 Lane Detection with OpenCV

A computer vision pipeline for real-time lane detection on images and videos using classical CV techniques — Canny edge detection, Hough transforms, and temporal smoothing.

---

## 📸 Overview

This project implements lane detection across three scenarios:

| Script | Input | Output |
|---|---|---|
| `detection_on_image.py` | Single image | Annotated image with lane overlay |
| `detection_on_vid.py` | Highway dashcam video | Processed `.avi` with lane overlay |
| `nyc_lane_detection.py` | Urban NYC dashcam video | Processed `.avi` tuned for city streets |

---

## ⚙️ Pipeline

Each script follows the same core pipeline:

```
Raw Frame
   │
   ▼
Grayscale Conversion
   │
   ▼
Gaussian Blur  ──────────────────── (noise suppression)
   │
   ▼
Adaptive Canny Edge Detection ───── (auto-thresholded from median intensity)
   │
   ▼
Proportional ROI Mask ───────────── (trapezoid, resolution-independent)
   │
   ▼
Hough Line Transform (HoughLinesP)
   │
   ▼
Slope Filtering ─────────────────── (removes noise, splits left/right)
   │
   ▼
Polynomial Line Fitting ─────────── (one clean line per lane side)
   │
   ▼
Temporal Smoothing (video only) ──── (rolling average over N frames)
   │
   ▼
Lane Overlay + Semi-transparent Fill
```

---

## 🔧 Key Techniques

### Adaptive Canny Thresholds
Rather than hard-coded values, thresholds are computed per-frame from the median pixel intensity. This auto-calibrates to varying lighting conditions (bright roads, tunnels, overcast days).

```python
median   = np.median(blur)
sigma    = 0.33
low_thr  = int(max(0,   (1.0 - sigma) * median))
high_thr = int(min(255, (1.0 + sigma) * median))
canny    = cv2.Canny(blur, low_thr, high_thr)
```

### Proportional ROI
The region of interest is defined as a percentage of frame dimensions — no pixel coordinates — so the mask works correctly on any resolution.

```python
roi_vertices = np.array([[
    (int(0.10 * w), h),
    (int(0.45 * w), int(0.60 * h)),
    (int(0.55 * w), int(0.60 * h)),
    (int(0.95 * w), h),
]], dtype=np.int32)
```

### Slope Filtering + Line Averaging
Raw Hough lines are noisy. Lines are filtered by slope (near-horizontal lines are discarded as road noise), split into left/right groups by slope sign, then a single polynomial is fit through all candidate points per side — producing one clean, stable lane line.

### Temporal Smoothing (video)
A `deque`-based rolling window averages lane endpoints across recent frames, eliminating the frame-to-frame jitter inherent to per-frame Hough detection.

```python
class LaneSmoother:
    def __init__(self, window=12):
        self.left_buf  = deque(maxlen=window)
        self.right_buf = deque(maxlen=window)
```

---

## 🗂️ Project Structure

```
├── Data/
│   ├── test_img.jpg          # Test image for detection_on_image.py
│   ├── lane_vid2.mp4         # Highway dashcam video
│   └── Manhattan_Trim.mp4    # NYC urban dashcam video
│
├── detection_on_image.py     # Lane detection on a single image
├── detection_on_vid.py       # Lane detection on highway video
├── nyc_lane_detection.py     # Lane detection on NYC urban video
│
├── lane_detection.avi        # Output — detection_on_vid.py result
└── Manhattan_detection.avi   # Output — nyc_lane_detection.py result
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python numpy matplotlib
```

### Run on Image

```bash
python detection_on_image.py
```

Reads `./Data/test_img.jpg`, displays the result with lane overlay using Matplotlib.

### Run on Highway Video

```bash
python detection_on_vid.py
```

Reads `./Data/lane_vid2.mp4`, shows a live preview window, and saves output to `lane_detection.avi`. Press **ESC** to stop early.

### Run on NYC Video

```bash
python nyc_lane_detection.py
```

Reads `./Data/Manhattan_Trim.mp4`, shows a live preview window, and saves output to `Manhattan_detection.avi`. Press **ESC** to stop early.

---

## 🎛️ Tuning Parameters

| Parameter | Location | Effect |
|---|---|---|
| `sigma` (Canny) | All scripts | Controls edge sensitivity. Increase for noisier footage. |
| `min_slope` | `slope_filter()` | Minimum slope to count as a lane line. Lower for shallow angles (urban roads). |
| `window` | `LaneSmoother` | Rolling average window size. Higher = smoother but slower to react. |
| `threshold` | `HoughLinesP` | Minimum votes for a line. Lower = more lines detected but more noise. |
| `minLineLength` | `HoughLinesP` | Minimum pixel length of a detected segment. |
| `maxLineGap` | `HoughLinesP` | Maximum gap to bridge between collinear segments. |

### NYC vs Highway differences

The NYC script uses slightly relaxed settings to handle urban conditions:
- `min_slope = 0.35` (vs `0.4`) — Manhattan lanes have shallower angles
- `y_top = 0.62 * h` (vs `0.60`) — lower horizon for tighter city streets
- Larger `maxLineGap` — broken/painted-over lane markings are common in urban footage

---

## 📋 Limitations

- Works best on clearly marked lanes in daylight conditions
- Curved roads are not handled (linear Hough transform only detects straight lines)
- Heavy occlusion by vehicles, rain, or glare can degrade detection
- Does not distinguish solid vs dashed lane markings

---

## 🛠️ Possible Extensions

- **Curved lane detection** — fit a degree-2 polynomial instead of a straight line for highway curves
- **Deep learning** — replace the classical pipeline with a segmentation model (e.g. LaneNet, UFLD) for more robust detection
- **Lane departure warning** — add logic to check whether the vehicle is drifting outside detected lane boundaries
- **Night mode** — pre-process with CLAHE (contrast limited adaptive histogram equalization) before the Canny step

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute.
