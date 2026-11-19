# Intelligent Vehicle Detection, Tracking and Bidirectional Counting System

An academic and industrial-grade computer vision application developed in Python with OpenCV for automated traffic surveillance monitoring, multi-vehicle tracking using discrete-time Kalman Filters, virtual line crossing analysis, and bidirectional counting.

---

## 1. System Overview & Core Capabilities

The system fulfills all 7 mandatory academic assignment objectives:
1. **Automated Vehicle Detection**: Employs adaptive Gaussian Mixture-based Background/Foreground Segmentation (MOG2 / KNN) with morphological spatial filtering and shadow suppression.
2. **Unique Object Identification**: Generates and persists unique tracking IDs using optimal Euclidean/Hungarian bipartite data association across consecutive frames.
3. **Mathematically Rigorous Kalman Filtering**: Uses OpenCV `cv2.KalmanFilter` with a 4-state kinematic model $[x, y, v_x, v_y]^T$ for every tracked vehicle, enabling accurate trajectory smoothing and coasting during temporary detection dropouts.
4. **Configurable Virtual Counting Line**: Supports arbitrary line positioning with a dual-boundary hysteresis buffer zone ($\pm \delta$) to prevent false triggers from detection jitter.
5. **Single-Crossing Guarantee**: Enforces an atomic crossing-state machine preventing duplicate increments.
6. **Bidirectional Traffic Counting**: Distinguishes traffic flows into **UPWARD (Northbound)**, **DOWNWARD (Southbound)**, and **TOTAL** counters based on signed vertical displacement vectors.
7. **Real-Time HUD Dashboard & Controls**: Overlays an interactive semi-transparent telemetry dashboard displaying live counts, active tracks, FPS, vehicle bounding boxes, centroid trails, and Kalman prediction crosshairs.

---

## 2. Mathematical Formulation of the Kalman Filter

Each detected vehicle is assigned an independent linear discrete-time Kalman Filter governed by Newtonian kinematics:

### State Vector
$$X_k = \begin{bmatrix} x_k \\ y_k \\ v_{x,k} \\ v_{y,k} \end{bmatrix}$$
where $(x_k, y_k)$ is the centroid coordinates in pixels, and $(v_{x,k}, v_{y,k})$ is the velocity in pixels per frame.

### Measurement Vector
$$Z_k = \begin{bmatrix} z_{x,k} \\ z_{y,k} \end{bmatrix}$$
representing the observed contour centroid from background subtraction.

### State Transition Matrix ($F$)
Assuming constant velocity over normalized frame interval $\Delta t = 1$:
$$F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### Measurement Matrix ($H$)
Maps the 4D state vector to the 2D measurement space:
$$H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

### Covariance Matrices
- **Process Noise Covariance ($Q$)**: Models vehicle acceleration and maneuvering noise:
  $$Q = \text{diag}(q_{\text{pos}}, q_{\text{pos}}, q_{\text{vel}}, q_{\text{vel}}), \quad q_{\text{pos}}=0.1, \; q_{\text{vel}}=0.5$$
- **Measurement Noise Covariance ($R$)**: Models centroid extraction uncertainty:
  $$R = \begin{bmatrix} r & 0 \\ 0 & r \end{bmatrix}, \quad r = 4.0$$
- **Error Covariance ($P_0$)**: Initialized with high uncertainty ($P_0 = 10 \cdot I_4$).

### Prediction Stage
$$\hat{X}_k^- = F \hat{X}_{k-1}$$
$$P_k^- = F P_{k-1} F^T + Q$$

### Correction (Measurement Update) Stage
$$K_k = P_k^- H^T (H P_k^- H^T + R)^{-1}$$
$$\hat{X}_k = \hat{X}_k^- + K_k (Z_k - H \hat{X}_k^-)$$
$$P_k = (I - K_k H) P_k^-$$

When a detection is temporarily occluded or missed, the system operates in **Coasting Mode**, propagating state estimates purely using $\hat{X}_k^-$ without measurement correction until re-acquired or pruned after `max_disappeared` frames.

---

## 3. Project Structure

```
Object_Counting/
├── main.py                     # Primary execution pipeline & OpenCV UI loop
├── gui.py                      # Lightweight Tkinter GUI launcher with video picker & sliders
├── detector.py                 # MOG2/KNN background subtraction & contour filtering
├── kalman_filter.py            # Mathematically rigorous 4D kinematic Kalman Filter
├── tracker.py                  # Hungarian data association & track lifecycle manager
├── counter.py                  # Hysteresis-buffered virtual line & bidirectional counting
├── config.py                   # Centralized parameter configuration
├── generate_sample_video.py    # Deterministic synthetic traffic video generator
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
│
├── videos/
│   ├── traffic_video.mp4       # 720p 30fps bidirectional highway surveillance benchmark
│   ├── traffic_real.mp4        # Real-world traffic surveillance video (Intel OpenVINO)
│   └── sobral_traffic.mp4      # Highway traffic video (Sobral BGSLibrary)
│
├── outputs/
│   ├── screenshots/            # High-resolution runtime screenshots (key 'S')
│   └── processed_videos/       # Annotated video recordings with HUD overlay
│
├── docs/
│   ├── project_report.md       # Comprehensive 21-section academic report
│   └── generate_pdf.py         # ReportLab PDF compiler
│
└── tests/
    └── test_counter.py         # Automated pytest suite validating all 7 criteria
```

---

## 4. Installation & Requirements

### Prerequisites
- Python 3.9+ (tested on Python 3.11)
- macOS, Linux, or Windows

### Setup Environment
```bash
# Clone or navigate to the project directory
cd Object_Counting

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Execution Instructions

### A. Quick Start (Default Benchmark)
Run the application directly with zero arguments:
```bash
python main.py
```
This launches the system with the default 720p bidirectional traffic video, displaying real-time tracking bounding boxes, unique IDs, Kalman predictions, and the interactive HUD dashboard.

### B. Interactive GUI Launcher
For a visual file picker and parameter tuning sliders:
```bash
python main.py --gui
# or
python gui.py
```

### C. Command-Line Options
```bash
# Run on a custom video file
python main.py --video videos/traffic_real.mp4

# Run with KNN background subtraction instead of MOG2
python main.py --method KNN

# Adjust counting line position (normalized vertical ratio 0.1 to 0.9)
python main.py --line-pos 0.60

# Run in headless mode (no GUI window, ideal for servers/benchmarks)
python main.py --no-display

# Disable video recording output
python main.py --no-save
```

### D. Interactive Runtime Keyboard Controls
While the OpenCV video window is active:
- **`Q` or `ESC`** : Quit the application and finalize the video recording.
- **`P` or `SPACE`** : Pause or resume video playback.
- **`R`** : Reset counting statistics and active tracker states.
- **`S`** : Capture and save an immediate high-resolution screenshot to `outputs/screenshots/`.

---

## 6. Automated Testing Suite

The project includes an automated test suite verifying all 7 mandatory requirements:
```bash
pytest tests/test_counter.py -v
```

### Test Coverage Summary:
| # | Test Function | Verified Behavior |
|---|---|---|
| 1 | `test_unique_id_assignment` | Monotonic, non-duplicate unique ID creation and persistence |
| 2 | `test_kalman_prediction_and_correction` | Kinematic state convergence, velocity estimation, and coasting |
| 3 | `test_upward_line_crossing` | Accurate detection of bottom-to-top crossing, incrementing UP count |
| 4 | `test_downward_line_crossing` | Accurate detection of top-to-bottom crossing, incrementing DOWN count |
| 5 | `test_prevent_duplicate_counts` | Single-count guarantee during centroid jitter/oscillation near line |
| 6 | `test_total_count_consistency` | Mathematical identity: $Total = Upward + Downward$ and reset verification |
| 7 | `test_object_disappearance_cleanup` | Coasting persistence and automatic deregistration after `max_disappeared` |

---

## 7. Video Datasets & Attribution

1. **Synthetic Traffic Benchmark (`videos/traffic_video.mp4`)**:
   - 1280x720 @ 30 FPS, 750 frames.
   - Ground truth: 6 Upward vehicles, 6 Downward vehicles, Total = 12 vehicles.
   - Generated by `generate_sample_video.py` for reproducible offline testing.
2. **Intel OpenVINO Sample Video (`videos/traffic_real.mp4`)**:
   - Public domain traffic surveillance dataset provided by Intel IoT DevKit.
3. **Sobral Vehicle Detection Dataset (`videos/sobral_traffic.mp4`)**:
   - Academic benchmark provided by Andrews Sobral (BGSLibrary vehicle surveillance series).

---

## 8. Academic Project Report

The complete academic project report is available in two formats:
- Markdown: `docs/project_report.md`
- Submission PDF: `submission/Vehicle_Counting_Project_Report.pdf` (compiled via `docs/generate_pdf.py`).
