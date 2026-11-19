"""
Automated Test Suite for Vehicle Detection, Kalman Tracking, and Bidirectional Counting.
Validates all 7 mandatory criteria:
1. Unique tracking ID assignment.
2. Kalman prediction and correction accuracy.
3. Upward line crossing detection.
4. Downward line crossing detection.
5. Duplicate count prevention.
6. Total count consistency.
7. Object disappearance and tracker cleanup.
"""

import os
import sys
import pytest
import numpy as np

# Ensure Object_Counting modules are in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import KalmanConfig, TrackerConfig, CounterConfig
from kalman_filter import VehicleKalmanFilter
from detector import Detection
from tracker import VehicleTracker, TrackedVehicle
from counter import VehicleCounter


def make_dummy_detection(cx: float, cy: float, w: int = 50, h: int = 40) -> Detection:
    """Helper to construct synthetic Detection instance."""
    x = int(cx - w / 2)
    y = int(cy - h / 2)
    cnt = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)
    return Detection(
        bbox=(x, y, w, h),
        centroid=(float(cx), float(cy)),
        area=float(w * h),
        contour=cnt
    )


# ---------------------------------------------------------------------------
# Test 1: Unique tracking ID assignment
# ---------------------------------------------------------------------------
def test_unique_id_assignment():
    """Verify newly detected objects receive unique, distinct, persistent IDs."""
    tracker = VehicleTracker(TrackerConfig(max_distance=50))

    # Frame 1: Two distinct vehicles appear
    det1 = make_dummy_detection(100, 100)
    det2 = make_dummy_detection(300, 100)
    tracks_f1 = tracker.update([det1, det2])

    assert len(tracks_f1) == 2, "Expected 2 active tracks"
    ids_f1 = {t.track_id for t in tracks_f1}
    assert len(ids_f1) == 2, "Assigned IDs must be unique"
    assert 1 in ids_f1 and 2 in ids_f1, "Expected sequential unique IDs starting at 1 and 2"

    # Frame 2: Vehicles move slightly, third vehicle appears
    det1_moved = make_dummy_detection(102, 108)
    det2_moved = make_dummy_detection(301, 107)
    det3 = make_dummy_detection(500, 150)
    tracks_f2 = tracker.update([det1_moved, det2_moved, det3])

    assert len(tracks_f2) == 3, "Expected 3 active tracks"
    id_map = {t.track_id: t.centroid for t in tracks_f2}

    # Verify ID persistence for existing vehicles
    assert 1 in id_map and 2 in id_map, "Original vehicle IDs must persist"
    assert 3 in id_map, "New vehicle must receive next unique ID (3)"


# ---------------------------------------------------------------------------
# Test 2: Kalman prediction and correction
# ---------------------------------------------------------------------------
def test_kalman_prediction_and_correction():
    """Verify Kalman filter kinematic state estimation, noise filtering, and coasting."""
    cfg = KalmanConfig(dt=1.0, process_noise_pos=0.05, process_noise_vel=0.1, measurement_noise=2.0)
    kf = VehicleKalmanFilter(initial_x=100.0, initial_y=200.0, config=cfg)

    # Initial position check
    assert kf.position == (100.0, 200.0)

    # Simulate constant velocity motion: vx = 4 px/frame, vy = 8 px/frame
    true_x, true_y = 100.0, 200.0
    vx, vy = 4.0, 8.0

    for step in range(1, 11):
        true_x += vx
        true_y += vy
        
        # Prediction stage
        pred_x, pred_y = kf.predict()
        assert np.isfinite(pred_x) and np.isfinite(pred_y)

        # Correction stage with slight sensor noise
        measured_x = true_x + np.random.uniform(-0.5, 0.5)
        measured_y = true_y + np.random.uniform(-0.5, 0.5)
        corr_x, corr_y = kf.correct(measured_x, measured_y)

        assert abs(corr_x - true_x) < 5.0, f"Corrected X deviated too far from true X at step {step}"
        assert abs(corr_y - true_y) < 5.0, f"Corrected Y deviated too far from true Y at step {step}"

    # Verify velocity convergence
    est_vx, est_vy = kf.velocity
    assert abs(est_vx - vx) < 1.5, f"Estimated vx {est_vx} did not converge to {vx}"
    assert abs(est_vy - vy) < 1.5, f"Estimated vy {est_vy} did not converge to {vy}"

    # Test Coasting: When detection measurements cease, predictions must continue extrapolating
    coast_start_y = kf.position[1]
    for _ in range(3):
        pred_x, pred_y = kf.predict()
    assert pred_y > coast_start_y + 15.0, "Kalman filter should extrapolate forward during coasting"


# ---------------------------------------------------------------------------
# Test 3: Upward line crossing
# ---------------------------------------------------------------------------
def test_upward_line_crossing():
    """Verify that a vehicle moving from bottom to top crossing virtual line increments upward count."""
    counter = VehicleCounter(CounterConfig(line_y_ratio=0.5, buffer_zone=15))
    frame_height = 600
    line_y = counter.get_line_y(frame_height)  # 300

    det = make_dummy_detection(200, 420)
    track = TrackedVehicle(track_id=101, detection=det)

    # Vehicle moves upwards: 420 -> 360 -> 320 -> 290 (crosses 300 from below)
    trajectory = [360, 320, 290]
    for y in trajectory:
        track.kalman.predict()
        track.kalman.correct(200, y)
        track.centroid = (200.0, float(y))
        track.history.append((200.0, float(y)))
        counter.update([track], frame_height=frame_height)

    assert counter.count_up == 1, "Upward count must be exactly 1"
    assert counter.count_down == 0, "Downward count must be 0"
    assert counter.total_count == 1, "Total count must be 1"
    assert track.counted is True, "Track must be flagged as counted"
    assert track.direction == "UP", "Track direction must be 'UP'"


# ---------------------------------------------------------------------------
# Test 4: Downward line crossing
# ---------------------------------------------------------------------------
def test_downward_line_crossing():
    """Verify that a vehicle moving from top to bottom crossing virtual line increments downward count."""
    counter = VehicleCounter(CounterConfig(line_y_ratio=0.5, buffer_zone=15))
    frame_height = 600
    line_y = counter.get_line_y(frame_height)  # 300

    det = make_dummy_detection(400, 180)
    track = TrackedVehicle(track_id=202, detection=det)

    # Vehicle moves downwards: 180 -> 240 -> 280 -> 315 (crosses 300 from above)
    trajectory = [240, 280, 315]
    for y in trajectory:
        track.kalman.predict()
        track.kalman.correct(400, y)
        track.centroid = (400.0, float(y))
        track.history.append((400.0, float(y)))
        counter.update([track], frame_height=frame_height)

    assert counter.count_down == 1, "Downward count must be exactly 1"
    assert counter.count_up == 0, "Upward count must be 0"
    assert counter.total_count == 1, "Total count must be 1"
    assert track.counted is True, "Track must be flagged as counted"
    assert track.direction == "DOWN", "Track direction must be 'DOWN'"


# ---------------------------------------------------------------------------
# Test 5: Preventing duplicate counts
# ---------------------------------------------------------------------------
def test_prevent_duplicate_counts():
    """Verify that centroid fluctuations or lingering near line does not re-trigger counting."""
    counter = VehicleCounter(CounterConfig(line_y_ratio=0.5, buffer_zone=15))
    frame_height = 600

    det = make_dummy_detection(300, 250)
    track = TrackedVehicle(track_id=303, detection=det)

    # Crosses line (y=300) downwards
    trajectory = [270, 305]
    for y in trajectory:
        track.centroid = (300.0, float(y))
        track.history.append((300.0, float(y)))
        counter.update([track], frame_height=frame_height)

    assert counter.count_down == 1, "Initial crossing must increment count to 1"

    # Vehicle lingers or oscillates right around line for 20 frames
    jitter_y = [304, 302, 306, 301, 305, 299, 303, 305, 302, 306]
    for y in jitter_y:
        track.centroid = (300.0, float(y))
        track.history.append((300.0, float(y)))
        counter.update([track], frame_height=frame_height)

    assert counter.count_down == 1, "Count must remain strictly 1 despite line oscillation"
    assert counter.count_up == 0
    assert counter.total_count == 1


# ---------------------------------------------------------------------------
# Test 6: Total count consistency
# ---------------------------------------------------------------------------
def test_total_count_consistency():
    """Verify bidirectional aggregation: Total = Upward + Downward across multiple tracks."""
    counter = VehicleCounter(CounterConfig(line_y_ratio=0.5, buffer_zone=15))
    frame_height = 600

    # Simulate 3 upward vehicles and 4 downward vehicles
    up_tracks = [
        TrackedVehicle(track_id=10 + i, detection=make_dummy_detection(100 + i * 50, 450))
        for i in range(3)
    ]
    down_tracks = [
        TrackedVehicle(track_id=20 + j, detection=make_dummy_detection(350 + j * 50, 150))
        for j in range(4)
    ]

    all_tracks = up_tracks + down_tracks

    # Move upward tracks across line
    for t in up_tracks:
        t.history.append((t.centroid[0], 350.0))
        t.centroid = (t.centroid[0], 280.0)
        t.history.append(t.centroid)

    # Move downward tracks across line
    for t in down_tracks:
        t.history.append((t.centroid[0], 250.0))
        t.centroid = (t.centroid[0], 330.0)
        t.history.append(t.centroid)

    counter.update(all_tracks, frame_height=frame_height)

    assert counter.count_up == 3, f"Expected count_up=3, got {counter.count_up}"
    assert counter.count_down == 4, f"Expected count_down=4, got {counter.count_down}"
    assert counter.total_count == counter.count_up + counter.count_down == 7

    # Test reset functionality
    counter.reset()
    assert counter.count_up == 0
    assert counter.count_down == 0
    assert counter.total_count == 0
    assert len(counter.counted_ids) == 0


# ---------------------------------------------------------------------------
# Test 7: Object disappearance and tracker cleanup
# ---------------------------------------------------------------------------
def test_object_disappearance_cleanup():
    """Verify inactive tracks are held in coasting mode then purged after max_disappeared frames."""
    max_disp = 5
    tracker = VehicleTracker(TrackerConfig(max_distance=50, max_disappeared=max_disp))

    det = make_dummy_detection(200, 200)
    tracker.update([det])
    assert 1 in tracker.tracks, "Vehicle 1 must be registered"

    # Frame with no detections: track enters coasting
    for frame in range(1, max_disp + 1):
        active = tracker.update([])  # No detections
        assert 1 in tracker.tracks, f"Track should survive missing frame {frame} (coasting)"
        assert tracker.tracks[1].disappeared == frame

    # Frame exceeding max_disappeared: track must be purged
    active_after = tracker.update([])
    assert 1 not in tracker.tracks, "Track must be deregistered after exceeding max_disappeared"
    assert len(active_after) == 0, "Active track list should be empty"
