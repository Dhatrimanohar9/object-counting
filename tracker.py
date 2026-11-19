"""
Multi-Object Tracker Module.
Maintains persistent unique vehicle IDs across frames, implements Hungarian/Euclidean
data association, and integrates dedicated Kalman Filters for each tracked object.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from typing import Dict, List, Tuple, Optional
from config import TrackerConfig, KalmanConfig
from kalman_filter import VehicleKalmanFilter
from detector import Detection


class TrackedVehicle:
    """
    Encapsulates state, trajectory history, and dedicated Kalman Filter for one vehicle.
    """

    def __init__(self, track_id: int, detection: Detection, kalman_config: Optional[KalmanConfig] = None, max_history: int = 30):
        self.track_id: int = track_id
        self.bbox: Tuple[int, int, int, int] = detection.bbox
        self.centroid: Tuple[float, float] = detection.centroid
        
        # Dedicated Kalman filter for continuous kinematics
        self.kalman = VehicleKalmanFilter(
            initial_x=detection.centroid[0],
            initial_y=detection.centroid[1],
            config=kalman_config
        )
        self.predicted_pos: Tuple[float, float] = detection.centroid

        # Trajectory trail history
        self.history: deque = deque(maxlen=max_history)
        self.history.append(detection.centroid)

        # Lifecycle tracking
        self.disappeared: int = 0
        self.age: int = 1
        
        # Counting attributes
        self.counted: bool = False
        self.direction: Optional[str] = None  # 'UP' or 'DOWN'
        self.counted_frame: Optional[int] = None

    def predict(self) -> Tuple[float, float]:
        """
        Advance vehicle kinematics using Kalman Filter prediction step.
        """
        self.predicted_pos = self.kalman.predict()
        return self.predicted_pos

    def update_with_detection(self, detection: Detection):
        """
        Update state using Kalman Filter correction step with observed detection.
        """
        corrected = self.kalman.correct(detection.centroid[0], detection.centroid[1])
        self.centroid = corrected
        self.bbox = detection.bbox
        self.history.append(self.centroid)
        self.disappeared = 0
        self.age += 1

    def update_coasting(self):
        """
        Coasting mode: propagate state using Kalman prediction when detection is missed.
        """
        self.centroid = self.predicted_pos
        # Shift bounding box by estimated velocity
        vx, vy = self.kalman.velocity
        x, y, w, h = self.bbox
        new_x = int(x + vx)
        new_y = int(y + vy)
        self.bbox = (new_x, new_y, w, h)
        self.history.append(self.centroid)
        self.disappeared += 1
        self.age += 1


class VehicleTracker:
    """
    Manages track lifecycle (creation, association, coasting, and deletion).
    """

    def __init__(self, config: Optional[TrackerConfig] = None, kalman_config: Optional[KalmanConfig] = None):
        self.config = config or TrackerConfig()
        self.kalman_config = kalman_config or KalmanConfig()
        self.next_id: int = 1
        self.tracks: Dict[int, TrackedVehicle] = {}

    def register(self, detection: Detection) -> TrackedVehicle:
        """
        Assign a new unique ID and register new tracked vehicle.
        """
        track = TrackedVehicle(
            track_id=self.next_id,
            detection=detection,
            kalman_config=self.kalman_config,
            max_history=self.config.max_history
        )
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track

    def deregister(self, track_id: int):
        """
        Remove tracked vehicle after prolonged disappearance.
        """
        if track_id in self.tracks:
            del self.tracks[track_id]

    def reset(self):
        """
        Clear all tracks and reset ID counter.
        """
        self.tracks.clear()
        self.next_id = 1

    def update(self, detections: List[Detection]) -> List[TrackedVehicle]:
        """
        Associate detections with existing tracks using Hungarian algorithm on predicted positions.
        
        Args:
            detections: List of current frame detections.
            
        Returns:
            List of active TrackedVehicle instances.
        """
        # Step 1: Predict new positions for all active tracks via Kalman Filter
        track_ids = list(self.tracks.keys())
        for tid in track_ids:
            self.tracks[tid].predict()

        # Step 2: If no detections exist, advance all tracks in coasting mode
        if len(detections) == 0:
            for tid in track_ids:
                self.tracks[tid].update_coasting()
                if self.tracks[tid].disappeared > self.config.max_disappeared:
                    self.deregister(tid)
            return list(self.tracks.values())

        # Step 3: If no active tracks exist, register all detections
        if len(self.tracks) == 0:
            for det in detections:
                self.register(det)
            return list(self.tracks.values())

        # Step 4: Compute Euclidean distance cost matrix between Kalman predicted
        # positions and incoming detection centroids
        N = len(track_ids)
        M = len(detections)
        cost_matrix = np.zeros((N, M), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            pred_pos = self.tracks[tid].predicted_pos
            for j, det in enumerate(detections):
                dx = pred_pos[0] - det.centroid[0]
                dy = pred_pos[1] - det.centroid[1]
                cost_matrix[i, j] = np.sqrt(dx * dx + dy * dy)

        # Step 5: Solve global optimal assignment using Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_detections = set()

        for r, c in zip(row_indices, col_indices):
            # Check gating distance threshold
            if cost_matrix[r, c] <= self.config.max_distance:
                tid = track_ids[r]
                self.tracks[tid].update_with_detection(detections[c])
                assigned_tracks.add(tid)
                assigned_detections.add(c)

        # Step 6: Handle unassigned active tracks (coast mode or deregistration)
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid].update_coasting()
                if self.tracks[tid].disappeared > self.config.max_disappeared:
                    self.deregister(tid)

        # Step 7: Handle unassigned detections (register as new tracks)
        for j, det in enumerate(detections):
            if j not in assigned_detections:
                self.register(det)

        return list(self.tracks.values())
