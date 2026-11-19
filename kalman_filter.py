"""
Kalman Filter Module for Vehicle Tracking.
Implements a 2D kinematic discrete-time linear Kalman Filter for continuous position and
velocity estimation of tracked vehicles.

State Vector:
    X = [x, y, vx, vy]^T
    where:
        x, y   : 2D centroid coordinates in pixel space
        vx, vy : 2D centroid velocities in pixels per frame

Measurement Vector:
    Z = [zx, zy]^T
    where:
        zx, zy : Observed contour centroid from detection module
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from config import KalmanConfig


class VehicleKalmanFilter:
    """
    Dedicated Kalman Filter instance for a single vehicle track.
    
    Discrete-Time State Space Equations:
        Prediction:
            X_k|k-1 = F * X_k-1|k-1
            P_k|k-1 = F * P_k-1|k-1 * F^T + Q
            
        Correction (Measurement Update):
            K_k     = P_k|k-1 * H^T * (H * P_k|k-1 * H^T + R)^(-1)
            X_k|k   = X_k|k-1 + K_k * (Z_k - H * X_k|k-1)
            P_k|k   = (I - K_k * H) * P_k|k-1
    """

    def __init__(self, initial_x: float, initial_y: float, config: Optional[KalmanConfig] = None):
        """
        Initialize Kalman Filter for a newly detected vehicle.
        
        Args:
            initial_x: Initial horizontal pixel coordinate of vehicle centroid.
            initial_y: Initial vertical pixel coordinate of vehicle centroid.
            config: Kalman configuration with noise covariances and time delta.
        """
        self.config = config or KalmanConfig()
        dt = self.config.dt

        # 4 dynamic params (x, y, vx, vy), 2 measurement params (x, y), 0 control params
        self.kf = cv2.KalmanFilter(4, 2, 0)

        # State Transition Matrix F (4x4)
        # [ x_k  ]   [ 1  0  dt  0 ] [ x_k-1  ]
        # [ y_k  ] = [ 0  1  0  dt ] [ y_k-1  ]
        # [ vx_k ]   [ 0  0  1   0 ] [ vx_k-1 ]
        # [ vy_k ]   [ 0  0  0   1 ] [ vy_k-1 ]
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.kf.transitionMatrix = F

        # Measurement Matrix H (2x4)
        # Maps state space [x, y, vx, vy] to measurement space [x, y]
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        self.kf.measurementMatrix = H

        # Process Noise Covariance Q (4x4)
        # Accounts for model inaccuracies such as acceleration and maneuvers
        q_p = float(self.config.process_noise_pos)
        q_v = float(self.config.process_noise_vel)
        Q = np.array([
            [q_p, 0.0, 0.0, 0.0],
            [0.0, q_p, 0.0, 0.0],
            [0.0, 0.0, q_v, 0.0],
            [0.0, 0.0, 0.0, q_v]
        ], dtype=np.float32)
        self.kf.processNoiseCov = Q

        # Measurement Noise Covariance R (2x2)
        # Accounts for centroid jitter from segmentation noise
        r = float(self.config.measurement_noise)
        R = np.array([
            [r,   0.0],
            [0.0, r  ]
        ], dtype=np.float32)
        self.kf.measurementNoiseCov = R

        # Initial Error Covariance P (4x4)
        p_val = float(self.config.initial_covariance)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * p_val

        # Initial State Vector X_0: Centroid initialized, initial velocities zero
        initial_state = np.array([
            [float(initial_x)],
            [float(initial_y)],
            [0.0],
            [0.0]
        ], dtype=np.float32)
        self.kf.statePost = initial_state
        self.kf.statePre = initial_state.copy()

        # Cached last state
        self.last_predicted: Tuple[float, float] = (initial_x, initial_y)
        self.last_corrected: Tuple[float, float] = (initial_x, initial_y)

    def predict(self) -> Tuple[float, float]:
        """
        Advance filter by one time-step (Prediction Stage).
        Propagates state estimate and covariance prior to incorporating new measurements.
        
        Returns:
            Tuple of predicted (x, y) coordinates.
        """
        predicted = self.kf.predict()
        px = float(predicted[0, 0])
        py = float(predicted[1, 0])
        self.last_predicted = (px, py)
        return self.last_predicted

    def correct(self, measured_x: float, measured_y: float) -> Tuple[float, float]:
        """
        Incorporate observed centroid measurement (Correction / Update Stage).
        Computes Kalman gain and updates posterior state estimate and covariance.
        
        Args:
            measured_x: Observed horizontal centroid coordinate.
            measured_y: Observed vertical centroid coordinate.
            
        Returns:
            Tuple of corrected (x, y) coordinates.
        """
        measurement = np.array([
            [float(measured_x)],
            [float(measured_y)]
        ], dtype=np.float32)
        
        corrected = self.kf.correct(measurement)
        cx = float(corrected[0, 0])
        cy = float(corrected[1, 0])
        self.last_corrected = (cx, cy)
        return self.last_corrected

    @property
    def position(self) -> Tuple[float, float]:
        """Current posterior (or prior) centroid position estimate (x, y)."""
        x = float(self.kf.statePost[0, 0])
        y = float(self.kf.statePost[1, 0])
        return (x, y)

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current posterior velocity estimate (vx, vy) in pixels/frame."""
        vx = float(self.kf.statePost[2, 0])
        vy = float(self.kf.statePost[3, 0])
        return (vx, vy)

    @property
    def state(self) -> np.ndarray:
        """Full 4x1 state vector [x, y, vx, vy]^T."""
        return self.kf.statePost.copy()
