"""
Vehicle Detection Module.
Performs background subtraction (MOG2 or KNN), Gaussian noise filtering,
morphological opening/closing, shadow suppression, and contour geometry filtering.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from config import DetectorConfig


@dataclass
class Detection:
    """Represents a single detected vehicle in a frame."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    centroid: Tuple[float, float]    # (cx, cy)
    area: float                      # Contour area in pixels
    contour: np.ndarray              # Raw contour points


class VehicleDetector:
    """
    Detects moving vehicles from video frames using adaptive background subtraction.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize detector with Gaussian blur, background subtractor, and morphology kernels.
        
        Args:
            config: Configuration parameters for detector.
        """
        self.config = config or DetectorConfig()

        # Initialize background subtractor
        if self.config.method.upper() == "KNN":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.config.history,
                dist2Threshold=self.config.var_threshold,
                detectShadows=self.config.detect_shadows
            )
        else:  # Default to MOG2
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.config.history,
                varThreshold=self.config.var_threshold,
                detectShadows=self.config.detect_shadows
            )

        # Pre-build morphological structuring elements
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, self.config.open_kernel
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, self.config.close_kernel
        )

        # Cache last generated foreground mask
        self.latest_mask: Optional[np.ndarray] = None
        self.latest_cleaned_mask: Optional[np.ndarray] = None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to suppress high-frequency camera sensor noise.
        
        Args:
            frame: Input BGR image frame.
            
        Returns:
            Preprocessed image.
        """
        if self.config.blur_kernel[0] > 0 and self.config.blur_kernel[1] > 0:
            return cv2.GaussianBlur(
                frame,
                self.config.blur_kernel,
                self.config.blur_sigma
            )
        return frame

    def extract_foreground_mask(self, preprocessed_frame: np.ndarray) -> np.ndarray:
        """
        Compute foreground mask and eliminate shadow pixels.
        
        Args:
            preprocessed_frame: Smoothed frame.
            
        Returns:
            Binary foreground mask where moving foreground = 255, background/shadow = 0.
        """
        # Apply background subtractor
        fg_mask = self.subtractor.apply(preprocessed_frame)
        self.latest_mask = fg_mask.copy()

        # Shadow suppression: When detectShadows=True, OpenCV sets shadows to 127.
        # Threshold at 200 eliminates shadows and preserves true foreground (255).
        if self.config.detect_shadows:
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        else:
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Morphological Opening: Erodes then dilates to remove isolated noisy speckles
        if self.config.open_iterations > 0:
            fg_mask = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_OPEN,
                self.open_kernel,
                iterations=self.config.open_iterations
            )

        # Morphological Closing: Dilates then erodes to fill interior holes & bridge contours
        if self.config.close_iterations > 0:
            fg_mask = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_CLOSE,
                self.close_kernel,
                iterations=self.config.close_iterations
            )

        self.latest_cleaned_mask = fg_mask
        return fg_mask

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles in frame, applying geometric and contour filtering.
        
        Args:
            frame: Input BGR image frame.
            
        Returns:
            List of valid Detection instances.
        """
        preprocessed = self.preprocess(frame)
        mask = self.extract_foreground_mask(preprocessed)

        # Find external contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))

            # Area filtering
            if area < self.config.min_area or area > self.config.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Dimensional filtering
            if w < self.config.min_width or h < self.config.min_height:
                continue

            aspect_ratio = float(w) / float(h)
            if aspect_ratio < self.config.min_aspect_ratio or aspect_ratio > self.config.max_aspect_ratio:
                continue

            # Centroid calculation via spatial moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx = float(x + w / 2.0)
                cy = float(y + h / 2.0)

            detections.append(Detection(
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area=area,
                contour=cnt
            ))

        return detections
