"""
Configuration Module for Intelligent Vehicle Detection, Tracking, and Counting System.
Contains all tunable parameters, model hyperparameters, video paths, and UI settings.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SCREENSHOTS_DIR = os.path.join(OUTPUTS_DIR, "screenshots")
PROCESSED_VIDEOS_DIR = os.path.join(OUTPUTS_DIR, "processed_videos")

# Ensure directories exist
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

DEFAULT_VIDEO_PATH = os.path.join(VIDEOS_DIR, "traffic_video.mp4")


@dataclass
class DetectorConfig:
    """Detection settings for Background Subtraction."""
    method: str = "MOG2"  # 'MOG2' or 'KNN'
    history: int = 500
    var_threshold: float = 40.0
    detect_shadows: bool = True
    shadow_value: int = 127  # MOG2 default shadow pixel intensity
    
    # Preprocessing
    blur_kernel: Tuple[int, int] = (5, 5)
    blur_sigma: float = 1.0
    
    # Morphological transformations
    open_kernel: Tuple[int, int] = (3, 3)
    open_iterations: int = 1
    close_kernel: Tuple[int, int] = (7, 7)
    close_iterations: int = 2
    
    # Contour filtering
    min_area: int = 600       # Minimum contour area in pixels
    max_area: int = 60000     # Maximum contour area in pixels
    min_width: int = 20
    min_height: int = 20
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 4.0


@dataclass
class KalmanConfig:
    """Hyperparameters for discrete-time Kalman Filter."""
    dt: float = 1.0  # Frame delta time normalized to 1 frame
    
    # Process noise covariance parameters (acceleration / dynamic uncertainty)
    process_noise_pos: float = 0.1
    process_noise_vel: float = 0.5
    
    # Measurement noise covariance (centroid extraction uncertainty)
    measurement_noise: float = 4.0
    
    # Initial error covariance
    initial_covariance: float = 10.0


@dataclass
class TrackerConfig:
    """Parameters for Multi-Object Association & Lifecycle Management."""
    max_distance: float = 75.0      # Maximum Euclidean pixel distance for association
    max_disappeared: int = 15      # Deregister object after this many missing frames
    max_history: int = 30          # Number of past centroid locations to retain for trail


@dataclass
class CounterConfig:
    """Parameters for Virtual Counting Line and Directional Crossing."""
    # Normalized vertical position of the counting line (0.0 = top, 1.0 = bottom)
    line_y_ratio: float = 0.55
    
    # Hysteresis buffer in pixels to prevent oscillation/flicker at line boundary
    buffer_zone: int = 15
    
    # Visual line thickness
    line_thickness: int = 3


@dataclass
class UIConfig:
    """Display colors (BGR) and visual styling."""
    # Colors
    COLOR_UP: Tuple[int, int, int] = (0, 230, 0)          # Vibrant Green
    COLOR_DOWN: Tuple[int, int, int] = (0, 140, 255)      # Orange
    COLOR_LINE: Tuple[int, int, int] = (0, 255, 255)      # Bright Yellow
    COLOR_BUFFER: Tuple[int, int, int] = (100, 100, 100)  # Gray buffer boundary
    COLOR_PRED: Tuple[int, int, int] = (255, 255, 0)      # Cyan for Kalman prediction
    COLOR_CENTROID: Tuple[int, int, int] = (0, 0, 255)    # Red dot for measured centroid
    COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)    # White
    COLOR_HUD_BG: Tuple[int, int, int] = (20, 20, 25)     # Dark HUD panel
    
    # Text font
    FONT: int = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 0.6
    FONT_THICKNESS: int = 2


@dataclass
class SystemConfig:
    """Master configuration aggregate."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    counter: CounterConfig = field(default_factory=CounterConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    video_path: str = DEFAULT_VIDEO_PATH
    save_output: bool = True
