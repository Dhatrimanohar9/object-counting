"""
Synthetic Traffic Video Generator.
Synthesizes a realistic 720p 30fps traffic surveillance video featuring a 4-lane highway
with bidirectional traffic, distinct vehicle types, realistic drop shadows, lane markings,
and controlled ground truth for automated evaluation.
"""

import os
import cv2
import numpy as np
from typing import List, Dict


class VehicleSprite:
    """Represents a simulated vehicle moving along a highway lane."""
    def __init__(self, v_type: str, color_bgr: tuple, lane: int, y_start: float, speed_y: float, start_frame: int):
        self.v_type = v_type
        self.color = color_bgr
        self.lane = lane
        self.y = y_start
        self.speed_y = speed_y  # Negative for UP, Positive for DOWN
        self.start_frame = start_frame
        
        # Dimensions based on vehicle category
        if v_type == "truck":
            self.w, self.h = 80, 190
        elif v_type == "suv":
            self.w, self.h = 70, 140
        else:  # sedan
            self.w, self.h = 60, 120

    def get_x_for_lane(self, lane: int) -> int:
        """Lane centers on 1280x720 canvas."""
        # Lanes 0, 1: Upward traffic (left carriageway)
        # Lanes 2, 3: Downward traffic (right carriageway)
        lane_centers = [430, 530, 750, 850]
        return lane_centers[lane]

    def update(self, current_frame: int) -> bool:
        """Advance vehicle position. Returns True if active on screen."""
        if current_frame < self.start_frame:
            return False
        self.y += self.speed_y
        # Active if within vertical range
        return -250 <= self.y <= 950

    def draw(self, frame: np.ndarray, current_frame: int):
        """Render vehicle sprite with shadow, body, windows, and lights."""
        if current_frame < self.start_frame or not (-220 <= self.y <= 920):
            return

        cx = self.get_x_for_lane(self.lane)
        cy = int(self.y)
        w, h = self.w, self.h
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h

        # 1. Cast realistic soft shadow beneath vehicle (offset right and down)
        shadow_offset_x = 12
        shadow_offset_y = 10
        sx1 = max(0, x1 + shadow_offset_x)
        sy1 = max(0, y1 + shadow_offset_y)
        sx2 = min(frame.shape[1], x2 + shadow_offset_x + 6)
        sy2 = min(frame.shape[0], y2 + shadow_offset_y + 6)
        if sx2 > sx1 and sy2 > sy1:
            sub = frame[sy1:sy2, sx1:sx2].astype(np.float32) * 0.55
            frame[sy1:sy2, sx1:sx2] = sub.astype(np.uint8)

        # Ensure within image bounds
        if x1 < 0 or x2 > frame.shape[1] or y1 < -100 or y2 > frame.shape[0] + 100:
            return

        # 2. Main vehicle chassis (rounded rectangle)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
        # Bevel highlight / border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (max(0, self.color[0] - 40), max(0, self.color[1] - 40), max(0, self.color[2] - 40)), 2)

        # 3. Windshield & Rear Window (tinted dark gray)
        windshield_color = (40, 40, 45)
        roof_inset_x = 6
        if self.speed_y > 0:  # Moving DOWN: windshield is at bottom (front), rear at top
            front_y1 = y2 - int(h * 0.32)
            front_y2 = y2 - int(h * 0.16)
            rear_y1 = y1 + int(h * 0.16)
            rear_y2 = y1 + int(h * 0.28)
            # Headlights (front)
            cv2.circle(frame, (x1 + 8, y2 - 4), 5, (220, 255, 255), -1)
            cv2.circle(frame, (x2 - 8, y2 - 4), 5, (220, 255, 255), -1)
            # Taillights (rear)
            cv2.circle(frame, (x1 + 8, y1 + 4), 4, (0, 0, 220), -1)
            cv2.circle(frame, (x2 - 8, y1 + 4), 4, (0, 0, 220), -1)
        else:  # Moving UP: windshield is at top (front), rear at bottom
            front_y1 = y1 + int(h * 0.16)
            front_y2 = y1 + int(h * 0.32)
            rear_y1 = y2 - int(h * 0.28)
            rear_y2 = y2 - int(h * 0.16)
            # Headlights (front)
            cv2.circle(frame, (x1 + 8, y1 + 4), 5, (220, 255, 255), -1)
            cv2.circle(frame, (x2 - 8, y1 + 4), 5, (220, 255, 255), -1)
            # Taillights (rear)
            cv2.circle(frame, (x1 + 8, y2 - 4), 4, (0, 0, 220), -1)
            cv2.circle(frame, (x2 - 8, y2 - 4), 4, (0, 0, 220), -1)

        cv2.rectangle(frame, (x1 + roof_inset_x, front_y1), (x2 - roof_inset_x, front_y2), windshield_color, -1)
        cv2.rectangle(frame, (x1 + roof_inset_x, rear_y1), (x2 - roof_inset_x, rear_y2), windshield_color, -1)

        # 4. Roof panel
        min_roof_y = min(front_y2, rear_y2)
        max_roof_y = max(front_y1, rear_y1)
        if max_roof_y > min_roof_y:
            roof_color = (min(255, self.color[0] + 25), min(255, self.color[1] + 25), min(255, self.color[2] + 25))
            cv2.rectangle(frame, (x1 + roof_inset_x, min_roof_y), (x2 - roof_inset_x, max_roof_y), roof_color, -1)


def generate_static_background(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create realistic highway road scene."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Grassy road shoulders
    bg[:, :320] = (45, 95, 40)    # Left roadside foliage
    bg[:, 960:] = (45, 95, 40)    # Right roadside foliage

    # 2. Highway asphalt surface
    road_color = (65, 68, 70)
    bg[:, 320:960] = road_color

    # Add subtle asphalt grain texture
    noise = np.random.normal(0, 3, (height, 640, 3)).astype(np.int16)
    asphalt_noisy = np.clip(bg[:, 320:960].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    bg[:, 320:960] = asphalt_noisy

    # 3. Outer solid white road edge lines
    cv2.line(bg, (340, 0), (340, height), (220, 220, 220), 4)
    cv2.line(bg, (940, 0), (940, height), (220, 220, 220), 4)

    # 4. Central median barrier (double yellow line with paved median)
    cv2.rectangle(bg, (630, 0), (650, height), (50, 52, 55), -1)
    cv2.line(bg, (632, 0), (632, height), (30, 200, 230), 3)  # Yellow 1
    cv2.line(bg, (648, 0), (648, height), (30, 200, 230), 3)  # Yellow 2

    return bg


def generate_traffic_video(output_path: str, duration_sec: int = 25, fps: int = 30):
    """
    Generate synthetic traffic surveillance video with bidirectional vehicle flow.
    """
    width, height = 1280, 720
    total_frames = duration_sec * fps
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    base_bg = generate_static_background(width, height)

    # Vehicle schedule: (v_type, color_bgr, lane, y_start, speed_y, start_frame)
    # Lanes 0, 1 move UP (speed_y negative: -6 to -10)
    # Lanes 2, 3 move DOWN (speed_y positive: +6 to +10)
    schedule = [
        # Upward vehicles (Lanes 0, 1) -> 6 vehicles
        VehicleSprite("sedan", (190, 190, 195), 0, 800, -7.0, 10),      # Silver Sedan
        VehicleSprite("suv",   (30, 30, 180),   1, 820, -6.5, 50),      # Crimson Red SUV
        VehicleSprite("truck", (140, 100, 40),  0, 860, -5.5, 130),     # Blue Utility Truck
        VehicleSprite("sedan", (30, 160, 40),   1, 800, -8.0, 220),     # Dark Green Sedan
        VehicleSprite("suv",   (180, 180, 180), 0, 820, -7.2, 330),     # White SUV
        VehicleSprite("sedan", (35, 80, 175),   1, 800, -8.5, 450),     # Orange Sedan

        # Downward vehicles (Lanes 2, 3) -> 6 vehicles
        VehicleSprite("suv",   (40, 40, 40),    2, -150, 7.5, 20),      # Black SUV
        VehicleSprite("sedan", (170, 70, 40),   3, -130, 8.2, 80),      # Deep Blue Sedan
        VehicleSprite("truck", (200, 200, 200), 2, -220, 6.0, 170),     # White Freight Truck
        VehicleSprite("sedan", (40, 130, 190),  3, -130, 7.8, 260),     # Golden Bronze Sedan
        VehicleSprite("suv",   (60, 60, 160),   2, -150, 7.0, 380),     # Dark Maroon SUV
        VehicleSprite("sedan", (160, 160, 160), 3, -130, 8.8, 480),     # Light Gray Sedan
    ]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    dash_length = 35
    dash_gap = 25
    cycle = dash_length + dash_gap

    print(f"Generating synthetic traffic video: {output_path} ({total_frames} frames)...")

    for frame_idx in range(total_frames):
        frame = base_bg.copy()

        # Draw animated dashed white lane lines to indicate motion
        # Left carriageway dashed line between lane 0 and lane 1 (x = 480)
        # Right carriageway dashed line between lane 2 and lane 3 (x = 800)
        for y_offset in range(0, height + cycle, cycle):
            # Left lane markings
            cv2.line(frame, (480, y_offset), (480, min(height, y_offset + dash_length)), (230, 230, 230), 2)
            # Right lane markings
            cv2.line(frame, (800, y_offset), (800, min(height, y_offset + dash_length)), (230, 230, 230), 2)

        # Update and render vehicles
        for v in schedule:
            v.update(frame_idx)
            v.draw(frame, frame_idx)

        # Add subtle random camera sensor noise (realistic surveillance footage)
        noise = np.random.normal(0, 1.2, (height, width, 3)).astype(np.int16)
        frame_noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        writer.write(frame_noisy)

    writer.release()
    print(f"Video generated successfully! Saved to {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    out_video = os.path.join(os.path.dirname(__file__), "videos", "traffic_video.mp4")
    generate_traffic_video(out_video, duration_sec=25, fps=30)
