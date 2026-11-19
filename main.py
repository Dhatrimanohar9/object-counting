"""
Main Application: Intelligent Vehicle Detection, Tracking, and Bidirectional Counting System.
Coordinates video capture, background subtraction detection, multi-object tracking via
discrete-time Kalman Filters, virtual line crossing analysis, real-time HUD dashboard,
and video rendering.

Controls:
    Q / ESC : Quit application
    P / SPACE : Pause / Resume
    R : Reset counting statistics
    S : Save current frame screenshot
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple

from config import SystemConfig, SCREENSHOTS_DIR, PROCESSED_VIDEOS_DIR
from detector import VehicleDetector
from tracker import VehicleTracker, TrackedVehicle
from counter import VehicleCounter


class VehicleCountingSystem:
    """
    Orchestrates detection, Kalman tracking, line crossing detection, and visualization.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.detector = VehicleDetector(self.config.detector)
        self.tracker = VehicleTracker(self.config.tracker, self.config.kalman)
        self.counter = VehicleCounter(self.config.counter)

        # Runtime states
        self.is_paused: bool = False
        self.frame_index: int = 0
        self.prev_time: float = time.time()
        self.fps: float = 0.0

    def reset(self):
        """Reset tracker, Kalman states, and directional counters."""
        self.tracker.reset()
        self.counter.reset()
        print("System counters and tracker states have been reset.")

    def save_screenshot(self, frame: np.ndarray) -> str:
        """Capture and save high-resolution screenshot to outputs directory."""
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"[Screenshot] Saved: {filepath}")
        return filepath

    def draw_hud(self, frame: np.ndarray, active_tracks_count: int) -> np.ndarray:
        """
        Render semi-transparent telemetry HUD dashboard overlay.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Top dashboard panel dimensions
        panel_w = 420
        panel_h = 160
        margin = 15

        # Draw semi-transparent dark rounded panel
        cv2.rectangle(
            overlay,
            (margin, margin),
            (margin + panel_w, margin + panel_h),
            self.config.ui.COLOR_HUD_BG,
            -1
        )
        cv2.rectangle(
            overlay,
            (margin, margin),
            (margin + panel_w, margin + panel_h),
            (70, 70, 80),
            1
        )

        # Blend overlay with original frame (alpha blending)
        alpha = 0.78
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Header Title
        title = "INTELLIGENT VEHICLE MONITORING"
        cv2.putText(
            frame, title,
            (margin + 15, margin + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 230, 255), 2, cv2.LINE_AA
        )
        cv2.line(
            frame,
            (margin + 15, margin + 36),
            (margin + panel_w - 15, margin + 36),
            (90, 95, 105), 1
        )

        # Telemetry counts
        stats = self.counter.get_statistics()
        up_cnt = stats["upward_count"]
        down_cnt = stats["downward_count"]
        tot_cnt = stats["total_count"]

        # Total Count
        cv2.putText(
            frame, f"TOTAL COUNT   : {tot_cnt}",
            (margin + 15, margin + 64),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        # Upward Count (Green)
        cv2.putText(
            frame, f"UPWARD (NORTH): {up_cnt}",
            (margin + 15, margin + 92),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60,
            self.config.ui.COLOR_UP, 2, cv2.LINE_AA
        )

        # Downward Count (Orange)
        cv2.putText(
            frame, f"DOWNWARD(SOUTH): {down_cnt}",
            (margin + 15, margin + 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60,
            self.config.ui.COLOR_DOWN, 2, cv2.LINE_AA
        )

        # System Metrics & Live Status (bottom row of panel)
        status_text = "PAUSED" if self.is_paused else "RUNNING"
        status_color = (0, 165, 255) if self.is_paused else (0, 255, 100)
        metrics_str = f"FPS: {self.fps:4.1f} | Active: {active_tracks_count} | [{status_text}]"
        cv2.putText(
            frame, metrics_str,
            (margin + 15, margin + 146),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            status_color, 1, cv2.LINE_AA
        )

        # Subtle key controls guide at bottom-right of frame
        ctrl_w = 340
        ctrl_h = 32
        cx1 = w - ctrl_w - 15
        cy1 = h - ctrl_h - 15
        cv2.rectangle(frame, (cx1, cy1), (cx1 + ctrl_w, cy1 + ctrl_h), (20, 20, 25), -1)
        cv2.rectangle(frame, (cx1, cy1), (cx1 + ctrl_w, cy1 + ctrl_h), (60, 60, 70), 1)
        cv2.putText(
            frame, "[P] Pause  |  [R] Reset  |  [S] Save  |  [Q] Quit",
            (cx1 + 10, cy1 + 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (190, 195, 200), 1, cv2.LINE_AA
        )

        return frame

    def draw_virtual_line(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw configurable virtual counting line with hysteresis boundary markers and direction arrows.
        """
        h, w = frame.shape[:2]
        line_y = self.counter.get_line_y(h)
        buf = self.config.counter.buffer_zone
        upper_y = line_y - buf
        lower_y = line_y + buf

        # Hysteresis buffer region boundaries (dashed / thin lines)
        cv2.line(frame, (0, upper_y), (w, upper_y), self.config.ui.COLOR_BUFFER, 1, cv2.LINE_AA)
        cv2.line(frame, (0, lower_y), (w, lower_y), self.config.ui.COLOR_BUFFER, 1, cv2.LINE_AA)

        # Primary Virtual Counting Line (Bright Yellow)
        cv2.line(frame, (0, line_y), (w, line_y), self.config.ui.COLOR_LINE, self.config.counter.line_thickness, cv2.LINE_AA)

        # Line label & Direction indicators
        line_label = f"-- VIRTUAL COUNTING LINE (Y={line_y}) --"
        cv2.putText(
            frame, line_label,
            (w // 2 - 190, line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            self.config.ui.COLOR_LINE, 1, cv2.LINE_AA
        )

        # Directional arrows on road carriageways
        # Left side: UPWARD arrow (North)
        cv2.arrowedLine(frame, (380, line_y + 45), (380, line_y - 45), self.config.ui.COLOR_UP, 2, tipLength=0.3)
        cv2.putText(frame, "UP", (395, line_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.ui.COLOR_UP, 2)

        # Right side: DOWNWARD arrow (South)
        cv2.arrowedLine(frame, (900, line_y - 45), (900, line_y + 45), self.config.ui.COLOR_DOWN, 2, tipLength=0.3)
        cv2.putText(frame, "DOWN", (915, line_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.ui.COLOR_DOWN, 2)

        return frame

    def draw_tracks(self, frame: np.ndarray, tracks: List[TrackedVehicle]) -> np.ndarray:
        """
        Render vehicle bounding boxes, persistent IDs, centroid trails, and Kalman filter predictions.
        """
        for t in tracks:
            x, y, w, h = t.bbox
            cx, cy = int(t.centroid[0]), int(t.centroid[1])

            # Determine color by vehicle direction/status
            if t.direction == "UP":
                color = self.config.ui.COLOR_UP
            elif t.direction == "DOWN":
                color = self.config.ui.COLOR_DOWN
            else:
                color = (255, 180, 0)  # Default light blue for pending crossing

            # 1. Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 2. Vehicle ID & Status Tag Banner
            id_text = f"ID: {t.track_id}"
            if t.direction:
                id_text += f" [{t.direction}]"
            elif t.counted:
                id_text += " [DONE]"

            # Background pill for ID text
            (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x, max(0, y - 20)), (x + tw + 8, y), color, -1)
            cv2.putText(
                frame, id_text,
                (x + 4, max(14, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (10, 10, 10), 1, cv2.LINE_AA
            )

            # 3. Motion Trajectory Trail (fading historic centroid positions)
            hist = list(t.history)
            for i in range(1, len(hist)):
                pt1 = (int(hist[i - 1][0]), int(hist[i - 1][1]))
                pt2 = (int(hist[i][0]), int(hist[i][1]))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

            # 4. Measured Centroid Point (Red Circle)
            cv2.circle(frame, (cx, cy), 4, self.config.ui.COLOR_CENTROID, -1)

            # 5. Kalman Filter Predicted Position (Cyan Crosshair)
            pred_x, pred_y = int(t.predicted_pos[0]), int(t.predicted_pos[1])
            cv2.drawMarker(
                frame, (pred_x, pred_y),
                self.config.ui.COLOR_PRED,
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=1
            )

        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Execute full pipeline for a single frame: Detect -> Track (Kalman) -> Count -> Render.
        """
        # Step 1: Detect moving vehicles
        detections = self.detector.detect(frame)

        # Step 2: Multi-object data association with Kalman Filter predictions/updates
        tracks = self.tracker.update(detections)

        # Step 3: Virtual counting line traversal evaluation
        h, _ = frame.shape[:2]
        self.counter.update(tracks, frame_height=h, frame_idx=self.frame_index)

        # Step 4: Render visualizations
        viz = frame.copy()
        viz = self.draw_virtual_line(viz)
        viz = self.draw_tracks(viz, tracks)
        viz = self.draw_hud(viz, active_tracks_count=len(tracks))

        return viz

    def run(
        self,
        video_source: Optional[str] = None,
        headless: bool = False,
        save_video: bool = True,
        max_frames: Optional[int] = None
    ) -> dict:
        """
        Run continuous video processing loop.
        
        Args:
            video_source: Video file path or webcam index.
            headless: If True, suppress cv2.imshow windows (ideal for automated test / server).
            save_video: If True, records processed video output to disk.
            max_frames: Optional maximum frame limit for testing.
            
        Returns:
            Dictionary containing final counting statistics.
        """
        src = video_source or self.config.video_path

        # Handle webcam vs file
        if isinstance(src, str):
            if src.isdigit():
                src = int(src)
            elif not os.path.exists(src):
                # Search candidate locations
                from config import BASE_DIR
                candidates = [
                    os.path.join(BASE_DIR, src),
                    os.path.join(os.path.dirname(BASE_DIR), src),
                    src.replace("Object_Counting/", ""),
                    os.path.join(BASE_DIR, src.replace("Object_Counting/", ""))
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        src = cand
                        break

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise FileNotFoundError(f"Error: Unable to open video source '{src}'. Please verify path.")

        # Capture stream parameters
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        fps_in = fps_in if (fps_in > 0 and not np.isnan(fps_in)) else 30.0

        print(f"\n[System Started]")
        print(f" Source Video : {src}")
        print(f" Resolution   : {width}x{height}")
        print(f" Source FPS   : {fps_in:.1f}")
        print(f" Headless Mode: {headless}")

        writer = None
        output_video_path = None
        if save_video:
            os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = os.path.join(PROCESSED_VIDEOS_DIR, f"vehicle_counting_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps_in, (width, height))
            print(f" Recording to : {output_video_path}")

        window_name = "Intelligent Vehicle Detection, Tracking & Counting System"
        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(1280, width), min(720, height))

        first_screenshot_saved = False

        try:
            while True:
                # Handle paused state
                if self.is_paused:
                    if not headless:
                        key = cv2.waitKey(30) & 0xFF
                        if key in (ord('q'), ord('Q'), 27):
                            break
                        elif key in (ord('p'), ord('P'), ord(' ')):
                            self.is_paused = False
                            print("Video resumed.")
                        elif key in (ord('r'), ord('R')):
                            self.reset()
                        elif key in (ord('s'), ord('S')):
                            self.save_screenshot(viz_frame)
                    else:
                        time.sleep(0.05)
                    continue

                ret, frame = cap.read()
                if not ret:
                    print("Reached end of video stream.")
                    break

                self.frame_index += 1

                # Measure processing FPS
                curr_time = time.time()
                dt = curr_time - self.prev_time
                self.prev_time = curr_time
                self.fps = (1.0 / dt) if dt > 0 else fps_in

                # Process current frame
                viz_frame = self.process_frame(frame)

                # Write to output video
                if writer is not None:
                    writer.write(viz_frame)

                # Automatically capture mid-run screenshot for academic report
                if not first_screenshot_saved and self.counter.total_count >= 2:
                    self.save_screenshot(viz_frame)
                    first_screenshot_saved = True

                # Display frame
                if not headless:
                    cv2.imshow(window_name, viz_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        print("User initiated exit.")
                        break
                    elif key in (ord('p'), ord('P'), ord(' ')):
                        self.is_paused = True
                        print("Video paused. Press 'P' to resume.")
                    elif key in (ord('r'), ord('R')):
                        self.reset()
                    elif key in (ord('s'), ord('S')):
                        self.save_screenshot(viz_frame)

                if max_frames and self.frame_index >= max_frames:
                    print(f"Reached max frame limit ({max_frames}).")
                    break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
                print(f"[Video Saved] Final output written to: {output_video_path}")
            if not headless:
                cv2.destroyAllWindows()

        stats = self.counter.get_statistics()
        print("\n" + "=" * 50)
        print(" FINAL VEHICLE COUNTING SUMMARY")
        print("=" * 50)
        print(f" Upward Count   (North) : {stats['upward_count']}")
        print(f" Downward Count (South) : {stats['downward_count']}")
        print(f" Total Count            : {stats['total_count']}")
        print(f" Total Frames Processed : {self.frame_index}")
        print("=" * 50 + "\n")

        stats["output_video_path"] = output_video_path
        return stats


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Intelligent Vehicle Detection, Tracking, and Bidirectional Counting System"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input video file or webcam index (default: videos/traffic_video.mp4)"
    )
    parser.add_argument(
        "--method", type=str, default="MOG2", choices=["MOG2", "KNN"],
        help="Background subtraction method (MOG2 or KNN)"
    )
    parser.add_argument(
        "--line-pos", type=float, default=0.55,
        help="Normalized Y position of the virtual counting line (0.1 to 0.9, default 0.55)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run in headless mode without graphical UI display"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Disable recording processed output video"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to process"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch graphical configuration launcher (Tkinter)"
    )

    args = parser.parse_args()

    # Optional GUI Launcher mode
    if args.gui:
        try:
            import gui
            gui.launch_gui()
            return
        except Exception as e:
            print(f"Failed to launch GUI ({e}), falling back to CLI execution.")

    # Configure system
    config = SystemConfig()
    config.detector.method = args.method
    config.counter.line_y_ratio = max(0.1, min(0.9, args.line_pos))
    if args.video:
        config.video_path = args.video

    system = VehicleCountingSystem(config)
    system.run(
        video_source=config.video_path,
        headless=args.no_display,
        save_video=not args.no_save,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
