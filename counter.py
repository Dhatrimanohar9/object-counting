"""
Virtual Counting Line and Bidirectional Vehicle Counter Module.
Implements a hysteresis-buffered virtual line to detect upward and downward crossings,
guaranteeing single-count integrity per vehicle track.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from config import CounterConfig
from tracker import TrackedVehicle


@dataclass
class CrossingEvent:
    """Record of a single verified line crossing event."""
    track_id: int
    direction: str          # 'UP' or 'DOWN'
    frame_idx: int
    centroid: Tuple[float, float]
    velocity_y: float


class VehicleCounter:
    """
    Manages bidirectional vehicle counting across a configurable horizontal virtual line.
    """

    def __init__(self, config: Optional[CounterConfig] = None):
        self.config = config or CounterConfig()
        self.count_up: int = 0
        self.count_down: int = 0
        self.counted_ids: set = set()
        self.crossing_events: List[CrossingEvent] = []

    @property
    def total_count(self) -> int:
        """Total vehicle count is the exact sum of directional counts."""
        return self.count_up + self.count_down

    def get_line_y(self, frame_height: int) -> int:
        """Calculate line Y coordinate in pixels based on frame height."""
        return int(frame_height * self.config.line_y_ratio)

    def reset(self):
        """Reset all counters and crossing event history."""
        self.count_up = 0
        self.count_down = 0
        self.counted_ids.clear()
        self.crossing_events.clear()

    def update(self, tracks: List[TrackedVehicle], frame_height: int, frame_idx: int = 0) -> List[CrossingEvent]:
        """
        Evaluate line crossing conditions for all active tracks.
        
        Args:
            tracks: List of active TrackedVehicle objects.
            frame_height: Height of the video frame in pixels.
            frame_idx: Current frame index.
            
        Returns:
            List of new CrossingEvent instances recorded in this frame.
        """
        line_y = self.get_line_y(frame_height)
        buffer = self.config.buffer_zone
        upper_bound = line_y - buffer
        lower_bound = line_y + buffer

        new_events: List[CrossingEvent] = []

        for track in tracks:
            # Enforce single-count guarantee per vehicle ID
            if track.track_id in self.counted_ids or track.counted:
                continue

            # Need at least 2 historical positions to confirm trajectory
            if len(track.history) < 2:
                continue

            current_y = track.centroid[1]
            previous_y = track.history[-2][1]
            _, vy = track.kalman.velocity

            # Check for Downward Crossing: Moving from above line (y < upper_bound) to below line (y >= line_y)
            # Look at earlier history to ensure vehicle originated definitively above
            originated_above = any(pos[1] < upper_bound for pos in track.history)
            originated_below = any(pos[1] > lower_bound for pos in track.history)

            if originated_above and current_y >= line_y and (current_y - previous_y >= 0 or vy >= 0):
                self.count_down += 1
                self.counted_ids.add(track.track_id)
                track.counted = True
                track.direction = "DOWN"
                track.counted_frame = frame_idx

                event = CrossingEvent(
                    track_id=track.track_id,
                    direction="DOWN",
                    frame_idx=frame_idx,
                    centroid=track.centroid,
                    velocity_y=vy
                )
                self.crossing_events.append(event)
                new_events.append(event)

            # Check for Upward Crossing: Moving from below line (y > lower_bound) to above line (y <= line_y)
            elif originated_below and current_y <= line_y and (current_y - previous_y <= 0 or vy <= 0):
                self.count_up += 1
                self.counted_ids.add(track.track_id)
                track.counted = True
                track.direction = "UP"
                track.counted_frame = frame_idx

                event = CrossingEvent(
                    track_id=track.track_id,
                    direction="UP",
                    frame_idx=frame_idx,
                    centroid=track.centroid,
                    velocity_y=vy
                )
                self.crossing_events.append(event)
                new_events.append(event)

        return new_events

    def get_statistics(self) -> Dict[str, int]:
        """Return snapshot dictionary of counting statistics."""
        return {
            "upward_count": self.count_up,
            "downward_count": self.count_down,
            "total_count": self.total_count
        }
