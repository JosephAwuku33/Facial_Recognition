import cv2
from core.types import FaceData


class VideoVisualizer:
    def draw_result(self, frame, result: FaceData):
        if not result:
            return

        name = result.name
        x, y, w, h = result.location

        # Color: Green for match, Red for unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw Box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw Label
        cv2.rectangle(frame, (x, y - 35), (x + w, y), color, cv2.FILLED)
        text = f"{name} ({result.distance:.2f})"
        cv2.putText(
            frame,
            text,
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1,
        )
