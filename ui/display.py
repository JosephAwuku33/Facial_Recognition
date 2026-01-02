from typing import List
from core.types import FaceData
import cv2

class VideoVisualizer:
    def draw_results(self, frame, faces: List[FaceData]):
        # CHANGE 3: Iterate through the list
        for face in faces:
            name = face.name
            x, y, w, h = face.location
            dist = face.distance

            # Logic is the same, just inside a loop
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 35), (x + w, y), color, cv2.FILLED)
            text = f"{name} ({dist:.2f})"
            cv2.putText(
                frame,
                text,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                1,
            )
