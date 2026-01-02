import threading
import time
import cv2
from deepface import DeepFace
from typing import Optional
from core.database import FaceDatabase
from core.types import FaceData
from config import settings


class RecognitionService(threading.Thread):
    def __init__(self, database: FaceDatabase):
        super().__init__(daemon=True)  # Daemon kills thread when main app quits
        self.db = database
        self._current_frame = None
        self._latest_result: Optional[FaceData] = None
        self._running = True
        self._lock = threading.Lock()  # Thread safety

    def update_frame(self, frame):
        """Thread-safe way to update the frame."""
        with self._lock:
            self._current_frame = frame.copy()

    def get_latest_result(self) -> Optional[FaceData]:
        """Thread-safe way to get results."""
        return self._latest_result

    def stop(self):
        self._running = False

    def run(self):
        """The main loop running in the background thread."""
        print("[INFO] AI Service started...")
        while self._running:
            # 1. Get Frame safely
            frame_to_process = None
            with self._lock:
                if self._current_frame is not None:
                    frame_to_process = self._current_frame

            if frame_to_process is None:
                time.sleep(0.05)
                continue

            # 2. Process (Heavy Math)
            try:
                self._process_frame(frame_to_process)
            except Exception as e:
                # Log error but keep thread alive
                print(f"[WARNING] AI Error: {e}")

    def _process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = DeepFace.represent(
            img_path=img_rgb, model_name=settings.MODEL_NAME, enforce_detection=False
        )

        if len(results) > 0:
            data = results[0]
            name, dist = self.db.find_closest_match(data["embedding"])

            # Map DeepFace area to our Tuple format
            area = data["facial_area"]
            box = (area["x"], area["y"], area["w"], area["h"])

            self._latest_result = FaceData(name=name, location=box, distance=dist)
        else:
            self._latest_result = None
