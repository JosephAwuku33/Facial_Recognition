from typing import List
import threading
import cv2
import time
from deepface import DeepFace
from core.types import FaceData
from core.database import FaceDatabase
from config import settings


class RecognitionService(threading.Thread):
    def __init__(self, database: FaceDatabase):
        super().__init__(daemon=True)
        self.db = database
        self._current_frame = None
        # CHANGE 1: Initialize as an empty list, not None
        self._latest_results: List[FaceData] = []
        self._running = True
        self._lock = threading.Lock()

    def update_frame(self, frame):
        with self._lock:
            self._current_frame = frame

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

    def get_latest_results(self) -> List[FaceData]:
        """Returns a list of all faces found."""
        return self._latest_results

    def _process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # DeepFace returns a LIST of result dictionaries
        results = DeepFace.represent(
            img_path=img_rgb,
            model_name=settings.MODEL_NAME,
            # detector_backend='opencv', # or 'retinaface'
            enforce_detection=False,
        )

        new_faces = []

        # CHANGE 2: Loop through ALL results, not just results[0]
        if len(results) > 0:
            for face_data in results:
                # 1. Get embedding
                embedding = face_data["embedding"]

                # 2. Identify
                name, dist = self.db.find_closest_match(embedding)

                # 3. Get coordinates
                area = face_data["facial_area"]
                box = (area["x"], area["y"], area["w"], area["h"])

                # 4. Create Object and Append
                face_obj = FaceData(name=name, location=box, distance=dist)
                new_faces.append(face_obj)

        # Update the global list safely
        self._latest_results = new_faces
