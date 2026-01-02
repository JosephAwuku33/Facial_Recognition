from typing import List

class RecognitionService(threading.Thread):
    def __init__(self, database: FaceDatabase):
        super().__init__(daemon=True)
        self.db = database
        self._current_frame = None
        # CHANGE 1: Initialize as an empty list, not None
        self._latest_results: List[FaceData] = [] 
        self._running = True
        self._lock = threading.Lock()

    # ... (update_frame, stop, run methods remain the same) ...

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
            enforce_detection=False
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
                box = (area['x'], area['y'], area['w'], area['h'])
                
                # 4. Create Object and Append
                face_obj = FaceData(name=name, location=box, distance=dist)
                new_faces.append(face_obj)

        # Update the global list safely
        self._latest_results = new_faces