import pickle
import numpy as np
from pathlib import Path
from typing import Tuple
from config import settings

class FaceDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.encodings = []
        self.names = []
        self._load_data()

    def _load_data(self):
        """Internal method to load data from disk."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        
        print(f"[INFO] Loading database from {self.db_path.name}...")
        data = pickle.loads(self.db_path.read_bytes())
        self.encodings = np.array(data["encodings"])
        self.names = data["names"]

    def find_closest_match(self, vector: list) -> Tuple[str, float]:
        """Calculates Euclidean distance to find the identity."""
        if len(self.encodings) == 0:
            return "Unknown", 0.0

        # Vectorized calculation (Fast)
        distances = np.linalg.norm(self.encodings - vector, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist < settings.MATCH_THRESHOLD:
            return self.names[min_idx], min_dist
        
        return "Unknown", min_dist