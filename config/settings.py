from pathlib import Path

# Use pathlib for robust cross-platform paths
BASE_DIR = Path(__file__).resolve().parent.parent

# File Paths
VIDEO_SOURCE = BASE_DIR / "test_videos" / "second_test.mp4"
DB_PATH = BASE_DIR / "encodings.pickle"

# AI Constants
MATCH_THRESHOLD = 1.0
MODEL_NAME = "VGG-Face"
SKIP_FRAMES = 30  # Processing frequency