from deepface import DeepFace
import pickle
import os
from pathlib import Path


# --- CONFIGURATION ---
DATASET_PATH = Path("dataset/known_faces")
ENCODINGS_FILE = Path("encodings.pickle")
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# DeepFace offers different models: "VGG-Face", "Facenet", "OpenFace", etc.
# "VGG-Face" is the default and is very balanced.
MODEL_NAME = "VGG-Face"


def train_model():
    print("[INFO] Quantifying faces using DeepFace...")

    known_encodings = []
    known_names = []

    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                image_path = os.path.join(root, file)
                name = os.path.basename(root)

                print(f"[INFO] Processing: {name} -> {file}")

                try:
                    # DeepFace.represent does the detection AND embedding in one step
                    # enforce_detection=False allows it to skip if it's unsure,
                    # but usually setting it to True is safer for the DB creation
                    # to ensure we only get good faces.
                    embeddings = DeepFace.represent(
                        img_path=image_path,
                        model_name=MODEL_NAME,
                        enforce_detection=True,
                    )

                    # DeepFace can return multiple faces. We assume 1 face per training image.
                    # The result is a list of dictionaries. We want the "embedding" key.
                    if len(embeddings) > 0:
                        embedding_vector = embeddings[0]["embedding"]
                        known_encodings.append(embedding_vector)
                        known_names.append(name)

                except ValueError:
                    # DeepFace raises ValueError if enforce_detection=True and no face is found
                    print(f"[WARNING] No face detected in {file}. Skipping.")
                except Exception as e:
                    print(f"[ERROR] Could not process {file}: {e}")

    print(f"[INFO] Serializing {len(known_encodings)} encodings...")
    data = {"encodings": known_encodings, "names": known_names}

    with open(ENCODINGS_FILE, "wb") as f:
        f.write(pickle.dumps(data))

    print(f"[INFO] Success! Encodings saved to {ENCODINGS_FILE}")


if __name__ == "__main__":
    train_model()
