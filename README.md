
# Modular Face Recognition System

A robust, real-time face recognition application built with Python, OpenCV, and DeepFace. This project uses a modular architecture (MVC pattern) and multi-threading to ensure smooth video playback while performing heavy AI inference in the background.

## 🚀 Key Features

* **Real-Time Recognition:** Identifies faces from video files or webcam feeds.
* **Multi-Threaded Performance:** Decouples video rendering from AI processing for lag-free playback.
* **Modular Architecture:** Clean separation of concerns (Database, AI Service, UI) for scalability.
* **Configurable:** easy-to-adjust settings for thresholds, models, and input sources.
* **Robust Matching:** Uses VGG-Face with Euclidean distance (L2) validation.

## 📂 Project Structure

```text
face_rec_system/
├── config/
│   └── settings.py          # Configuration (Video path, Thresholds)
├── core/
│   ├── database.py          # Handles loading/querying the pickle DB
│   └── types.py             # Data Transfer Objects (FaceData)
├── services/
│   └── recognition.py       # Background AI Worker thread
├── ui/
│   └── display.py           # Drawing logic (Bounding boxes, Labels)
├── tools/
│   └── train_faces.py       # Script to generate encodings (optional loc)
├── main.py                  # Entry point
└── requirements.txt         # Dependency list
```

## 🛠️ Setup & Installation

### 1. Prerequisites
* **Python 3.10+** (Recommended version for DeepFace stability)
* **Anaconda** or **Miniconda** (Highly recommended for managing environments on Windows)

### 2. Clone the Repository
```bash
git clone [https://github.com/JosephAwuku33/face-rec-system.git]
cd face-rec-system
```

### 3. Create a new environment named 'deepface_env'
```bash 
conda create -n deepface_env python=3.10
```
### 4. Activate the environment
```bash
conda activate deepface_env
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

The project is controlled via `config/settings.py`. You do not need to edit the code logic to change parameters.

* **`VIDEO_SOURCE`**:
    * Set to `0` for Webcam.
    * Set to `"path/to/video.mp4"` for video files.

* **`MATCH_THRESHOLD`**: Controls how strict the matching is.
    * **0.8 - 0.9**: Strict (Low false positives, might miss some faces).
    * **1.0 - 1.1**: Balanced (Recommended for VGG-Face).
    * **1.2+**: Loose (Good for webcam lighting, but higher risk of wrong identity).

* **`SKIP_FRAMES`**:
    * Default is `30`. Increases performance by only running AI every N frames.

## 🏃 Usage

### Step 1: Build the Database
Ensure your reference images are organized in folders by name:
`dataset/known_faces/John_Doe/image.jpg`

Run the training tool:
```bash
python tools/train_faces.py
```

### Step 2: Run the System
Ensure your environment is active and run the main script
```bash
py main.py
```

### Controls
- Press **q** to quit the application


## 🐛 Troubleshooting

| Issue | Possible Cause | Solution |
| :--- | :--- | :--- |
| **Video Lags** | AI running too frequently | Increase `SKIP_FRAMES` in `settings.py` to `60`. |
| **Faces "Unknown"** | Threshold too strict | Increase `MATCH_THRESHOLD` to `1.2` or `1.3`. |
| **Import Error** | Wrong environment | Ensure `(deepface_env)` is active in your terminal. |
| **Video doesn't open** | Incorrect Path | Check `VIDEO_SOURCE` path. Use `/` slashes on Windows. |