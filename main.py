import cv2
import time
from config import settings
from core.database import FaceDatabase
from services.recognition import RecognitionService
from ui.display import VideoVisualizer


def main():
    # 1. Initialize Components
    try:
        database = FaceDatabase(settings.DB_PATH)
    except Exception as e:
        print(f"[CRITICAL] {e}")
        return

    ai_service = RecognitionService(database)
    visualizer = VideoVisualizer()

    # 2. Setup Video
    # cap = cv2.VideoCapture(str(settings.VIDEO_SOURCE))

    # Use raw camera instead for now
    cap = cv2.VideoCapture(0)

    # if not cap.isOpened():
    #     print(f"[ERROR] Cannot open video: {settings.VIDEO_SOURCE}")
    #     return

    # Calculate FPS delay
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay_ms = int(1000 / fps)

    # 3. Start AI Thread
    ai_service.start()

    print(f"[INFO] System Ready. Playing at {fps} FPS. Press 'q' to exit.")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            # Auto-loop video
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1

            # A. Send frame to AI (Background) - Only every Nth frame
            if frame_count % settings.SKIP_FRAMES == 0:
                ai_service.update_frame(frame)

            # B. Get latest result (List)
            faces = ai_service.get_latest_results()

            # C. Draw (Pass the list)
            visualizer.draw_results(frame, faces)

            cv2.imshow("Modular Face Recognition", frame)

            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break
    finally:
        # Graceful Shutdown
        ai_service.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete.")


if __name__ == "__main__":
    main()
