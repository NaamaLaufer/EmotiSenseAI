"""
video_emotion.py
-----------------
Video channel - real-time emotion recognition from facial expressions.
Uses OpenCV to open the camera and DeepFace to analyze the emotion.

Usage:
    detector = VideoEmotionDetector()
    detector.open()
    while True:
        ret, frame = detector.read_frame()
        if not ret: break
        emotion, conf, probs = detector.analyze_frame(frame)
    detector.release()
"""

import cv2
import numpy as np
from deepface import DeepFace

from fusion import EMOTION_LABELS, EMOTION_COLORS, deepface_to_probs


class VideoEmotionDetector:
    """
    Detects emotion from a camera image in real time.

    DeepFace runs every `analyze_every` frames (not every frame - to avoid slowdown).
    Between DeepFace runs, the last result is returned.
    """

    def __init__(self, camera_index: int = 0, analyze_every: int = 8,
                 detector_backend: str = 'opencv'):
        """
        camera_index:     0 = primary camera, 1 = second camera, etc.
        analyze_every:    run DeepFace every N frames
        detector_backend: face detector. 'retinaface' = accurate, 'ssd'/'opencv' = fast
        """
        self.camera_index     = camera_index
        self.analyze_every    = analyze_every
        self.detector_backend = detector_backend
        self.cap            = None
        self.frame_count    = 0

        # Last result - returned between DeepFace runs.
        self.last_emotion = 'neutral'
        self.last_conf    = 0.0
        self.last_probs   = np.ones(7) / 7.0

    def open(self) -> bool:
        """Open the camera. Returns True on success."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"❌ לא ניתן לפתוח מצלמה {self.camera_index}")
            return False
        print(f"✅ מצלמה {self.camera_index} פתוחה")
        return True

    def release(self):
        """Release the camera."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def read_frame(self):
        """
        Read a single frame from the camera.
        Returns: (ret: bool, frame: np.ndarray)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()

    def analyze_frame(self, frame: np.ndarray) -> tuple:
        """
        Analyze a frame and return emotion, confidence, and a probability vector.
        DeepFace only runs every `analyze_every` frames.

        Returns:
            (emotion_str, confidence_float, probs_array)
        """
        self.frame_count += 1

        if self.frame_count % self.analyze_every == 0:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    silent=True
                )
                emotion_dict      = result[0]['emotion']
                self.last_emotion = result[0]['dominant_emotion']
                self.last_conf    = emotion_dict[self.last_emotion]
                self.last_probs   = deepface_to_probs(emotion_dict)
            except Exception:
                pass   # no face detected - keep the last result

        return self.last_emotion, self.last_conf, self.last_probs.copy()

    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the emotion label onto the frame (BGR).
        Returns a copy of the frame with the caption.
        """
        color_hex = EMOTION_COLORS.get(self.last_emotion, '#FFFFFF')
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)

        display = frame.copy()
        label   = f"{self.last_emotion.upper()}  {self.last_conf:.1f}%"
        cv2.putText(display, label,
                    (12, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (b, g, r), 2,
                    cv2.LINE_AA)
        return display


# ─────────────────────────────────────────────────────────────
# Standalone run (for testing this channel alone)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = VideoEmotionDetector(analyze_every=8)

    if not detector.open():
        exit(1)

    print("מצלמה פעילה — לחץ Q לעצירה")

    while True:
        ret, frame = detector.read_frame()
        if not ret:
            break

        emotion, conf, probs = detector.analyze_frame(frame)
        display = detector.draw_on_frame(frame)

        cv2.imshow("Video Emotion", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.release()
    cv2.destroyAllWindows()
