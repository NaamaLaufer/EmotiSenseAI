"""
video_emotion.py
-----------------
ערוץ הוידאו — זיהוי רגש מהבעות פנים בזמן אמת.
משתמש ב-OpenCV לפתיחת המצלמה וב-DeepFace לניתוח הרגש.

שימוש:
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
    מזהה רגש מתמונת מצלמה בזמן אמת.

    DeepFace רץ כל analyze_every פריימים (לא כל פריים — כדי לא להאט).
    בין ריצות DeepFace מוחזרת התוצאה האחרונה.
    """

    def __init__(self, camera_index: int = 0, analyze_every: int = 8):
        """
        camera_index:  0 = מצלמה ראשית, 1 = מצלמה שנייה וכו׳
        analyze_every: הרץ DeepFace כל N פריימים
        """
        self.camera_index   = camera_index
        self.analyze_every  = analyze_every
        self.cap            = None
        self.frame_count    = 0

        # תוצאה אחרונה — מוחזרת בין ריצות DeepFace
        self.last_emotion = 'neutral'
        self.last_conf    = 0.0
        self.last_probs   = np.ones(7) / 7.0

    # ──────────────────────────────────────────────
    # פתיחה / סגירה
    # ──────────────────────────────────────────────
    def open(self) -> bool:
        """פותח את המצלמה. מחזיר True אם הצליח."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"❌ לא ניתן לפתוח מצלמה {self.camera_index}")
            return False
        print(f"✅ מצלמה {self.camera_index} פתוחה")
        return True

    def release(self):
        """משחרר את המצלמה."""
        if self.cap:
            self.cap.release()
            self.cap = None

    # ──────────────────────────────────────────────
    # קריאת פריים
    # ──────────────────────────────────────────────
    def read_frame(self):
        """
        קורא פריים אחד מהמצלמה.
        מחזיר: (ret: bool, frame: np.ndarray)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()

    # ──────────────────────────────────────────────
    # ניתוח רגש
    # ──────────────────────────────────────────────
    def analyze_frame(self, frame: np.ndarray) -> tuple:
        """
        מנתח פריים ומחזיר רגש, ביטחון, וקטור הסתברויות.
        DeepFace רץ רק כל analyze_every פריימים.

        Returns:
            (emotion_str, confidence_float, probs_array)
        """
        self.frame_count += 1

        if self.frame_count % self.analyze_every == 0:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                emotion_dict      = result[0]['emotion']
                self.last_emotion = result[0]['dominant_emotion']
                self.last_conf    = emotion_dict[self.last_emotion]
                self.last_probs   = deepface_to_probs(emotion_dict)
            except Exception:
                pass   # אם לא זוהו פנים — נשמרת התוצאה האחרונה

        return self.last_emotion, self.last_conf, self.last_probs.copy()

    # ──────────────────────────────────────────────
    # ציור על הפריים
    # ──────────────────────────────────────────────
    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        מוסיף תווית רגש על הפריים (BGR).
        מחזיר עותק של הפריים עם הכיתוב.
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
# הרצה עצמאית (בדיקה)
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