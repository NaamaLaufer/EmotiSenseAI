"""
audio_emotion.py
-----------------
זיהוי רגש מקול בזמן אמת.
מקליט 3 שניות מהמיקרופון, ממיר ל-Mel Spectrogram,
ומריץ את מודל ה-CNN2D שאומן על RAVDESS.
"""

import numpy as np
import librosa
import sounddevice as sd
import os
import tf_keras

from fusion import EMOTION_LABELS

SAMPLE_RATE = 22050
DURATION    = 3
N_MELS      = 128
MAX_FRAMES  = 130


class AudioEmotionDetector:

    def __init__(self, model_path: str):
        print("טוען מודל קול...")

        model = tf_keras.models.Sequential([
            tf_keras.layers.Conv2D(32, (3,3), padding='same', activation='relu',
                                   input_shape=(128, 130, 1)),
            tf_keras.layers.BatchNormalization(),
            tf_keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf_keras.layers.BatchNormalization(),
            tf_keras.layers.MaxPooling2D((2,2)),
            tf_keras.layers.Dropout(0.25),

            tf_keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf_keras.layers.BatchNormalization(),
            tf_keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf_keras.layers.BatchNormalization(),
            tf_keras.layers.MaxPooling2D((2,2)),
            tf_keras.layers.Dropout(0.25),

            tf_keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
            tf_keras.layers.BatchNormalization(),
            tf_keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
            tf_keras.layers.BatchNormalization(),
            tf_keras.layers.MaxPooling2D((2,2)),
            tf_keras.layers.Dropout(0.3),

            tf_keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
            tf_keras.layers.BatchNormalization(),
            tf_keras.layers.GlobalAveragePooling2D(),
            tf_keras.layers.Dropout(0.4),

            tf_keras.layers.Dense(256, activation='relu'),
            tf_keras.layers.BatchNormalization(),
            tf_keras.layers.Dropout(0.4),
            tf_keras.layers.Dense(128, activation='relu'),
            tf_keras.layers.Dropout(0.3),
            tf_keras.layers.Dense(7, activation='softmax')
        ])

        # טעינת משקלים
        data    = np.load('model_weights.npz', allow_pickle=True)
        weights = [data[f'w_{i}'] for i in range(len(data.files))]
        model.build((None, 128, 130, 1))
        model.set_weights(weights)

        self.model = model
        print("✅ מודל קול נטען")

    def record_audio(self, duration=DURATION, sr=SAMPLE_RATE):
        print(f"מקליט {duration} שניות...")
        audio = sd.rec(int(duration * sr), samplerate=sr,
                       channels=1, dtype='float32')
        sd.wait()
        print("הקלטה הסתיימה")
        return audio.flatten()

    def extract_spectrogram(self, audio, sr=SAMPLE_RATE):
        mel    = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        if mel_db.shape[1] < MAX_FRAMES:
            mel_db = np.pad(mel_db, ((0,0), (0, MAX_FRAMES - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :MAX_FRAMES]

        return mel_db

    def predict(self, audio):
        spec  = self.extract_spectrogram(audio)
        spec  = spec[np.newaxis, ..., np.newaxis]   # (1, 128, 130, 1)
        probs = self.model.predict(spec, verbose=0)[0]

        emotion_idx = int(np.argmax(probs))
        emotion     = EMOTION_LABELS[emotion_idx]
        confidence  = float(probs[emotion_idx]) * 100

        return emotion, confidence, probs

    def analyze(self):
        audio = self.record_audio()
        emotion, confidence, probs = self.predict(audio)
        print(f"קול --> {emotion} ({confidence:.1f}%)")
        return emotion, confidence, probs, audio


if __name__ == "__main__":
    if not os.path.exists('model_weights.npz'):
        print("לא נמצא model_weights.npz")
    else:
        detector = AudioEmotionDetector("model_weights.npz")
        emotion, confidence, probs, audio = detector.analyze()
        print(f"\nרגש:    {emotion}")
        print(f"ביטחון: {confidence:.1f}%")