"""
audio_emotion.py - Voice channel (emotion from tone of speech)
==============================================================
Detects emotion from *how* you speak (tone, energy, rhythm) - not what you say.

How it works:
1. Convert the audio waveform into a "mel-spectrogram" - an image showing the
   intensity of frequencies over time.
2. A convolutional neural network (CNN) analyzes that image and detects the emotion.
3. VAD (Voice Activity Detection) filters silence - with no speech it returns
   neutral at confidence 0.
"""

import numpy as np
import librosa                # audio processing (spectrogram)
import sounddevice as sd      # recording from the microphone
import os
import tf_keras               # the CNN model (Keras 2)

from fusion import EMOTION_LABELS

SAMPLE_RATE = 22050   # sample rate (samples per second)
DURATION    = 3       # analysis window length in seconds
N_MELS      = 128     # number of mel bands (height of the spectrogram image)
MAX_FRAMES  = 130     # width of the spectrogram image (number of time frames)


class AudioEmotionDetector:

    def __init__(self, model_path: str):
        """Build the CNN architecture and load the trained weights into it."""
        print("טוען מודל קול...")

        # Build the CNN architecture (must be identical to the one the model was
        # trained with, so we can load the weights from model_weights.npz).
        # Structure: 4 convolution blocks (extract patterns from the spectrogram),
        # then Dense layers that classify into 7 emotions (softmax = probability
        # distribution).
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

        # Load the trained weights from the file into the architecture we built.
        data    = np.load('model_weights.npz', allow_pickle=True)
        weights = [data[f'w_{i}'] for i in range(len(data.files))]
        model.build((None, 128, 130, 1))   # input: a 128x130 spectrogram image
        model.set_weights(weights)

        self.model            = model
        self.speech_threshold = 0.01   # default - replaced after calibration

        # Rolling 3-second buffer - for frequent analysis on fresh data (less delay).
        self._rolling = np.zeros(int(DURATION * SAMPLE_RATE), dtype='float32')

        print("✅ מודל קול נטען")

    def calibrate(self, seconds: float = 1.5) -> float:
        """
        Measure the ambient noise level and compute a silence/speech threshold
        automatically. Called once at startup.
        """
        print(f"מכייל רעש סביבתי ({seconds} שניות) — אנא שמרי שקט...")
        try:
            audio = sd.rec(int(seconds * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            rms = float(np.sqrt(np.mean(audio.flatten() ** 2)))
        except Exception as e:
            print(f"⚠ שגיאת מיקרופון בכיול: {e} — משתמש בסף ברירת מחדל")
            self.speech_threshold = 0.015
            return self.speech_threshold

        # Threshold = 2x the background noise, clamped to a sensible range so that a
        # noisy calibration doesn't set it too high ("everything is silence") or too low.
        self.speech_threshold = float(np.clip(rms * 2.0, 0.005, 0.025))
        print(f"✅ כיול הסתיים! רעש רקע: {rms:.4f} | סף: {self.speech_threshold:.4f}")
        return self.speech_threshold

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Return True if the clip contains speech. Speech is "bursty", so we check the
        loudest 0.4 s window rather than the average of the whole clip (which would be
        diluted by the silence between words).
        """
        win  = int(0.4 * SAMPLE_RATE)
        step = win // 2
        if len(audio) >= win:
            levels = [float(np.sqrt(np.mean(audio[i:i + win] ** 2)))
                      for i in range(0, len(audio) - win + 1, step)]
            level = max(levels) if levels else 0.0
        else:
            level = float(np.sqrt(np.mean(audio ** 2)))
        return level > self.speech_threshold

    def record_audio(self, duration=DURATION, sr=SAMPLE_RATE):
        """Record `duration` seconds from the mic. On failure returns silence (so the thread doesn't die)."""
        try:
            audio = sd.rec(int(duration * sr), samplerate=sr,
                           channels=1, dtype='float32')
            sd.wait()
            return audio.flatten()
        except Exception as e:
            print(f"⚠ שגיאת מיקרופון: {e}")
            import time as _t
            _t.sleep(1.0)
            return np.zeros(int(duration * sr), dtype='float32')

    def extract_spectrogram(self, audio, sr=SAMPLE_RATE):
        """Convert a waveform into a normalized mel-spectrogram of fixed size (128 x MAX_FRAMES)."""
        mel    = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)                       # to decibels
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)  # normalize 0..1
        # Pad or crop to a fixed width so the CNN input size is constant.
        if mel_db.shape[1] < MAX_FRAMES:
            mel_db = np.pad(mel_db, ((0,0), (0, MAX_FRAMES - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :MAX_FRAMES]
        return mel_db

    def predict(self, audio):
        """Take a waveform, compute its spectrogram, and detect emotion. Returns (emotion, percent, probs)."""
        spec  = self.extract_spectrogram(audio)          # waveform -> spectrogram image
        spec  = spec[np.newaxis, ..., np.newaxis]         # add batch + channel dims for the CNN
        probs = self.model.predict(spec, verbose=0)[0]    # run the model -> 7 probabilities
        idx   = int(np.argmax(probs))                     # strongest emotion
        return EMOTION_LABELS[idx], float(probs[idx]) * 100, probs

    def analyze(self) -> tuple:
        """
        Record a full clip and analyze it. Returns (emotion, conf, probs, audio).
        conf=0 when there is no speech - fusion then ignores the audio automatically.
        """
        audio = self.record_audio()

        if not self.is_speech(audio):
            return 'neutral', 0.0, np.ones(7) / 7, audio

        emotion, conf, probs = self.predict(audio)
        print(f"קול --> {emotion} ({conf:.1f}%)")
        return emotion, conf, probs, audio

    def analyze_rolling(self, chunk_sec: float = 1.0) -> tuple:
        """
        Record a short chunk (~1 s), append it to the 3-second window, and analyze
        the window. This updates the audio ~every second on fresh data (instead of
        every 3 seconds). Returns (emotion, conf, probs, audio) - same as analyze().
        """
        chunk = self.record_audio(duration=chunk_sec)
        keep  = int(DURATION * SAMPLE_RATE)
        self._rolling = np.concatenate([self._rolling, chunk])[-keep:]
        audio = self._rolling

        if not self.is_speech(audio):
            return 'neutral', 0.0, np.ones(7) / 7, audio

        emotion, conf, probs = self.predict(audio)
        print(f"קול --> {emotion} ({conf:.1f}%)")
        return emotion, conf, probs, audio
