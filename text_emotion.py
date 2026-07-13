"""
text_emotion.py
---------------
Text channel: detects emotion from the CONTENT of the speech (English).
faster-whisper transcribes the speech, and the distilroberta classifier detects the
emotion from the text. The channel displays emotion + confidence %.

English is recommended: more accurate (strong ASR + native English classifier, no
translation) and the 'base' model is fast and light - less load.
"""

import os
import numpy as np
import librosa
from faster_whisper import WhisperModel

EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Maps the classifier's labels to our emotion names.
EMOTION_MAP = {
    'joy':      'happy',
    'anger':    'angry',
    'sadness':  'sad',
    'fear':     'fearful',
    'disgust':  'disgust',
    'surprise': 'surprised',
    'neutral':  'neutral',
}

LANGUAGE = "en"     # spoken language - English (recommended). For Hebrew: "he" + task="translate"

# Common Whisper hallucinations on silence/noise - filtered out.
_HALLUCINATIONS = {
    '', '.', '. .', 'you', 'bye', 'bye.', 'thank you', 'thanks for watching',
    'thank you for watching', 'thank you very much', 'please subscribe', 'thanks',
}

_CPU_THREADS = max(2, (os.cpu_count() or 4) // 2)


def _load_whisper():
    """Load the lightest available model: 'base' (fast), falling back to 'small'."""
    for name in ("base", "small"):
        try:
            m = WhisperModel(name, device="cpu", compute_type="int8",
                             cpu_threads=_CPU_THREADS)
            print(f"✅ מודל תמלול: faster-whisper {name}")
            return m
        except Exception as e:
            print(f"(faster-whisper {name} not available: {e})")
    raise RuntimeError("No faster-whisper model found")


print("טוען מודל תמלול...")
whisper_model = _load_whisper()

print("טוען מסווג רגשות טקסט...")
from transformers import pipeline
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)
print("✅ ערוץ טקסט מוכן")


def _clean(text: str) -> str:
    """Normalize text for comparison against the hallucination list."""
    return text.strip().strip('.').strip().lower()


def analyze_text_emotion(audio_array, sr=22050):
    """
    Transcribe the speech (English) and detect the emotion from the text.
    Returns: (emotion, confidence, probs_vector, text)
    """
    neutral_probs = np.ones(7) / 7

    # Resample to 16 kHz (required by Whisper).
    audio_16k = librosa.resample(
        audio_array.astype(np.float32), orig_sr=sr, target_sr=16000
    )

    # Transcribe with the built-in VAD (Silero) - skips silence, far more robust
    # than an energy threshold.
    segments, _info = whisper_model.transcribe(
        audio_16k,
        language=LANGUAGE,
        task="transcribe",
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    text = " ".join(s.text for s in segments).strip()
    print(f"📝 טקסט: '{text}'")

    # Filter hallucinations / too-short text.
    if _clean(text) in _HALLUCINATIONS or len(_clean(text)) < 2:
        return 'neutral', 0.0, neutral_probs, ""

    # English text -> emotion + confidence.
    emotions = emotion_classifier(text)[0]
    probs = np.zeros(7)
    for item in emotions:
        mapped = EMOTION_MAP.get(item['label'], 'neutral')
        probs[EMOTION_LABELS.index(mapped)] += item['score']
    probs = probs / probs.sum() if probs.sum() > 0 else neutral_probs

    idx        = int(np.argmax(probs))
    emotion    = EMOTION_LABELS[idx]
    confidence = float(probs[idx]) * 100
    return emotion, confidence, probs, text
