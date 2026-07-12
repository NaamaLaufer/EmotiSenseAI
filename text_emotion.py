import whisper
import numpy as np
import librosa
from transformers import pipeline

EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

EMOTION_MAP = {
    'joy':      'happy',
    'anger':    'angry',
    'sadness':  'sad',
    'fear':     'fearful',
    'disgust':  'disgust',
    'surprise': 'surprised',
    'neutral':  'neutral'
}

# טעינת מודלים
whisper_model = whisper.load_model("base")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

def analyze_text_emotion(audio_array, sr=22050):
    """
    מקבל numpy array של קול (מ-audio_emotion.py)
    מחזיר: (emotion, confidence, probs_vector, text)
    """
    # המרה ל-16kHz כי Whisper דורש
    audio_16k = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    # שלב 1: קול לטקסט (זיהוי שפה אוטומטי)
    result = whisper_model.transcribe(audio_16k, fp16=False)
    text = result["text"].strip()
    print(f"📝 תמלול: '{text}'")

    if not text:
        probs = np.ones(7) / 7
        return 'neutral', 14.3, probs, text

    # שלב 2: טקסט לרגש
    emotions = emotion_classifier(text)[0]

    # בניית וקטור הסתברויות מלא ב-7 רגשות
    probs = np.zeros(7)
    for item in emotions:
        mapped = EMOTION_MAP.get(item['label'], 'neutral')
        idx = EMOTION_LABELS.index(mapped)
        probs[idx] += item['score']

    # נרמול
    probs = probs / probs.sum()

    emotion_idx = int(np.argmax(probs))
    emotion     = EMOTION_LABELS[emotion_idx]
    confidence  = float(probs[emotion_idx]) * 100

    return emotion, confidence, probs, text