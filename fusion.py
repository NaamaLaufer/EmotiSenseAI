"""
fusion.py - The fusion engine of EmotiSense
============================================
This is where the three channels (video, audio, text) are intelligently merged
into a single final emotion.

Core idea: not every channel is equally reliable at every moment. A channel that
is very confident (high confidence %) should have more influence. We therefore use
"power weights" (conf^2), which amplify the advantage of a confident channel over a
hesitant one.
"""

import numpy as np

# The seven emotions the system detects. The order is fixed and important
# (it is used as the index into every probability vector).
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# A representative color per emotion (used by the GUI).
EMOTION_COLORS = {
    'neutral':   '#95A5A6',   # gray
    'happy':     '#FFD700',   # gold
    'sad':       '#6B9FD4',   # blue
    'angry':     '#E74C3C',   # red
    'fearful':   '#9B59B6',   # purple
    'disgust':   '#2ECC71',   # green
    'surprised': '#F39C12',   # orange
}

# Maps DeepFace's emotion names (video channel) to our own names.
DEEPFACE_MAP = {
    'angry':    'angry',
    'disgust':  'disgust',
    'fear':     'fearful',
    'happy':    'happy',
    'sad':      'sad',
    'surprise': 'surprised',
    'neutral':  'neutral',
}


def deepface_to_probs(deepface_dict: dict) -> np.ndarray:
    """
    Convert DeepFace output (a dict of emotion -> percent) into a normalized
    probability vector of length 7.
    Example: {'happy': 80, 'sad': 20, ...} -> [0.0, 0.8, 0.2, ...]
    """
    probs = np.zeros(7)
    for label, score in deepface_dict.items():
        our = DEEPFACE_MAP.get(label.lower())          # translate to our emotion name
        if our and our in EMOTION_LABELS:
            probs[EMOTION_LABELS.index(our)] = float(score) / 100.0
    total = probs.sum()
    # Normalize so the probabilities sum to 1 (fall back to uniform if empty).
    return probs / total if total > 0 else np.ones(7) / 7


def _power_weights(conf_a: float, conf_b: float):
    """
    Compute two weights based on the square of each confidence - this amplifies
    the gap between a strong and a weak channel.
    Example: video=80%, audio=30%:
      linear (by confidence):  80:30   = 2.7:1
      power  (conf^2):         6400:900 = 7.1:1   <- video dominates much more
    """
    wa = conf_a ** 2
    wb = conf_b ** 2
    total = wa + wb
    if total == 0:                # both channels at confidence 0 -> split evenly
        return 0.5, 0.5
    return wa / total, wb / total


def fuse_emotions(video_probs: np.ndarray, video_conf: float,
                  audio_probs: np.ndarray, audio_conf: float) -> tuple:
    """
    Fuse 2 channels (video + audio) when the text channel is off.
    Returns: (winning_emotion, final_confidence_percent, fused_probability_vector)
    """
    wv, wa = _power_weights(video_conf, audio_conf)     # weight per channel
    final  = wv * video_probs + wa * audio_probs        # weighted average of the vectors

    idx = int(np.argmax(final))                         # index of the strongest emotion
    return EMOTION_LABELS[idx], float(final[idx]) * 100, final


def fuse_three_channels(video_probs: np.ndarray, video_conf: float,
                        audio_probs: np.ndarray, audio_conf: float,
                        text_probs:  np.ndarray, text_conf:  float) -> tuple:
    """
    Fuse 3 channels (video + audio + text) using power weights.

    Step 1 - weight per channel = confidence^2 divided by the total:
        w_video = conf_video^2 / (conf_video^2 + conf_audio^2 + conf_text^2)
    Step 2 - take the weighted average of the probability vectors.
    Step 3 - the emotion with the highest value wins; its value * 100 is the
             final confidence.

    Returns: (winning_emotion, final_confidence_percent, fused_probability_vector)
    """
    vc2 = video_conf ** 2
    ac2 = audio_conf ** 2
    tc2 = text_conf  ** 2
    total = vc2 + ac2 + tc2

    if total == 0:                # all confidences 0 -> simple average (safe fallback)
        final = (video_probs + audio_probs + text_probs) / 3
    else:
        final = (vc2 * video_probs + ac2 * audio_probs + tc2 * text_probs) / total

    idx = int(np.argmax(final))
    return EMOTION_LABELS[idx], float(final[idx]) * 100, final


def detect_contradiction(text_emotion: str,
                         audio_emotion: str,
                         video_emotion: str) -> tuple:
    """
    Detect an "emotional contradiction" - when the words say one thing but the
    face + voice say the opposite (e.g. positive words in a sad tone and sad
    expression - possibly sarcasm or masking).
    Returns: (is_contradiction: bool, explanation_message: str)
    """
    positive = {'happy'}                                  # positive emotions
    negative = {'sad', 'angry', 'fearful', 'disgust'}     # negative emotions

    def polarity(e):                                      # map an emotion to a polarity
        if e in positive:  return 'positive'
        if e in negative:  return 'negative'
        return 'neutral'

    tp, ap, vp = polarity(text_emotion), polarity(audio_emotion), polarity(video_emotion)

    # Contradiction = text has one polarity, and voice + face share the opposite one.
    if tp != 'neutral' and ap != 'neutral' and tp != ap and ap == vp:
        return True, f"סתירה: טקסט={text_emotion} | קול+פנים={audio_emotion}"
    return False, ""
