import numpy as np

EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

EMOTION_COLORS = {
    'neutral':   '#95A5A6',
    'happy':     '#FFD700',
    'sad':       '#6B9FD4',
    'angry':     '#E74C3C',
    'fearful':   '#9B59B6',
    'disgust':   '#2ECC71',
    'surprised': '#F39C12',
}

DEEPFACE_MAP = {
    'angry':    'angry',
    'disgust':  'disgust',
    'fear':     'fearful',
    'happy':    'happy',
    'sad':      'sad',
    'surprise': 'surprised',
    'neutral':  'neutral',
}

def deepface_to_probs(deepface_dict):
    probs = np.zeros(7)
    for label, score in deepface_dict.items():
        our = DEEPFACE_MAP.get(label.lower())
        if our and our in EMOTION_LABELS:
            probs[EMOTION_LABELS.index(our)] = float(score) / 100.0
    total = probs.sum()
    return probs / total if total > 0 else np.ones(7) / 7

def fuse_emotions(video_probs, video_conf, audio_probs, audio_conf):
    total = video_conf + audio_conf
    if total == 0:
        final = (video_probs + audio_probs) / 2
    else:
        final = (video_conf/total)*video_probs + (audio_conf/total)*audio_probs
    idx = int(np.argmax(final))
    return EMOTION_LABELS[idx], float(final[idx]) * 100, final

def fuse_three_channels(video_probs, video_conf, audio_probs, audio_conf, text_probs, text_conf):
    total = video_conf + audio_conf + text_conf
    if total == 0:
        final = (video_probs + audio_probs + text_probs) / 3
    else:
        final = (video_conf/total)*video_probs + (audio_conf/total)*audio_probs + (text_conf/total)*text_probs
    idx = int(np.argmax(final))
    return EMOTION_LABELS[idx], float(final[idx]) * 100, final

def detect_contradiction(text_emotion, audio_emotion, video_emotion):
    positive = {'happy'}
    negative  = {'sad', 'angry', 'fearful', 'disgust'}
    def polarity(e):
        if e in positive: return 'positive'
        if e in negative: return 'negative'
        return 'neutral'
    tp = polarity(text_emotion)
    ap = polarity(audio_emotion)
    vp = polarity(video_emotion)
    if tp != 'neutral' and ap != 'neutral' and tp != ap and ap == vp:
        return True, f"contradiction: text={text_emotion} | audio+video={audio_emotion}"
    return False, ""