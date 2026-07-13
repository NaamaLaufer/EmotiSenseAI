# EmotiSense AI — Code Documentation

Real-time **multimodal emotion recognition**. The system reads emotion from three
independent channels — **face** (video), **voice tone** (audio), and **speech content**
(text) — and fuses them into a single, confidence-weighted result in real time.

---

## 1. Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Main application. Orchestrates the three channels + GUI (tkinter). |
| `fusion.py` | The fusion engine — combines the channels into one final emotion. |
| `video_emotion.py` | Standalone facial-emotion detector (DeepFace). Also usable alone. |
| `audio_emotion.py` | Voice-tone emotion detector (CNN over a mel-spectrogram). |
| `audio_stream.py` | Continuous, gapless microphone capture shared by audio + text. |
| `text_emotion.py` | Speech → text (faster-whisper) → emotion (distilroberta). |
| `model_weights.npz` | Trained weights for the audio CNN. |

The 7 emotions (fixed order, used as vector indices everywhere):
`neutral, happy, sad, angry, fearful, disgust, surprised`.

---

## 2. Architecture & Data Flow

Each channel runs on its **own thread** so none blocks the others. Threads communicate
with the UI through thread-safe **queues**. The UI thread pulls from the queues every
80 ms, fuses, and redraws.

```
 [Video thread]              [Audio thread]                [Text thread]
 camera (OpenCV)             AudioCapture stream           AudioCapture stream
 DeepFace (yunet)            mel-spectrogram + CNN          faster-whisper + distilroberta
      │ probs                     │ probs                        │ emotion+conf
      ▼                           ▼                              ▼
 video_queue                 audio_queue                    text_out_queue
      └───────────────┬───────────────┴──────────────────────────┘
                      ▼
              _update_ui()  (main thread, every 80 ms)
                      │  → fusion.py (weighted combine + smoothing)
                      ▼
              tkinter GUI  (live video, final emotion %, per-channel breakdown, chart)
```

**Why threads + queues?** Whisper transcription (~1–4 s) and the audio window
(~1 s) are slow; running them on the UI thread would freeze the window. Isolating them
keeps the interface responsive.

---

## 3. Module Reference

### 3.1 `fusion.py` — the fusion engine
The core idea: a channel that is **more confident** should have **more influence**.
Weights are the **square** of each channel's confidence (`conf²`), which amplifies the
gap between a confident and a hesitant channel.

- `deepface_to_probs(dict)` → converts DeepFace's `{emotion: percent}` output into a
  normalized 7-element probability vector.
- `fuse_emotions(video_probs, video_conf, audio_probs, audio_conf)` → 2-channel fusion
  (used when the text channel is off).
- `fuse_three_channels(...)` → 3-channel fusion. Weight of channel *i* is
  `conf_i² / Σ conf²`; the final probability vector is the weighted average of the three
  vectors. The winning emotion's value × 100 is the **final confidence**.
- `detect_contradiction(text, audio, video)` → flags when the **words** are positive but
  the **face + voice** are negative (or vice-versa) — e.g. sarcasm or masking.

**Example** (video happy 85%, audio sad 40%, text happy 70%):
weights = 85²:40²:70² = 7225:1600:4900 → video 52.6%, audio 11.7%, text 35.7%.
Video + text agree on *happy* (88.3% of the weight) → final emotion **happy**,
confidence ≈ 0.526·85 + 0.357·70 ≈ **70%**. Audio (sad) triggers a contradiction flag.

### 3.2 `audio_stream.py` — continuous microphone capture
`AudioCapture` opens **one** continuous `sounddevice.InputStream` (22050 Hz, 100 ms
blocks). Its callback keeps a **ring buffer** of the last few seconds. Both the audio and
text channels read from it via `get_window(sec)`. Using a single continuous stream
(instead of repeated blocking recordings) means **no audio is lost between reads**.

### 3.3 `audio_emotion.py` — voice-tone channel
Detects emotion from *how* you speak (tone, energy), not the words.
1. `extract_spectrogram(audio)` → mel-spectrogram (a 128×130 "image" of frequency vs.
   time).
2. A convolutional neural network (CNN) — 4 conv blocks + dense layers, `softmax` over 7
   emotions — classifies that image. The architecture is rebuilt in code and the trained
   weights are loaded from `model_weights.npz`.
3. `is_speech(audio)` → energy-based VAD (checks the loudest 0.4 s window) so silence is
   ignored (returns neutral, confidence 0, so fusion drops it).
4. `calibrate()` → measures ambient noise once at startup to set the speech threshold.

### 3.4 `text_emotion.py` — speech-content channel
1. **faster-whisper** (`base`, CTranslate2 int8) transcribes the speech. Configured for
   **English** (`language="en"`) — more accurate than Hebrew and ~3× faster; built-in VAD
   (`vad_filter`) skips silence.
2. The English text is classified by **`j-hartmann/emotion-english-distilroberta-base`**
   into the 7 emotions.
3. `analyze_text_emotion(audio)` returns `(emotion, confidence, probs, text)`. The channel
   displays the emotion + confidence %.

### 3.5 `video_emotion.py` — facial channel (standalone)
`VideoEmotionDetector` wraps DeepFace emotion analysis with a `yunet` face detector and
runs it every N frames. `main.py` contains its own threaded copy of this logic; this file
is a clean, runnable standalone version.

### 3.6 `main.py` — application & orchestration
`EmotionApp` builds the GUI and starts the threads:
- `_video_loop` — reads frames, runs DeepFace in a sub-thread (`yunet` backend, falls back
  to `opencv`), pushes results + frame to `video_queue`.
- `_audio_loop` — every ~0.8 s analyzes the last 3 s from the audio stream.
- `_text_loop` — every ~2.5 s transcribes the last 4 s and classifies the emotion.
- `_update_ui` — every 80 ms: drains the queues, calls `_compute_fusion`, redraws.
- `_compute_fusion` — applies **temporal smoothing** (`0.15·new + 0.85·old`, stable, no
  flicker) then calls the `fusion.py` functions and updates the final result + history.

---

## 4. Key Configuration (top of `main.py`)

| Constant | Meaning |
|----------|---------|
| `UI_UPDATE_MS = 80` | GUI refresh interval (~12 fps). |
| `SMOOTHING_ALPHA = 0.15` | Temporal smoothing factor (higher = more reactive, less stable). |
| `AUDIO_WEIGHT = 0.35` | Down-weights the audio channel (voice is less reliable than face). |
| `DEEPFACE_BACKEND = 'yunet'` | Face detector. `yunet` is OpenCV-based and does not clash with TensorFlow. |

---

## 5. Notable Technical Decisions

- **`yunet`, not `retinaface`.** `retinaface`/`mtcnn` are TensorFlow-based and crash once
  the audio model's `tf_keras` is loaded (`KerasTensor` / `streams::fork` errors). `yunet`
  (OpenCV) is safe; a fallback to `opencv` is built in.
- **torch loaded on the main thread.** The text classifier (PyTorch) is imported/loaded in
  the main thread *before* the TensorFlow channels start, to avoid a torch/TensorFlow
  concurrency crash.
- **Single continuous audio stream** shared by both audio channels — gapless capture.
- **English + `base` model** for text — best accuracy/speed trade-off on CPU.
- **`sys.stdout.reconfigure('utf-8')`** at startup so Hebrew/emoji console prints don't
  crash on legacy Windows code pages.

---

## 6. Running

```powershell
.\emotion_env\Scripts\python.exe main.py
```

Startup loads all models first (~10–15 s) — wait for the window to appear, then all three
channels run in parallel. Speak in **English**, in short sentences, for the text channel
to work best.

**Requirements:** Windows, Python 3.10 (the `emotion_env` virtualenv), a webcam, and a
microphone.
