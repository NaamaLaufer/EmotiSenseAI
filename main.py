"""
main.py — EmotiSense AI
"""

import sys
# Ensure Hebrew/emoji prints don't crash the app on legacy Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import threading
import queue
import time
import os
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from fusion import (
    EMOTION_LABELS, EMOTION_COLORS, deepface_to_probs,
    fuse_emotions, fuse_three_channels, detect_contradiction,
)
from audio_emotion import AudioEmotionDetector
from audio_stream import AudioCapture

MODEL_PATH      = "model_weights.npz"   # audio model weights
HISTORY_MAX     = 30    # how many emotions to keep in the history chart
UI_UPDATE_MS    = 80    # GUI refresh interval in ms (80ms ~ 12 times per second)
SMOOTHING_ALPHA = 0.15  # temporal smoothing: 0.15 = 15% new + 85% old (stable, less jumpy)
AUDIO_WEIGHT    = 0.35  # <- audio has only 35% influence - raise it for more

# DeepFace face detector. 'yunet' = fast, accurate, OpenCV-based (no clash with the
# audio model's tf_keras). Note: 'retinaface'/'mtcnn' clash with tf_keras and are avoided.
# An automatic fallback to 'opencv' is built into the code in case of failure.
DEEPFACE_BACKEND = 'yunet'

BG     = '#0A1120'
CARD   = '#131D33'
CARD2  = '#1B2740'
WHITE  = '#FFFFFF'
MUTED  = '#92A4C2'
MUTED2 = '#5E7193'
CYAN   = '#22D3EE'

EMOTION_EMOJI = {
    'neutral': '😐', 'happy': '😊', 'sad': '😢',
    'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲',
}


class EmotionApp:
    """
    האפליקציה הראשית — מתזמרת את שלושת הערוצים ואת הממשק הגרפי.

    ארכיטקטורה (כל ערוץ ב-thread נפרד כדי שאף אחד לא יחסום את השני):
      • _video_loop  — קורא מהמצלמה, מריץ DeepFace, ושולח תוצאות ל-video_queue.
      • _audio_loop  — מנתח את הקול מהזרם הרציף, ושולח ל-audio_queue.
      • _text_loop   — מתמלל דיבור (Whisper) ומזהה רגש, ושולח ל-text_out_queue.
      • _update_ui   — רץ על ה-thread הראשי כל 80ms: אוסף מכל התורים, ממזג
                        (fusion.py), ומצייר את הממשק.

    התקשורת בין ה-threads נעשית דרך תורים (Queue) — בטוח ל-threads ולא חוסם.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EmotiSense AI")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.running          = True
        self.use_text_channel = tk.BooleanVar(value=True)
        self._use_text        = True
        self.calibrated       = False

        self.video_result  = {'emotion': 'neutral', 'conf': 0.0, 'probs': np.ones(7)/7}
        self.audio_result  = {'emotion': 'neutral', 'conf': 0.0, 'probs': np.ones(7)/7}
        self.text_result   = {'emotion': 'neutral', 'conf': 0.0, 'probs': np.ones(7)/7, 'text': ''}
        self.fusion_result = {'emotion': 'neutral', 'conf': 0.0}
        self.contradiction = ''
        self.emotion_history = []
        self.current_frame   = None

        self.smooth_video = np.ones(7) / 7
        self.smooth_audio = np.ones(7) / 7
        self.smooth_text  = np.ones(7) / 7

        self.got_audio = False
        self.got_video = False

        self.video_queue       = queue.Queue(maxsize=3)
        self.audio_queue       = queue.Queue(maxsize=1)
        self.text_in_queue     = queue.Queue(maxsize=1)   # audio chunks for transcription
        self.text_out_queue    = queue.Queue(maxsize=1)   # transcription results

        # DeepFace state - updated from a separate thread
        self._df_emotion = 'neutral'
        self._df_conf    = 0.0
        self._df_probs   = np.ones(7) / 7
        self._df_running = False
        self._df_lock    = threading.Lock()

        self._load_models()
        self._build_ui()

        threading.Thread(target=self._video_loop, daemon=True).start()
        threading.Thread(target=self._calibrate_then_audio, daemon=True).start()
        threading.Thread(target=self._text_loop, daemon=True).start()

        self.root.after(UI_UPDATE_MS, self._update_ui)

    # ══════════════════════════════════════════════════════════
    def _load_models(self):
        """Load the audio CNN and (on the main thread) the text model, before starting threads."""
        print("מאתחל מודלים...")
        self.audio_detector = AudioEmotionDetector(MODEL_PATH)
        # Load the text model here - on the main thread and before the other threads
        # run - so that torch (the classifier) doesn't clash with TensorFlow
        # (audio/DeepFace) during concurrent loading. One-time at startup (~a few seconds).
        if self._use_text:
            try:
                print("טוען ערוץ טקסט (מומלץ לדבר אנגלית)...")
                import text_emotion
                self._analyze_text  = text_emotion.analyze_text_emotion
                self._text_preloaded = True
            except Exception as e:
                print(f"שגיאת טעינת טקסט: {e}")
        print("כל המודלים נטענו")

    def _calibrate_then_audio(self):
        """Calibrate the mic noise, start the continuous audio stream, then run the audio loop."""
        self.audio_detector.calibrate(seconds=1.5)
        # Shared continuous mic stream - feeds both the audio and the text channels (gapless)
        self.audio_capture = AudioCapture()
        self.audio_capture.start()
        self.calibrated = True
        self._audio_loop()

    # ══════════════════════════════════════════════════════════
    # VIDEO - frame reading and DeepFace run separately (no delay!)
    # ══════════════════════════════════════════════════════════
    def _run_deepface(self, frame_small):
        """Run DeepFace emotion analysis on a frame (in a sub-thread, so it never blocks video)."""
        from deepface import DeepFace
        # Try the accurate detector first; on failure, fall back to opencv so it never gets stuck
        for backend in (DEEPFACE_BACKEND, 'opencv'):
            try:
                result = DeepFace.analyze(
                    frame_small, actions=['emotion'],
                    detector_backend=backend,
                    enforce_detection=False, silent=True
                )
                emotion_dict = result[0]['emotion']
                dom_emotion  = result[0]['dominant_emotion']
                conf         = emotion_dict[dom_emotion]
                probs        = deepface_to_probs(emotion_dict)
                with self._df_lock:
                    self._df_emotion = dom_emotion
                    self._df_conf    = conf
                    self._df_probs   = probs
                break
            except Exception:
                continue
        self._df_running = False

    def _open_camera(self):
        """Open the camera reliably on Windows (CAP_DSHOW first), with a fallback."""
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print("✅ מצלמה נפתחה")
                return cap
            cap.release()
        print("❌ לא נמצאה מצלמה")
        return None

    def _video_loop(self):
        """Video thread: read camera frames, trigger DeepFace periodically, push to video_queue."""
        cap         = self._open_camera()
        frame_count = 0
        fails       = 0

        while self.running:
            if cap is None:
                time.sleep(0.5)
                cap = self._open_camera()   # retry if the camera isn't available
                continue

            ret, frame = cap.read()
            if not ret:
                fails += 1
                if fails > 30:              # camera disconnected - try to reopen
                    cap.release()
                    cap = None
                    fails = 0
                time.sleep(0.033)
                continue
            fails = 0

            frame_count += 1

            # Run DeepFace in a separate thread every 10 frames (less CPU contention)
            if frame_count % 10 == 0 and not self._df_running:
                self._df_running = True
                small = cv2.resize(frame, (480, 360))   # larger = more accurate face detection
                threading.Thread(
                    target=self._run_deepface,
                    args=(small.copy(),),
                    daemon=True
                ).start()

            # Take the latest DeepFace result (without waiting)
            with self._df_lock:
                emotion = self._df_emotion
                conf    = self._df_conf
                probs   = self._df_probs.copy()

            # Send the frame immediately - non-blocking!
            try:
                self.video_queue.put_nowait({
                    'emotion': emotion, 'conf': conf,
                    'probs': probs, 'frame': frame.copy()
                })
            except queue.Full:
                try:
                    self.video_queue.get_nowait()
                    self.video_queue.put_nowait({
                        'emotion': emotion, 'conf': conf,
                        'probs': probs, 'frame': frame.copy()
                    })
                except queue.Empty:
                    pass

        if cap is not None:
            cap.release()

    # ══════════════════════════════════════════════════════════
    # AUDIO
    # ══════════════════════════════════════════════════════════
    def _audio_loop(self):
        """Analyze the last 3 s from the continuous stream every ~0.8 s (audio thread)."""
        while self.running:
            time.sleep(0.8)
            window = self.audio_capture.get_window(3.0)
            if self.audio_detector.is_speech(window):
                emotion, conf, probs = self.audio_detector.predict(window)
            else:
                emotion, conf, probs = 'neutral', 0.0, np.ones(7) / 7
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait({'audio': (emotion, conf, probs)})
            except queue.Full:
                pass

    def _text_loop(self):
        """
        Continuous windowed transcription: transcribes the last 4 s from the stream
        every ~2.5 s. Whisper's built-in VAD skips silence (fast), so this captures any
        speech without relying on a fragile "end-of-sentence" detection.
        """
        while self.running:
            cap     = getattr(self, 'audio_capture', None)
            analyze = getattr(self, '_analyze_text', None)   # set by the model preload
            if cap is None or analyze is None or not self._use_text:
                time.sleep(0.3)
                continue

            window = cap.get_window(4.0)
            try:
                te, tc, tp, txt = analyze(window, sr=22050)
                if txt:   # only when real speech was detected
                    print(f"[TEXT] {te} ({tc:.1f}%) | '{txt}'")
                    try:
                        self.text_out_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.text_out_queue.put_nowait((te, tc, tp, txt))
            except Exception as e:
                print(f"שגיאת טקסט: {e}")

            time.sleep(2.5)

    # ══════════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════════
    def _build_ui(self):
        """Build the whole tkinter GUI (video canvas, final-emotion card, per-channel rows, chart)."""
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        header = tk.Frame(self.root, bg=BG, pady=8)
        header.pack(fill='x')
        tk.Label(header, text="EmotiSense AI",
                 font=('Helvetica', 18, 'bold'), fg=CYAN, bg=BG).pack(side='left', padx=18)
        tk.Checkbutton(header, text="ערוץ טקסט (Whisper)",
                       variable=self.use_text_channel,
                       command=self._on_text_toggle,
                       fg=MUTED, bg=BG, selectcolor=CARD,
                       activeforeground=WHITE, activebackground=BG,
                       font=('Helvetica', 11)).pack(side='right', padx=18)
        self.lbl_calibration = tk.Label(header, text="⏳ מכייל...",
                                         fg='#FFD700', bg=BG, font=('Helvetica', 11))
        self.lbl_calibration.pack(side='right', padx=12)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=10, pady=4)

        left = tk.Frame(main, bg=BG)
        left.pack(side='left', fill='both', expand=True, padx=(0,6))

        self.video_canvas = tk.Canvas(left, width=540, height=380,
                                      bg='#000000', highlightthickness=1,
                                      highlightbackground=CARD2)
        self.video_canvas.pack(pady=(0,6))
        # A single image item that gets updated - instead of creating a new item every frame (a leak that freezes the UI)
        self._canvas_img = self.video_canvas.create_image(0, 0, anchor='nw')

        video_bar = tk.Frame(left, bg=CARD, pady=6, padx=10)
        video_bar.pack(fill='x', pady=(0,6))
        tk.Label(video_bar, text="וידאו:", fg=MUTED, bg=CARD,
                 font=('Helvetica', 11)).pack(side='right')
        self.lbl_video = tk.Label(video_bar, text="ממתין...",
                                  fg=CYAN, bg=CARD, font=('Helvetica', 13, 'bold'))
        self.lbl_video.pack(side='right', padx=8)

        self._build_chart(left)

        right = tk.Frame(main, bg=BG, width=340)
        right.pack(side='right', fill='y', padx=(6,0))
        right.pack_propagate(False)

        fc = tk.Frame(right, bg=CARD, pady=14, padx=14)
        fc.pack(fill='x', pady=(0,8))
        tk.Label(fc, text="רגש סופי", fg=MUTED, bg=CARD,
                 font=('Helvetica', 11)).pack()
        self.lbl_emoji   = tk.Label(fc, text="😐", bg=CARD, font=('Helvetica', 42))
        self.lbl_emoji.pack()
        self.lbl_emotion = tk.Label(fc, text="NEUTRAL", fg=WHITE, bg=CARD,
                                    font=('Helvetica', 22, 'bold'))
        self.lbl_emotion.pack()
        self.lbl_conf    = tk.Label(fc, text="—", fg=MUTED, bg=CARD,
                                    font=('Helvetica', 14))
        self.lbl_conf.pack()
        self.conf_bar = ttk.Progressbar(fc, length=280, mode='determinate', maximum=100)
        self.conf_bar.pack(pady=(8,0))

        cc = tk.Frame(right, bg=CARD, pady=10, padx=14)
        cc.pack(fill='x', pady=(0,8))
        tk.Label(cc, text="פירוט לפי ערוץ", fg=MUTED, bg=CARD,
                 font=('Helvetica', 10)).pack(anchor='e')
        self.lbl_audio_ch = self._ch_row(cc, "קול")
        self.lbl_text_ch  = self._ch_row(cc, "טקסט")

        self.lbl_contradiction = tk.Label(right, text="", fg='#FFD700', bg=BG,
                                          font=('Helvetica', 11, 'bold'), wraplength=320)
        self.lbl_contradiction.pack(pady=4)

    def _ch_row(self, parent, label):
        """Create one per-channel status row (label + value) and return the value Label widget."""
        f = tk.Frame(parent, bg=CARD)
        f.pack(fill='x', pady=3)
        tk.Label(f, text=label, fg=MUTED, bg=CARD,
                 font=('Helvetica', 11), width=8, anchor='e').pack(side='right')
        lbl = tk.Label(f, text="ממתין...", fg=MUTED2, bg=CARD,
                       font=('Helvetica', 11, 'bold'))
        lbl.pack(side='right', padx=6)
        return lbl

    def _on_text_toggle(self):
        """Handle the 'text channel' checkbox: enable/disable the text channel."""
        self._use_text = self.use_text_channel.get()
        print(f"ערוץ טקסט: {'פעיל' if self._use_text else 'כבוי'}")
        if self._use_text:
            self._preload_text_model()

    def _preload_text_model(self):
        """Preload the text model (Whisper) in the background so the first transcription doesn't stall."""
        if getattr(self, '_text_preloaded', False):
            return
        self._text_preloaded = True
        try:
            self.lbl_text_ch.config(text="טוען מודל...", fg=MUTED2)
        except Exception:
            pass

        def _preload():
            try:
                # Single import in one place only - avoids an import race between threads
                import text_emotion
                self._analyze_text = text_emotion.analyze_text_emotion
                print("✅ מודל תמלול מוכן")
            except Exception as e:
                print(f"שגיאת טעינת טקסט: {e}")
                self._text_preloaded = False

        threading.Thread(target=_preload, daemon=True).start()

    def _build_chart(self, parent):
        """Create the matplotlib bar chart that shows the recent emotion history."""
        self.fig, self.ax = plt.subplots(figsize=(5.2, 1.8), facecolor=CARD)
        self.ax.set_facecolor(CARD)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.chart_canvas.get_tk_widget().pack(fill='x')
        self._draw_chart()

    def _draw_chart(self):
        """Redraw the emotion-history chart (throttled - see below)."""
        # matplotlib drawing is expensive - throttle to once every 0.8 s so the UI doesn't freeze
        now = time.time()
        if now - getattr(self, '_last_chart_draw', 0.0) < 0.8:
            return
        self._last_chart_draw = now

        self.ax.clear()
        self.ax.set_facecolor(CARD)
        self.fig.patch.set_facecolor(CARD)
        if not self.emotion_history:
            self.ax.text(0.5, 0.5, 'ממתין לנתונים...', ha='center', va='center',
                         color=MUTED, fontsize=10, transform=self.ax.transAxes)
        else:
            colors = [EMOTION_COLORS.get(e, '#95A5A6') for e in self.emotion_history]
            self.ax.bar(range(len(self.emotion_history)), [1]*len(self.emotion_history),
                        color=colors, width=0.9, edgecolor='none')
            self.ax.set_yticks([])
            self.ax.set_xticks([])
            for s in self.ax.spines.values():
                s.set_visible(False)
        self.chart_canvas.draw()

    # ══════════════════════════════════════════════════════════
    # UI LOOP
    # ══════════════════════════════════════════════════════════
    def _update_ui(self):
        """Main UI loop (every 80 ms): drain the queues, fuse, and redraw. Runs on the main thread."""
        if not self.running:
            return

        if self.calibrated:
            self.lbl_calibration.config(text="✅ מכוייל", fg=MUTED2)

        changed   = False
        new_frame = False

        try:
            vdata = self.video_queue.get_nowait()
            self.video_result  = vdata
            self.current_frame = vdata['frame']
            self.got_video     = True
            changed   = True
            new_frame = True
        except queue.Empty:
            pass

        try:
            adata = self.audio_queue.get_nowait()
            ae, ac, ap = adata['audio']
            self.audio_result = {'emotion': ae, 'conf': ac, 'probs': ap}
            self.got_audio    = True
            changed = True
        except queue.Empty:
            pass

        # Text results arrive from a separate loop - they don't block the audio
        try:
            te, tc, tp, txt = self.text_out_queue.get_nowait()
            self.text_result = {'emotion': te, 'conf': tc, 'probs': tp, 'text': txt}
            changed = True
        except queue.Empty:
            pass

        if changed:
            self._compute_fusion()
            self._refresh_labels()

        # Render video only when a new frame arrived - saves redundant image conversions on the main thread
        if new_frame and self.current_frame is not None:
            self._show_frame(self.current_frame)

        self.root.after(UI_UPDATE_MS, self._update_ui)

    # ══════════════════════════════════════════════════════════
    # FUSION - audio is scaled by AUDIO_WEIGHT
    # ══════════════════════════════════════════════════════════
    def _compute_fusion(self):
        """Smooth each channel over time, then fuse them (fusion.py) into the final emotion."""
        # ── Temporal smoothing ───────────────────────────────────
        # במקום לקפוץ לכל תוצאה חדשה, מערבבים 15% חדש + 85% ישן.
        # זה מונע "ריצוד" של הרגש הסופי ונותן תחושה יציבה.
        self.smooth_video = (SMOOTHING_ALPHA * self.video_result['probs'] +
                             (1 - SMOOTHING_ALPHA) * self.smooth_video)

        ac = self.audio_result['conf']
        if ac > 0:   # מחליקים קול רק כשזוהה דיבור (אחרת משאירים את הקודם)
            self.smooth_audio = (SMOOTHING_ALPHA * self.audio_result['probs'] +
                                 (1 - SMOOTHING_ALPHA) * self.smooth_audio)

        vc = self.video_result['conf']

        # מחלישים את השפעת הקול לפי AUDIO_WEIGHT (הקול פחות אמין מהפנים)
        ac_scaled = ac * AUDIO_WEIGHT if ac > 0 else 0

        # ── המיזוג עצמו (ראה fusion.py) ──────────────────────────
        if self._use_text and self.text_result['conf'] > 0:
            # 3 ערוצים פעילים: וידאו + קול + טקסט
            self.smooth_text = (SMOOTHING_ALPHA * self.text_result['probs'] +
                                (1 - SMOOTHING_ALPHA) * self.smooth_text)
            tc = self.text_result['conf']
            emotion, conf, _ = fuse_three_channels(
                self.smooth_video, vc,
                self.smooth_audio, ac_scaled,
                self.smooth_text,  tc
            )
            # בדיקת סתירה בין הטקסט (מילים) לבין הקול+הפנים
            is_con, msg = detect_contradiction(
                self.text_result['emotion'],
                self.audio_result['emotion'],
                self.video_result['emotion']
            )
            self.contradiction = msg if is_con else ''
        else:
            # רק 2 ערוצים: וידאו + קול (הטקסט כבוי או ללא דיבור)
            emotion, conf, _ = fuse_emotions(
                self.smooth_video, vc,
                self.smooth_audio, ac_scaled
            )
            self.contradiction = ''

        # שומרים את התוצאה הסופית + מוסיפים להיסטוריה (לגרף)
        self.fusion_result = {'emotion': emotion, 'conf': conf}
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > HISTORY_MAX:
            self.emotion_history.pop(0)

    # ══════════════════════════════════════════════════════════
    # תוויות
    # ══════════════════════════════════════════════════════════
    def _refresh_labels(self):
        """Update all on-screen labels (final emotion, per-channel emotions, contradiction, chart)."""
        fe    = self.fusion_result['emotion']
        fc    = self.fusion_result['conf']
        color = EMOTION_COLORS.get(fe, WHITE)

        self.lbl_emoji.config(text=EMOTION_EMOJI.get(fe, '😐'))
        self.lbl_emotion.config(text=fe.upper(), fg=color)
        self.lbl_conf.config(text=f"{fc:.1f}%")
        self.conf_bar['value'] = fc

        ve       = self.video_result['emotion']
        vc       = self.video_result['conf']
        ve_color = EMOTION_COLORS.get(ve, WHITE)
        if self.got_video:
            self.lbl_video.config(
                text=f"{EMOTION_EMOJI.get(ve,'')} {ve}  {vc:.1f}%", fg=ve_color)

        ae       = self.audio_result['emotion']
        ac       = self.audio_result['conf']
        ae_color = EMOTION_COLORS.get(ae, WHITE)
        if not self.got_audio:
            self.lbl_audio_ch.config(text="מקליט...", fg=MUTED2)
        else:
            self.lbl_audio_ch.config(
                text=f"{EMOTION_EMOJI.get(ae,'')} {ae}  {ac:.1f}%", fg=ae_color)

        te       = self.text_result['emotion']
        tc       = self.text_result['conf']
        te_color = EMOTION_COLORS.get(te, MUTED)
        if not self._use_text:
            self.lbl_text_ch.config(text="לא פעיל", fg=MUTED2)
        elif not getattr(self, '_text_preloaded', False):
            self.lbl_text_ch.config(text="טוען מודל...", fg=MUTED2)
        else:
            self.lbl_text_ch.config(
                text=f"{EMOTION_EMOJI.get(te,'')} {te}  {tc:.1f}%", fg=te_color)

        self.lbl_contradiction.config(text=self.contradiction)
        self._draw_chart()

    def _show_frame(self, frame: np.ndarray):
        """Draw the current frame (with the emotion label) onto the video canvas."""
        fe    = self.fusion_result['emotion']
        color = EMOTION_COLORS.get(fe, '#FFFFFF')
        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

        display = frame.copy()
        cv2.putText(display, fe.upper(), (12,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (b,g,r), 2, cv2.LINE_AA)

        h, w  = display.shape[:2]
        new_h = int(h * 540 / w)
        display = cv2.resize(display, (540, new_h))

        rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_canvas.config(height=new_h)
        # מעדכנים את הפריט הקיים במקום ליצור חדש — מונע הצטברות ותקיעה
        self.video_canvas.itemconfig(self._canvas_img, image=photo)
        self.video_canvas.image = photo

    def _on_close(self):
        """Handle window close: stop the threads and destroy the window."""
        self.running = False
        time.sleep(0.3)
        self.root.destroy()


if __name__ == "__main__":
    if not os.path.exists('model_weights.npz'):
        print("לא נמצא model_weights.npz")
        exit(1)
    root = tk.Tk()
    app  = EmotionApp(root)
    root.mainloop()