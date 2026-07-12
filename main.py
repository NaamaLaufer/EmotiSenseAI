"""
main.py — EmotiSense AI
"""

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
    EMOTION_LABELS, EMOTION_COLORS,
    fuse_emotions, fuse_three_channels, detect_contradiction,
)
from video_emotion import VideoEmotionDetector
from audio_emotion import AudioEmotionDetector

# ─────────────────────────────────────────────────────────────
# הגדרות
# ─────────────────────────────────────────────────────────────
MODEL_PATH          = "model_weights.npz"
VIDEO_ANALYZE_EVERY = 8
HISTORY_MAX         = 30
UI_UPDATE_MS        = 100
SMOOTHING_ALPHA     = 0.15   # נמוך = שינוי איטי יותר (0.05–0.4)

BG     = '#0A1120'
CARD   = '#131D33'
CARD2  = '#1B2740'
WHITE  = '#FFFFFF'
MUTED  = '#92A4C2'
MUTED2 = '#5E7193'
CYAN   = '#22D3EE'

EMOTION_EMOJI = {
    'neutral':   '😐',
    'happy':     '😊',
    'sad':       '😢',
    'angry':     '😠',
    'fearful':   '😨',
    'disgust':   '🤢',
    'surprised': '😲',
}


class EmotionApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EmotiSense AI")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.running          = True
        self.use_text_channel = tk.BooleanVar(value=False)

        # תוצאות מכל ערוץ
        self.video_result  = {'emotion': 'neutral', 'conf': 0.0, 'probs': np.ones(7)/7}
        self.audio_result  = {'emotion': 'neutral', 'conf': 0.0, 'probs': np.ones(7)/7}
        self.text_result   = {'emotion': 'neutral', 'conf': 0.0, 'probs': np.ones(7)/7, 'text': ''}
        self.fusion_result = {'emotion': 'neutral', 'conf': 0.0}
        self.contradiction = ''
        self.emotion_history = []
        self.current_frame   = None

        # EMA smoothing vectors
        self.smooth_video = np.ones(7) / 7
        self.smooth_audio = np.ones(7) / 7
        self.smooth_text  = np.ones(7) / 7

        # קבלת תוצאה ראשונה
        self.got_audio = False
        self.got_video = False

        self.video_queue = queue.Queue(maxsize=2)
        self.audio_queue = queue.Queue(maxsize=1)

        self._load_models()
        self._build_ui()

        threading.Thread(target=self._video_loop, daemon=True).start()
        threading.Thread(target=self._audio_loop, daemon=True).start()

        self.root.after(UI_UPDATE_MS, self._update_ui)

    # ══════════════════════════════════════════════════════════
    # מודלים
    # ══════════════════════════════════════════════════════════
    def _load_models(self):
        print("מאתחל מודלים...")
        self.video_detector = VideoEmotionDetector(analyze_every=VIDEO_ANALYZE_EVERY)
        self.audio_detector = AudioEmotionDetector(MODEL_PATH)
        print("כל המודלים נטענו")

    # ══════════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════════
    def _build_ui(self):
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        header = tk.Frame(self.root, bg=BG, pady=8)
        header.pack(fill='x')
        tk.Label(header, text="EmotiSense AI",
                 font=('Helvetica', 18, 'bold'), fg=CYAN, bg=BG).pack(side='left', padx=18)
        tk.Checkbutton(header, text="ערוץ טקסט (Whisper)",
                       variable=self.use_text_channel,
                       fg=MUTED, bg=BG, selectcolor=CARD,
                       activeforeground=WHITE, activebackground=BG,
                       font=('Helvetica', 11)).pack(side='right', padx=18)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=10, pady=4)

        # ─── שמאל — וידאו ───
        left = tk.Frame(main, bg=BG)
        left.pack(side='left', fill='both', expand=True, padx=(0,6))

        self.video_canvas = tk.Canvas(left, width=540, height=380,
                                      bg='#000000', highlightthickness=1,
                                      highlightbackground=CARD2)
        self.video_canvas.pack(pady=(0,6))

        video_bar = tk.Frame(left, bg=CARD, pady=6, padx=10)
        video_bar.pack(fill='x', pady=(0,6))
        tk.Label(video_bar, text="וידאו:", fg=MUTED, bg=CARD,
                 font=('Helvetica', 11)).pack(side='right')
        self.lbl_video = tk.Label(video_bar, text="ממתין...",
                                  fg=CYAN, bg=CARD, font=('Helvetica', 13, 'bold'))
        self.lbl_video.pack(side='right', padx=8)

        self._build_chart(left)

        # ─── ימין — תוצאות ───
        right = tk.Frame(main, bg=BG, width=340)
        right.pack(side='right', fill='y', padx=(6,0))
        right.pack_propagate(False)

        # כרטיס Fusion
        fc = tk.Frame(right, bg=CARD, pady=14, padx=14)
        fc.pack(fill='x', pady=(0,8))
        tk.Label(fc, text="רגש סופי", fg=MUTED, bg=CARD,
                 font=('Helvetica', 11)).pack()
        self.lbl_emoji = tk.Label(fc, text="😐", bg=CARD, font=('Helvetica', 42))
        self.lbl_emoji.pack()
        self.lbl_emotion = tk.Label(fc, text="NEUTRAL", fg=WHITE, bg=CARD,
                                    font=('Helvetica', 22, 'bold'))
        self.lbl_emotion.pack()
        self.lbl_conf = tk.Label(fc, text="—", fg=MUTED, bg=CARD,
                                 font=('Helvetica', 14))
        self.lbl_conf.pack()
        self.conf_bar = ttk.Progressbar(fc, length=280, mode='determinate', maximum=100)
        self.conf_bar.pack(pady=(8,0))

        # כרטיס ערוצים
        cc = tk.Frame(right, bg=CARD, pady=10, padx=14)
        cc.pack(fill='x', pady=(0,8))
        tk.Label(cc, text="פירוט לפי ערוץ", fg=MUTED, bg=CARD,
                 font=('Helvetica', 10)).pack(anchor='e')
        self.lbl_audio_ch = self._ch_row(cc, "קול")
        self.lbl_text_ch  = self._ch_row(cc, "טקסט")

        # תמלול
        tc2 = tk.Frame(right, bg=CARD, pady=10, padx=14)
        tc2.pack(fill='x', pady=(0,8))
        tk.Label(tc2, text="תמלול", fg=MUTED, bg=CARD,
                 font=('Helvetica', 10)).pack(anchor='e')
        self.txt_transcript = tk.Text(tc2, height=4, bg=CARD2, fg=WHITE,
                                      font=('Helvetica', 11), wrap='word',
                                      state='disabled', relief='flat')
        self.txt_transcript.pack(fill='x', pady=(4,0))

        self.lbl_contradiction = tk.Label(right, text="", fg='#FFD700', bg=BG,
                                          font=('Helvetica', 11, 'bold'), wraplength=320)
        self.lbl_contradiction.pack(pady=4)

    def _ch_row(self, parent, label):
        f = tk.Frame(parent, bg=CARD)
        f.pack(fill='x', pady=3)
        tk.Label(f, text=label, fg=MUTED, bg=CARD,
                 font=('Helvetica', 11), width=8, anchor='e').pack(side='right')
        lbl = tk.Label(f, text="ממתין...", fg=MUTED2, bg=CARD,
                       font=('Helvetica', 11, 'bold'))
        lbl.pack(side='right', padx=6)
        return lbl

    def _build_chart(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(5.2, 1.8), facecolor=CARD)
        self.ax.set_facecolor(CARD)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.chart_canvas.get_tk_widget().pack(fill='x')
        self._draw_chart()

    def _draw_chart(self):
        self.ax.clear()
        self.ax.set_facecolor(CARD)
        self.fig.patch.set_facecolor(CARD)
        if not self.emotion_history:
            self.ax.text(0.5, 0.5, 'ממתין לנתונים...',
                         ha='center', va='center', color=MUTED,
                         fontsize=10, transform=self.ax.transAxes)
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
    # Threads
    # ══════════════════════════════════════════════════════════
    def _video_loop(self):
        if not self.video_detector.open():
            return
        while self.running:
            ret, frame = self.video_detector.read_frame()
            if not ret:
                time.sleep(0.05)
                continue
            emotion, conf, probs = self.video_detector.analyze_frame(frame)
            try:
                self.video_queue.put_nowait(
                    {'emotion': emotion, 'conf': conf, 'probs': probs, 'frame': frame.copy()}
                )
            except queue.Full:
                pass
        self.video_detector.release()

    def _audio_loop(self):
        while self.running:
            emotion, conf, probs, audio = self.audio_detector.analyze()
            result = {'audio': (emotion, conf, probs), 'text': None}
            if self.use_text_channel.get():
                try:
                    from text_emotion import analyze_text_emotion
                    te, tc, tp, txt = analyze_text_emotion(audio)
                    result['text'] = (te, tc, tp, txt)
                except Exception as e:
                    print(f"שגיאת טקסט: {e}")
            try:
                self.audio_queue.put_nowait(result)
            except queue.Full:
                pass

    # ══════════════════════════════════════════════════════════
    # לולאת UI
    # ══════════════════════════════════════════════════════════
    def _update_ui(self):
        if not self.running:
            return

        changed = False

        try:
            vdata = self.video_queue.get_nowait()
            self.video_result  = vdata
            self.current_frame = vdata['frame']
            self.got_video     = True
            changed = True
        except queue.Empty:
            pass

        try:
            adata = self.audio_queue.get_nowait()
            ae, ac, ap = adata['audio']
            self.audio_result = {'emotion': ae, 'conf': ac, 'probs': ap}
            self.got_audio    = True
            if adata['text']:
                te, tc, tp, txt = adata['text']
                self.text_result = {'emotion': te, 'conf': tc, 'probs': tp, 'text': txt}
                self._set_transcript(txt)
            changed = True
        except queue.Empty:
            pass

        if changed:
            self._compute_fusion()
            self._refresh_labels()

        if self.current_frame is not None:
            self._show_frame(self.current_frame)

        self.root.after(UI_UPDATE_MS, self._update_ui)

    # ══════════════════════════════════════════════════════════
    # Fusion — EMA smoothing בלבד, ללא stability gate שבור
    # ══════════════════════════════════════════════════════════
    def _compute_fusion(self):
        # EMA — כל עדכון חדש משפיע רק SMOOTHING_ALPHA מהתוצאה הסופית
        self.smooth_video = (SMOOTHING_ALPHA * self.video_result['probs'] +
                             (1 - SMOOTHING_ALPHA) * self.smooth_video)
        self.smooth_audio = (SMOOTHING_ALPHA * self.audio_result['probs'] +
                             (1 - SMOOTHING_ALPHA) * self.smooth_audio)

        vc = self.video_result['conf']
        ac = self.audio_result['conf']

        if self.use_text_channel.get() and self.text_result['conf'] > 0:
            self.smooth_text = (SMOOTHING_ALPHA * self.text_result['probs'] +
                                (1 - SMOOTHING_ALPHA) * self.smooth_text)
            tc = self.text_result['conf']
            emotion, conf, _ = fuse_three_channels(
                self.smooth_video, vc,
                self.smooth_audio, ac,
                self.smooth_text,  tc
            )
            is_con, msg = detect_contradiction(
                self.text_result['emotion'],
                self.audio_result['emotion'],
                self.video_result['emotion']
            )
            self.contradiction = msg if is_con else ''
        else:
            emotion, conf, _ = fuse_emotions(
                self.smooth_video, vc,
                self.smooth_audio, ac
            )
            self.contradiction = ''

        self.fusion_result = {'emotion': emotion, 'conf': conf}

        self.emotion_history.append(emotion)
        if len(self.emotion_history) > HISTORY_MAX:
            self.emotion_history.pop(0)

    # ══════════════════════════════════════════════════════════
    # עדכון תוויות
    # ══════════════════════════════════════════════════════════
    def _refresh_labels(self):
        fe    = self.fusion_result['emotion']
        fc    = self.fusion_result['conf']
        color = EMOTION_COLORS.get(fe, WHITE)

        # Fusion
        self.lbl_emoji.config(text=EMOTION_EMOJI.get(fe, '😐'))
        self.lbl_emotion.config(text=fe.upper(), fg=color)
        self.lbl_conf.config(text=f"{fc:.1f}%")
        self.conf_bar['value'] = fc

        # וידאו
        ve       = self.video_result['emotion']
        vc       = self.video_result['conf']
        ve_color = EMOTION_COLORS.get(ve, WHITE)
        if self.got_video:
            self.lbl_video.config(
                text=f"{EMOTION_EMOJI.get(ve,'')} {ve}  {vc:.1f}%",
                fg=ve_color
            )

        # קול
        ae       = self.audio_result['emotion']
        ac       = self.audio_result['conf']
        ae_color = EMOTION_COLORS.get(ae, WHITE)
        if self.got_audio:
            self.lbl_audio_ch.config(
                text=f"{EMOTION_EMOJI.get(ae,'')} {ae}  {ac:.1f}%",
                fg=ae_color
            )
        else:
            self.lbl_audio_ch.config(text="מקליט...", fg=MUTED2)

        # טקסט
        te       = self.text_result['emotion']
        tc       = self.text_result['conf']
        te_color = EMOTION_COLORS.get(te, MUTED)
        if not self.use_text_channel.get():
            self.lbl_text_ch.config(text="לא פעיל", fg=MUTED2)
        elif tc > 0:
            self.lbl_text_ch.config(
                text=f"{EMOTION_EMOJI.get(te,'')} {te}  {tc:.1f}%",
                fg=te_color
            )
        else:
            self.lbl_text_ch.config(text="ממתין...", fg=MUTED2)

        self.lbl_contradiction.config(text=self.contradiction)
        self._draw_chart()

    # ══════════════════════════════════════════════════════════
    # הצגת פריים
    # ══════════════════════════════════════════════════════════
    def _show_frame(self, frame: np.ndarray):
        fe    = self.fusion_result['emotion']
        color = EMOTION_COLORS.get(fe, '#FFFFFF')
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        display = frame.copy()
        cv2.putText(display, fe.upper(), (12, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (b, g, r), 2, cv2.LINE_AA)

        h, w  = display.shape[:2]
        new_h = int(h * 540 / w)
        display = cv2.resize(display, (540, new_h))

        rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_canvas.config(height=new_h)
        self.video_canvas.create_image(0, 0, anchor='nw', image=photo)
        self.video_canvas.image = photo

    def _set_transcript(self, text: str):
        self.txt_transcript.config(state='normal')
        self.txt_transcript.delete('1.0', 'end')
        self.txt_transcript.insert('end', text or "—")
        self.txt_transcript.config(state='disabled')

    def _on_close(self):
        self.running = False
        time.sleep(0.3)
        self.root.destroy()


if __name__ == "__main__":
    if not os.path.exists('model_weights.npz'):
        print("לא נמצא model_weights.npz בתיקייה")
        exit(1)
    root = tk.Tk()
    app  = EmotionApp(root)
    root.mainloop()