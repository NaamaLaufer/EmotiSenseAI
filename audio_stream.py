"""
audio_stream.py
---------------
Continuous (gapless) microphone capture using a sounddevice InputStream.
Keeps a ring buffer of the most recent audio and lets both channels (audio + text)
read the latest window at any time - one microphone stream, no gaps in recording.
"""

import numpy as np
import sounddevice as sd
import threading

SAMPLE_RATE = 22050
BLOCK       = int(0.1 * SAMPLE_RATE)   # 100 ms block


class AudioCapture:
    def __init__(self, buffer_sec: float = 6.0):
        """Create the capture with a ring buffer of `buffer_sec` seconds."""
        self.sr         = SAMPLE_RATE
        self._buf       = np.zeros(int(buffer_sec * SAMPLE_RATE), dtype='float32')
        self._lock      = threading.Lock()
        self._stream    = None

    def start(self):
        """Open and start the continuous microphone stream."""
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='float32',
            blocksize=BLOCK, callback=self._callback,
        )
        self._stream.start()
        print("✅ זרם מיקרופון רציף הופעל")

    def stop(self):
        """Stop and close the microphone stream."""
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, indata, frames, time_info, status):
        """
        Called by sounddevice on its own audio thread for every 100 ms block.
        Kept lightweight: just pushes the block into the ring buffer.
        """
        x = indata[:, 0]
        with self._lock:
            n = len(x)
            self._buf = np.roll(self._buf, -n)   # shift old samples left
            self._buf[-n:] = x                   # append the new block at the end

    def get_window(self, sec: float = 3.0) -> np.ndarray:
        """Return the last `sec` seconds of audio from the stream."""
        n = int(sec * SAMPLE_RATE)
        with self._lock:
            return self._buf[-n:].copy() if n <= self._buf.size else self._buf.copy()
