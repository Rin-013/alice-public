#!/usr/bin/env python3
"""
Audio listener - taps Alice's voice envelope so motion can react to speech.

This is the wire that turns procedural_motion from "random fidgeting" into
"expression". The reactive paths in ProceduralMotion.tick (head nod, lean-in,
brow pops) all key off audio_env + onsets, but nothing was feeding them.
This module feeds them.

How it captures without touching the TTS path:
  TTS subprocess plays Alice's PCM into "CABLE Input" (VB-Audio Virtual Cable).
  VB-Cable loops that straight back out of "CABLE Output" as a *recording*
  device. We open an input stream on CABLE Output and read the envelope.
  Result: we hear exactly what Alice says - no game audio, no music, no mic -
  and the TTS subprocess never knows we're listening. Fully decoupled.

For dev-testing without chat.py running, pass --device "Microphone" (or any
name substring) and just talk into your mic to drive the motion.

Standalone meter (verify capture before wiring VTS):
    python streaming/animation/audio_listener.py
    python streaming/animation/audio_listener.py --device "Microphone"
"""

import threading
import time
from typing import Optional

try:
    import numpy as np
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    print(f"   Failed in {__file__}: {e}")


DEFAULT_DEVICE_MATCH = "CABLE Output"   # VB-Cable loopback = Alice's voice only
PREFERRED_HOSTAPI = "WASAPI"            # low latency, clean 2ch capture


def resolve_device(match: str = DEFAULT_DEVICE_MATCH) -> Optional[int]:
    """Find an input device whose name contains `match`, preferring WASAPI.

    Returns the device index, or None if nothing matches (caller can fall
    back to the default input or disable audio reactivity)."""
    if not AUDIO_AVAILABLE:
        return None
    match_l = match.lower()
    candidates = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        if match_l not in dev["name"].lower():
            continue
        hostapi = sd.query_hostapis(dev["hostapi"])["name"]
        # Prefer WASAPI, fewest channels (cheap reads), as a tiebreak.
        score = (0 if PREFERRED_HOSTAPI in hostapi else 1, dev["max_input_channels"])
        candidates.append((score, idx, dev["name"], hostapi))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    _, idx, name, hostapi = candidates[0]
    print(f"[AudioListener] capture device: [{idx}] {name} ({hostapi})")
    return idx


class AudioListener:
    """Background RMS-envelope tap with onset detection.

    Runs a PortAudio input stream on its own callback thread. The main loop
    just polls get_env() / poll_onset() each frame - no blocking, no audio
    data crosses into the render loop.

    Tunables:
      gain       - multiplier on raw RMS before clamping to 0..1. Voice RMS
                   sits ~0.01-0.2, so ~6-8 maps normal speech to a healthy
                   envelope. Bump if Alice is quiet, drop if she pins to 1.0.
      attack     - per-block rise rate (fast, so she reacts on syllable onset).
      release    - per-block fall rate (slow, so she doesn't twitch silent
                   between words - holds the "talking" pose through gaps).
      onset_thr  - how far env must jump above its slow baseline to fire an
                   onset (a brow-pop / gaze-flick on emphasis).
    """

    def __init__(
        self,
        device: Optional[int] = None,
        samplerate: Optional[int] = None,
        gain: float = 7.0,
        attack: float = 0.55,
        release: float = 0.12,
        onset_thr: float = 0.16,
        onset_min_gap: float = 0.18,
    ):
        self.device = device
        self.samplerate = samplerate
        self.gain = gain
        self.attack = attack
        self.release = release
        self.onset_thr = onset_thr
        self.onset_min_gap = onset_min_gap

        self._env = 0.0          # fast envelope (the "is she talking + how loud")
        self._env_slow = 0.0     # slow baseline for onset comparison
        self._onset_pending = 0.0
        self._last_onset_t = 0.0
        self._lock = threading.Lock()
        self._stream = None
        self._running = False

    def start(self) -> bool:
        """Open the input stream. Returns False if audio is unavailable or the
        device can't be opened - caller should carry on with audio_env=0."""
        if not AUDIO_AVAILABLE:
            print("[AudioListener] sounddevice/numpy unavailable - motion will be idle-only")
            return False

        if self.device is None:
            self.device = resolve_device()
            if self.device is None:
                print(f"[AudioListener] no '{DEFAULT_DEVICE_MATCH}' device found - "
                      "motion will be idle-only (is VB-Cable installed / Alice routed to it?)")
                return False

        if self.samplerate is None:
            try:
                self.samplerate = int(sd.query_devices(self.device)["default_samplerate"])
            except Exception:
                self.samplerate = 48000

        blocksize = max(256, int(self.samplerate * 0.02))  # ~20ms blocks
        try:
            self._stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.samplerate,
                blocksize=blocksize,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[AudioListener] failed to open stream: {e} - motion will be idle-only")
            self._stream = None
            return False

        self._running = True
        print(f"[AudioListener] listening @ {self.samplerate}Hz, block {blocksize} "
              f"(gain={self.gain})")
        return True

    def _callback(self, indata, frames, time_info, status):
        # PortAudio thread. Keep it tiny.
        try:
            rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
        except Exception:
            return
        target = min(1.0, rms * self.gain)

        with self._lock:
            # Asymmetric smoothing: snap up on speech, ease down in gaps.
            coef = self.attack if target > self._env else self.release
            self._env += (target - self._env) * coef
            # Slow baseline tracks the running level for onset comparison.
            self._env_slow += (self._env - self._env_slow) * 0.04

            jump = self._env - self._env_slow
            now = time.monotonic()
            if jump > self.onset_thr and (now - self._last_onset_t) > self.onset_min_gap:
                self._onset_pending = min(1.5, jump / self.onset_thr)
                self._last_onset_t = now

    def get_env(self) -> float:
        """Current 0..1 speech envelope. 0 = silent, ~1 = loud/emphatic."""
        with self._lock:
            return self._env

    def poll_onset(self) -> float:
        """Return onset intensity since last poll (0 if none), then clear it.
        Call once per frame; feed the return into motion.trigger_onset()."""
        with self._lock:
            o = self._onset_pending
            self._onset_pending = 0.0
            return o

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio envelope meter (verify capture)")
    parser.add_argument("--device", default=None,
                        help=f"Device name substring (default: '{DEFAULT_DEVICE_MATCH}'). "
                             "Use 'Microphone' to test by talking into your mic.")
    parser.add_argument("--gain", type=float, default=7.0)
    args = parser.parse_args()

    dev = resolve_device(args.device) if args.device else None
    listener = AudioListener(device=dev, gain=args.gain)
    if not listener.start():
        raise SystemExit("Could not start audio capture.")

    print("\nTalking meter - make noise into the captured device. Ctrl+C to stop.\n")
    try:
        while True:
            env = listener.get_env()
            onset = listener.poll_onset()
            bar = "#" * int(env * 50)
            mark = "  <ONSET>" if onset else ""
            print(f"\renv {env:.3f} |{bar:<50}|{mark}   ", end="", flush=True)
            time.sleep(1 / 40)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        listener.stop()
