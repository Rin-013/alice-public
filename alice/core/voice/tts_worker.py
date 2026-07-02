"""
TTS Worker — Spawns the TTS model in an isolated .venv_tts subprocess.

Process isolation allows TTS to use a different transformers version while the
main process uses another. Communication is via stdin/stdout JSON lines.
Audio playback happens entirely inside the subprocess.

Falls back to multiprocessing.Process (same-venv) if .venv_tts doesn't exist.
"""
import json
import os
import queue as _queue
import subprocess
import sys
import threading
import time


_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
_VENV_PYTHON = os.path.join(_BASE_DIR, ".venv_tts", "Scripts", "python.exe")
_SUBPROCESS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_subprocess.py")


def _log(msg):
    sys.stderr.write(f"[TTS-worker] {msg}\n")
    sys.stderr.flush()


class TTSWorker:
    """Main-process handle to the TTS subprocess."""

    def __init__(self, device: str = "cuda:0"):
        venv_python = _VENV_PYTHON
        if not os.path.exists(venv_python):
            # Fallback: try Unix-style path
            venv_python = os.path.join(_BASE_DIR, ".venv_tts", "bin", "python")
        if not os.path.exists(venv_python):
            raise RuntimeError(
                f"TTS venv not found at {_VENV_PYTHON}. "
                "Create it with: python -m venv .venv_tts && "
                ".venv_tts/Scripts/pip install -e vendor/tts-streaming sounddevice numpy"
            )

        _log(f"Spawning TTS subprocess: {venv_python}")
        self._process = subprocess.Popen(
            [venv_python, _SUBPROCESS_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Windows pipes can't be polled with select/selectors (only sockets).
        # Use a background thread to drain stdout into a queue, and Queue.get
        # for timed waits in _recv.
        self._stdout_queue: _queue.Queue = _queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True
        )
        self._reader_thread.start()

        # Wait for ready signal. Cold start = model load (~20s) + 3-bucket
        # warmup with torch.compile + CUDA graph capture (~120-200s on first
        # run, faster once torchinductor cache is populated). 300s covers
        # the worst case; warm starts complete in ~30s.
        status = self._recv(timeout=300)
        if status is None:
            self._process.kill()
            raise RuntimeError("TTS subprocess failed to start (no response)")
        if status.get("status") == "error":
            self._process.kill()
            raise RuntimeError(f"TTS subprocess failed: {status.get('msg')}")

        self.compiled = status.get("compiled", False)
        _log(f"TTS subprocess ready, compiled={self.compiled}")

    def _send(self, obj):
        """Write JSON line to subprocess stdin."""
        try:
            self._process.stdin.write(json.dumps(obj) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"TTS subprocess pipe broken: {e}")

    def _reader_loop(self):
        """Background thread: drain subprocess stdout into a queue."""
        try:
            for line in iter(self._process.stdout.readline, ''):
                if not line:
                    break
                self._stdout_queue.put(line)
        except Exception as e:
            _log(f"reader thread died: {e}")
        finally:
            self._stdout_queue.put(None)  # sentinel for EOF

    def _recv(self, timeout=60):
        """Read JSON line from subprocess stdout. Returns None on timeout/EOF.

        Vendor libraries occasionally emit log lines (e.g. tokenizer init) to
        stdout instead of stderr. Skip those and keep reading until we hit a
        real JSON message, the timeout elapses, or the pipe closes.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._stdout_queue.get(timeout=remaining)
            except _queue.Empty:
                return None
            if line is None:
                return None  # EOF sentinel
            stripped = line.strip()
            if not stripped:
                continue
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                _log(f"recv: skipping non-JSON line from subprocess: {stripped!r}")
                continue

    def speak(self, text: str, emotion: str = "neutral"):
        """Send text to TTS subprocess and block until generation completes.

        Subprocess returns "done" once generation finishes; playback runs in a
        persistent background thread on the subprocess side. Call drain() if
        you need to wait for audio to actually play out.

        Timeout sized for cold-start: first real generation can hit a
        torch.compile/CUDA-graph recapture for shapes not covered by warmup
        (~90s observed). Steady-state turns return in seconds.
        """
        self._send({"cmd": "speak", "text": text, "emotion": emotion})
        status = self._recv(timeout=180)
        if status is None:
            raise RuntimeError("TTS subprocess timed out or died")
        if status.get("status") == "error":
            raise RuntimeError(f"TTS speak failed: {status.get('msg')}")
        return status

    def drain(self, timeout: float = 180.0):
        """Wait for any in-flight playback to finish. Use after speak() when
        you need to know the audio has actually played out (e.g. before
        printing the next prompt)."""
        try:
            self._send({"cmd": "drain"})
            self._recv(timeout=timeout)
        except Exception as e:
            _log(f"drain failed: {e}")

    def shutdown(self):
        """Stop the TTS subprocess."""
        try:
            self._send({"cmd": "shutdown"})
            self._process.wait(timeout=10)
        except Exception:
            pass
        if self._process.poll() is None:
            self._process.kill()
