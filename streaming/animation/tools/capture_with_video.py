#!/usr/bin/env python3
"""
Capture VTS parameters + video of avatar simultaneously.

Records:
1. VTS parameters (what's driving the model)
2. Screen region showing the avatar (what it looks like)

This gives training data for a VLM to learn:
"When the avatar looks like THIS, the parameters are THESE"

Usage:
    python3 capture_with_video.py session_01
    python3 capture_with_video.py session_01 --region 0,0,800,600  # custom capture region
    python3 capture_with_video.py session_01 --window "VTube Studio"  # capture specific window

Press Ctrl+C to stop.
"""

import asyncio
import argparse
import json
import time
import csv
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue

try:
    import websockets
except ImportError:
    print("Install: pip3 install websockets")
    exit(1)

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Install: pip3 install opencv-python numpy")

try:
    from mss import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("Install: pip3 install mss")

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


class VideoCapture:
    """Captures screen region in a separate thread."""

    def __init__(self, region=None, fps=30):
        self.region = region  # (x, y, width, height) or None for full screen
        self.fps = fps
        self.frames = []
        self.timestamps = []
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _capture_loop(self):
        frame_time = 1.0 / self.fps
        start_time = time.time()

        with mss() as sct:
            # Setup capture region
            if self.region:
                monitor = {
                    "left": self.region[0],
                    "top": self.region[1],
                    "width": self.region[2],
                    "height": self.region[3]
                }
            else:
                # Full primary monitor
                monitor = sct.monitors[1]

            print(f"Capturing region: {monitor}")

            while self.running:
                loop_start = time.time()

                # Capture
                img = sct.grab(monitor)
                frame = np.array(img)
                # Convert BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                self.frames.append(frame)
                self.timestamps.append(time.time() - start_time)

                # Maintain fps
                elapsed = time.time() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)


class VTSCapture:
    def __init__(self, host="localhost", port=8001):
        self.host = host
        self.port = port
        self.ws = None
        self.recording = []
        self.param_names = []

    async def connect(self):
        uri = f"ws://{self.host}:{self.port}"
        print(f"Connecting to VTS at {uri}...")

        self.ws = await websockets.connect(uri)

        # Auth
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth-token",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": "Motion Capture",
                "pluginDeveloper": "Alice"
            }
        }))
        resp = json.loads(await self.ws.recv())
        token = resp.get("data", {}).get("authenticationToken")

        if not token:
            print("Check VTS - allow the plugin!")
            return False

        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": "Motion Capture",
                "pluginDeveloper": "Alice",
                "authenticationToken": token
            }
        }))
        resp = json.loads(await self.ws.recv())

        if resp.get("data", {}).get("authenticated"):
            print("Connected to VTS!")
            return True
        return False

    async def get_params(self):
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "params",
            "messageType": "Live2DParameterListRequest"
        }))
        resp = json.loads(await self.ws.recv())

        params = {}
        for p in resp.get("data", {}).get("parameters", []):
            params[p["name"]] = p.get("value", 0)
        return params


async def main(args):
    if not MSS_AVAILABLE or not CV2_AVAILABLE:
        print("Missing dependencies!")
        return

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse region
    region = None
    if args.region:
        parts = [int(x) for x in args.region.split(",")]
        if len(parts) == 4:
            region = tuple(parts)

    # Setup captures
    vts = VTSCapture(args.host, args.port)
    video = VideoCapture(region=region, fps=args.fps)

    if not await vts.connect():
        return

    # Get param names
    params = await vts.get_params()
    vts.param_names = sorted(params.keys())
    print(f"Tracking {len(vts.param_names)} VTS parameters")

    # Start video capture
    video.start()
    print(f"\nCapturing at {args.fps}fps")
    print("=" * 60)
    print("Pilot your avatar! Press Ctrl+C to stop.")
    print("=" * 60)

    frame_time = 1.0 / args.fps
    start_time = time.time()
    frame_num = 0

    import signal
    stop_flag = False

    def handle_stop(sig, frame):
        nonlocal stop_flag
        stop_flag = True
        print("\n\nStopping (saving data)...")

    signal.signal(signal.SIGINT, handle_stop)

    while not stop_flag:
        loop_start = time.time()

        # Get VTS params
        try:
            params = await vts.get_params()
        except:
            break

        vts.recording.append({
            "frame": frame_num,
            "time": time.time() - start_time,
            **params
        })

        # Display progress
        elapsed = time.time() - start_time
        n_video = len(video.frames)
        print(f"\r[{elapsed:6.1f}s] VTS frames: {frame_num} | Video frames: {n_video}", end="", flush=True)

        frame_num += 1

        # Maintain fps
        elapsed_frame = time.time() - loop_start
        if elapsed_frame < frame_time:
            await asyncio.sleep(frame_time - elapsed_frame)

    # Stop video capture
    video.stop()

    # Check we have data
    if not vts.recording:
        print("\nNo data captured!")
        return

    # Save everything
    print(f"\nSaving to {output_path}/")

    # Save VTS params
    csv_path = output_path / "params.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time"] + vts.param_names)
        writer.writeheader()
        writer.writerows(vts.recording)
    print(f"  params.csv - {len(vts.recording)} frames, {len(vts.param_names)} params")

    # Save video
    if video.frames:
        video_path = output_path / "avatar.mp4"
        h, w = video.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))

        for frame in video.frames:
            out.write(frame)
        out.release()
        print(f"  avatar.mp4 - {len(video.frames)} frames, {w}x{h}")

        # Also save frames as images (for VLM training)
        frames_dir = output_path / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Save every Nth frame to avoid too many files
        step = max(1, len(video.frames) // 500)  # Max ~500 frames
        saved = 0
        for i in range(0, len(video.frames), step):
            cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), video.frames[i])
            saved += 1
        print(f"  frames/ - {saved} images (every {step} frames)")

    # Save metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "vts_frames": len(vts.recording),
        "video_frames": len(video.frames),
        "duration": vts.recording[-1]["time"] if vts.recording else 0,
        "fps": args.fps,
        "parameters": vts.param_names,
        "region": region
    }
    with open(output_path / "info.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  info.json - metadata")

    print("\nDone! You now have:")
    print("  1. params.csv - what the VTS params were each frame")
    print("  2. avatar.mp4 - what the avatar looked like")
    print("  3. frames/    - individual frames for VLM training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture VTS params + avatar video")
    parser.add_argument("output", help="Output folder name")
    parser.add_argument("--host", default="localhost", help="VTS host")
    parser.add_argument("--port", type=int, default=8001, help="VTS port")
    parser.add_argument("--fps", type=int, default=30, help="Capture rate")
    parser.add_argument("--region", help="Screen region: x,y,width,height")
    args = parser.parse_args()

    asyncio.run(main(args))
