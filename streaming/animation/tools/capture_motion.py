#!/usr/bin/env python3
"""
Capture VTS parameters live while you pilot.

Usage:
    python3 capture_motion.py                    # Live view only
    python3 capture_motion.py --record session1  # Record to file
    python3 capture_motion.py --host 192.168.1.5 # Different VTS host

Press Ctrl+C to stop.
"""

import asyncio
import argparse
import json
import time
import csv
from datetime import datetime
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Install: pip3 install websockets")
    exit(1)


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
            print("Connected!\n")
            return True
        return False

    async def get_params(self):
        """Get current Live2D parameter values."""
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

    async def run(self, record_path=None, fps=30):
        """Main capture loop."""
        if not await self.connect():
            return

        # Get initial params to know what we're tracking
        params = await self.get_params()
        self.param_names = sorted(params.keys())

        print(f"Tracking {len(self.param_names)} parameters at {fps}fps")
        print("=" * 60)

        # Key params to show live
        show_params = [
            "FaceAngleX", "FaceAngleY", "FaceAngleZ",
            "EyeOpenLeft", "EyeOpenRight",
            "MouthOpen", "MouthSmile"
        ]
        show_params = [p for p in show_params if p in self.param_names]

        frame_time = 1.0 / fps
        start_time = time.time()
        frame = 0

        try:
            while True:
                loop_start = time.time()

                params = await self.get_params()

                # Record
                if record_path:
                    self.recording.append({
                        "frame": frame,
                        "time": time.time() - start_time,
                        **params
                    })

                # Display
                elapsed = time.time() - start_time
                display = f"\r[{elapsed:6.1f}s] "
                for name in show_params:
                    val = params.get(name, 0)
                    short = name.replace("FaceAngle", "").replace("EyeOpen", "Eye")
                    display += f"{short}:{val:6.1f} "

                print(display, end="", flush=True)

                frame += 1

                # Maintain fps
                elapsed_frame = time.time() - loop_start
                if elapsed_frame < frame_time:
                    await asyncio.sleep(frame_time - elapsed_frame)

        except KeyboardInterrupt:
            print("\n\nStopped.")

        # Save if recording
        if record_path and self.recording:
            self.save(record_path)

    def save(self, path):
        """Save recorded data."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # CSV
        csv_path = path / "params.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "time"] + self.param_names)
            writer.writeheader()
            writer.writerows(self.recording)

        # Metadata
        meta = {
            "timestamp": datetime.now().isoformat(),
            "frames": len(self.recording),
            "duration": self.recording[-1]["time"] if self.recording else 0,
            "parameters": self.param_names
        }
        with open(path / "info.json", 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\nSaved {len(self.recording)} frames to {path}/")
        print(f"  params.csv  - {len(self.param_names)} parameters")
        print(f"  info.json   - metadata")


def main():
    parser = argparse.ArgumentParser(description="Capture VTS parameters live")
    parser.add_argument("--record", "-r", help="Record to this folder")
    parser.add_argument("--host", default="localhost", help="VTS host")
    parser.add_argument("--port", type=int, default=8001, help="VTS port")
    parser.add_argument("--fps", type=int, default=30, help="Capture rate")
    args = parser.parse_args()

    capture = VTSCapture(args.host, args.port)
    asyncio.run(capture.run(args.record, args.fps))


if __name__ == "__main__":
    main()
