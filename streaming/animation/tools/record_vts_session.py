#!/usr/bin/env python3
"""
Record VTube Studio parameters while you pilot the model.

This records:
1. VTS parameter values (what the model is actually doing)
2. Audio from your microphone

Use this to collect data - pilot your avatar naturally,
then train a model to predict the VTS params from audio.

Usage:
    python3 record_vts_session.py output_folder [duration_seconds]

Example:
    python3 record_vts_session.py training_session_01 60

Press Ctrl+C to stop recording early.
"""

import asyncio
import json
import sys
import time
import wave
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not installed. Run: pip3 install numpy")

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not installed. Run: pip3 install sounddevice")

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("Warning: websockets not installed. Run: pip3 install websockets")


class VTSRecorder:
    def __init__(self, output_dir: str, vts_host: str = "localhost", vts_port: int = 8001):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vts_host = vts_host
        self.vts_port = vts_port
        self.ws = None

        self.sample_rate = 16000
        self.fps = 30  # Record at 30fps to match typical VTS update rate

        # Recording buffers
        self.param_frames = []
        self.audio_buffer = []
        self.param_names = None

    async def connect(self):
        """Connect and authenticate with VTS."""
        uri = f"ws://{self.vts_host}:{self.vts_port}"
        print(f"Connecting to VTS at {uri}...")

        self.ws = await websockets.connect(uri)

        # Request auth token
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth-token",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": "VTS Recorder",
                "pluginDeveloper": "Alice"
            }
        }))
        resp = json.loads(await self.ws.recv())
        token = resp.get("data", {}).get("authenticationToken")

        if not token:
            print(f"Failed to get token: {resp}")
            return False

        # Authenticate
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": "VTS Recorder",
                "pluginDeveloper": "Alice",
                "authenticationToken": token
            }
        }))
        resp = json.loads(await self.ws.recv())

        if resp.get("data", {}).get("authenticated"):
            print("Connected to VTS!")
            return True
        else:
            print(f"Auth failed: {resp}")
            return False

    async def get_input_parameters(self):
        """Get list of Live2D parameters from the model (works with iPhone tracking)."""
        all_params = []

        # Get Live2D parameters - these update when iPhone is tracking
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "live2d",
            "messageType": "Live2DParameterListRequest"
        }))
        resp = json.loads(await self.ws.recv())

        all_params = resp.get("data", {}).get("parameters", [])

        if all_params:
            print(f"Found {len(all_params)} Live2D parameters")
        else:
            # Fallback to input parameters
            await self.ws.send(json.dumps({
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "input",
                "messageType": "InputParameterListRequest"
            }))
            resp = json.loads(await self.ws.recv())

            params = resp.get("data", {}).get("defaultParameters", [])
            custom = resp.get("data", {}).get("customParameters", [])
            all_params = params + custom

            if all_params:
                print(f"Found {len(all_params)} Input parameters (fallback)")
            else:
                print("WARNING: No parameters found!")

        # Debug output
        if all_params:
            print("Sample parameters:")
            for p in all_params[:8]:
                print(f"  {p['name']}: value={p.get('value', 'N/A')}")

        return all_params

    def get_default_param_names(self):
        """Fallback list of common VTS input parameters."""
        return [
            # Face angles
            "FaceAngleX", "FaceAngleY", "FaceAngleZ",
            # Eyes
            "EyeOpenLeft", "EyeOpenRight",
            "EyeLeftX", "EyeLeftY", "EyeRightX", "EyeRightY",
            # Mouth
            "MouthOpen", "MouthSmile",
            "MouthFunnel", "MouthPucker",
            # Brows
            "BrowLeftY", "BrowRightY",
            # Other
            "CheekPuff", "TongueOut",
            "VoiceFrequency", "VoiceVolume", "VoiceVolumePlusMouthOpen"
        ]

    async def get_live_parameters(self):
        """Get current values of Live2D parameters (works with iPhone tracking)."""
        result = {}

        # Get Live2D parameters with current values
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "live2d-params",
            "messageType": "Live2DParameterListRequest"
        }))
        resp = json.loads(await self.ws.recv())

        for p in resp.get("data", {}).get("parameters", []):
            result[p["name"]] = p.get("value", 0)

        return result

    async def record(self, duration: float = 60.0):
        """Record VTS parameters and audio for specified duration."""
        if not NUMPY_AVAILABLE or not AUDIO_AVAILABLE:
            print("Missing required libraries!")
            return

        # Get parameter list
        param_info = await self.get_input_parameters()

        if param_info:
            self.param_names = [p["name"] for p in param_info]
        else:
            # Fallback to default parameter names
            self.param_names = self.get_default_param_names()
            print(f"Using {len(self.param_names)} default parameter names")

        print(f"\nRecording parameters: {', '.join(self.param_names[:10])}...")
        if len(self.param_names) > 10:
            print(f"  ... and {len(self.param_names) - 10} more")

        # Setup audio recording
        audio_queue = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            audio_queue.put_nowait(indata.copy())

        chunk_samples = int(self.sample_rate / self.fps)  # Samples per frame

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_samples,
            callback=audio_callback
        )

        print(f"\n{'='*50}")
        print(f"RECORDING for {duration:.0f} seconds")
        print(f"Pilot your avatar naturally - talk, emote, move!")
        print(f"Press Ctrl+C to stop early")
        print(f"{'='*50}\n")

        frame_time = 1.0 / self.fps
        start_time = time.time()
        frame_count = 0

        with stream:
            try:
                while (time.time() - start_time) < duration:
                    frame_start = time.time()

                    # Get VTS parameters
                    params = await self.get_live_parameters()
                    self.param_frames.append(params)

                    # Get audio chunk
                    try:
                        audio_chunk = await asyncio.wait_for(
                            audio_queue.get(),
                            timeout=frame_time
                        )
                        self.audio_buffer.append(audio_chunk)
                    except asyncio.TimeoutError:
                        # No audio this frame, append silence
                        self.audio_buffer.append(np.zeros((chunk_samples, 1), dtype=np.float32))

                    frame_count += 1
                    elapsed = time.time() - start_time

                    # Progress
                    if frame_count % 30 == 0:
                        # Show some live values
                        face_y = params.get("FaceAngleY", 0)
                        mouth = params.get("MouthOpen", 0)
                        print(f"\rTime: {elapsed:.1f}s | Frames: {frame_count} | "
                              f"FaceAngleY: {face_y:6.1f} | MouthOpen: {mouth:.2f}", end="")

                    # Maintain frame rate
                    frame_elapsed = time.time() - frame_start
                    if frame_elapsed < frame_time:
                        await asyncio.sleep(frame_time - frame_elapsed)

            except KeyboardInterrupt:
                print("\n\nStopping early...")

        print(f"\n\nRecorded {frame_count} frames ({frame_count/self.fps:.1f} seconds)")

        # Save data
        await self.save_data()

    async def save_data(self):
        """Save recorded data to files."""
        print("\nSaving data...")

        # Save audio
        audio_path = self.output_dir / "audio.wav"
        audio_data = np.concatenate(self.audio_buffer, axis=0)

        # Convert to int16 for WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(str(audio_path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        print(f"  Audio: {audio_path} ({len(audio_data)/self.sample_rate:.1f}s)")

        # Save parameters as CSV
        csv_path = self.output_dir / "vts_params.csv"

        # Get all unique parameter names
        all_names = set()
        for frame in self.param_frames:
            all_names.update(frame.keys())
        all_names = sorted(list(all_names))

        with open(csv_path, 'w') as f:
            # Header
            f.write("frame," + ",".join(all_names) + "\n")

            # Data
            for i, frame in enumerate(self.param_frames):
                values = [str(frame.get(name, 0)) for name in all_names]
                f.write(f"{i}," + ",".join(values) + "\n")

        print(f"  Parameters: {csv_path} ({len(self.param_frames)} frames, {len(all_names)} params)")

        # Save metadata
        meta_path = self.output_dir / "recording_info.json"
        meta = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": len(self.param_frames) / self.fps,
            "fps": self.fps,
            "n_frames": len(self.param_frames),
            "n_parameters": len(all_names),
            "parameter_names": all_names,
            "sample_rate": self.sample_rate
        }

        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  Metadata: {meta_path}")

        # Also save as NPZ for easy loading
        npz_path = self.output_dir / "training_data.npz"

        # Build parameter array
        param_array = np.zeros((len(self.param_frames), len(all_names)), dtype=np.float32)
        for i, frame in enumerate(self.param_frames):
            for j, name in enumerate(all_names):
                param_array[i, j] = frame.get(name, 0)

        np.savez(
            npz_path,
            vts_params=param_array,
            param_names=np.array(all_names),
            audio=audio_data.flatten(),
            sample_rate=self.sample_rate,
            fps=self.fps
        )

        print(f"  Training data: {npz_path}")
        print(f"\nDone! Use this data to train a model that predicts VTS params from audio.")


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    output_dir = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

    vts_host = sys.argv[3] if len(sys.argv) > 3 else "localhost"
    vts_port = int(sys.argv[4]) if len(sys.argv) > 4 else 8001

    recorder = VTSRecorder(output_dir, vts_host, vts_port)

    if not await recorder.connect():
        print("Failed to connect to VTS")
        sys.exit(1)

    await recorder.record(duration)

    if recorder.ws:
        await recorder.ws.close()


if __name__ == "__main__":
    if not WEBSOCKETS_AVAILABLE:
        print("websockets required: pip3 install websockets")
        sys.exit(1)

    asyncio.run(main())
