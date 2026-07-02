#!/usr/bin/env python3
"""
VLM Motion Corrector - Visual feedback loop for avatar animation.

Workflow:
1. TRAIN: Load reference frames from your piloting sessions
2. RUN: VLM watches avatar live, compares to references, outputs corrections

The VLM learns what "correct" movement looks like from YOUR piloting,
then corrects the animation engine to match.

Usage:
    # First, capture some reference sessions with capture_with_video.py
    # Then use those to correct live animation:

    python3 vlm_motion_corrector.py --references session_01/frames --live

    # Or analyze a recorded session:
    python3 vlm_motion_corrector.py --references session_01/frames --analyze session_02/frames
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

try:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_image
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Install: pip3 install mlx-vlm")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from mss import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import websockets
    import asyncio
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False


MODEL_ID = "vision-language-model"


class VLMCorrector:
    """Uses VLM to compare current avatar to reference and suggest corrections."""

    def __init__(self, reference_dir: str):
        if not MLX_AVAILABLE:
            raise RuntimeError("mlx-vlm required: pip3 install mlx-vlm")

        print(f"Loading VLM: {MODEL_ID}")
        self.model, self.processor = load(MODEL_ID)

        # Load reference frames
        self.references = self._load_references(reference_dir)
        print(f"Loaded {len(self.references)} reference frames")

    def _load_references(self, ref_dir: str) -> List[str]:
        """Load reference frame paths."""
        ref_path = Path(ref_dir)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference dir not found: {ref_dir}")

        frames = sorted(ref_path.glob("*.jpg")) + sorted(ref_path.glob("*.png"))
        return [str(f) for f in frames]

    def _query(self, image_path: str, prompt: str, max_tokens: int = 300) -> str:
        """Query VLM with image."""
        image = load_image(image_path)

        formatted = apply_chat_template(
            self.processor,
            config=self.model.config,
            prompt=prompt,
            num_images=1
        )

        output = generate(
            self.model,
            self.processor,
            image,
            formatted,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temp for more consistent outputs
            verbose=False
        )
        return output.strip()

    def get_reference_sample(self, n: int = 3) -> List[str]:
        """Get random sample of reference frames for comparison."""
        if len(self.references) <= n:
            return self.references
        return random.sample(self.references, n)

    def analyze_frame(self, frame_path: str) -> Dict:
        """
        Analyze a single frame - extract pose/expression info.
        """
        prompt = """Look at this avatar and describe its current state.

Output JSON only:
{
    "head_x": estimate horizontal rotation (-30 to 30),
    "head_y": estimate vertical tilt (-20 to 20),
    "head_z": estimate roll/tilt (-15 to 15),
    "eyes_open": 0.0 to 1.0,
    "mouth_open": 0.0 to 1.0,
    "expression": "neutral/happy/surprised/etc",
    "energy": "low/medium/high"
}"""

        response = self._query(frame_path, prompt, max_tokens=200)

        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass

        return {"raw": response, "error": True}

    def compare_to_reference(self, current_frame_path: str, reference_frame_path: str) -> Dict:
        """
        Compare current avatar state to a reference frame.
        Returns suggested parameter adjustments.
        """
        # First analyze the reference
        ref_prompt = """Look at this avatar (this is the REFERENCE - correct movement).
Describe the head position, expression, and energy level briefly."""

        ref_analysis = self._query(reference_frame_path, ref_prompt, max_tokens=150)

        # Now analyze current with reference context
        compare_prompt = f"""Look at this avatar (CURRENT state).

The REFERENCE (correct) state was:
{ref_analysis}

Compare CURRENT to REFERENCE. What needs to change?

Output JSON with adjustments needed:
{{
    "head_x_adjust": delta to add (-10 to 10),
    "head_y_adjust": delta to add (-10 to 10),
    "eyes_adjust": delta (-0.3 to 0.3),
    "mouth_adjust": delta (-0.3 to 0.3),
    "overall_match": 0.0 to 1.0 (how close to reference),
    "main_issue": "what's most wrong"
}}"""

        response = self._query(current_frame_path, compare_prompt, max_tokens=200)

        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass

        return {"raw": response, "error": True}

    def get_correction(self, current_frame_path: str) -> Dict:
        """
        Get parameter corrections for current frame based on reference set.
        """
        # Pick a random reference to compare against
        ref = random.choice(self.references)

        correction = self.compare_to_reference(current_frame_path, ref)

        # Map to VTS parameter deltas
        vts_deltas = {}

        if "error" not in correction:
            if "head_x_adjust" in correction:
                vts_deltas["FaceAngleX"] = float(correction.get("head_x_adjust", 0))
            if "head_y_adjust" in correction:
                vts_deltas["FaceAngleY"] = float(correction.get("head_y_adjust", 0))
            if "eyes_adjust" in correction:
                delta = float(correction.get("eyes_adjust", 0))
                vts_deltas["EyeOpenLeft"] = delta
                vts_deltas["EyeOpenRight"] = delta
            if "mouth_adjust" in correction:
                vts_deltas["MouthOpen"] = float(correction.get("mouth_adjust", 0))

        return {
            "vts_deltas": vts_deltas,
            "match_score": correction.get("overall_match", 0),
            "issue": correction.get("main_issue", "unknown"),
            "raw": correction
        }


class LiveCorrector:
    """Live correction loop - captures screen, runs VLM, sends corrections to VTS."""

    def __init__(self, corrector: VLMCorrector, region: Tuple[int, int, int, int] = None,
                 vts_host: str = "localhost", vts_port: int = 8001):
        self.corrector = corrector
        self.region = region
        self.vts_host = vts_host
        self.vts_port = vts_port
        self.ws = None

    async def connect_vts(self):
        """Connect to VTS for sending corrections."""
        uri = f"ws://{self.vts_host}:{self.vts_port}"
        self.ws = await websockets.connect(uri)

        # Auth
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth-token",
            "messageType": "AuthenticationTokenRequest",
            "data": {"pluginName": "VLM Corrector", "pluginDeveloper": "Alice"}
        }))
        resp = json.loads(await self.ws.recv())
        token = resp.get("data", {}).get("authenticationToken")

        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth",
            "messageType": "AuthenticationRequest",
            "data": {"pluginName": "VLM Corrector", "pluginDeveloper": "Alice",
                     "authenticationToken": token}
        }))
        resp = json.loads(await self.ws.recv())
        return resp.get("data", {}).get("authenticated", False)

    async def send_corrections(self, deltas: Dict[str, float]):
        """Send parameter corrections to VTS."""
        if not self.ws or not deltas:
            return

        params = [{"id": k, "value": v, "weight": 0.5} for k, v in deltas.items()]

        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "inject",
            "messageType": "InjectParameterDataRequest",
            "data": {"parameterValues": params}
        }))
        await self.ws.recv()  # Consume response

    def capture_frame(self) -> str:
        """Capture current screen region, save to temp file."""
        import tempfile

        with mss() as sct:
            if self.region:
                monitor = {
                    "left": self.region[0], "top": self.region[1],
                    "width": self.region[2], "height": self.region[3]
                }
            else:
                monitor = sct.monitors[1]

            img = sct.grab(monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            path = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(path, frame)
            return path

    async def run(self, interval: float = 1.0):
        """
        Main correction loop.

        Args:
            interval: Seconds between corrections (VLM is slow, 1-2s realistic)
        """
        print("Connecting to VTS...")
        if not await self.connect_vts():
            print("Failed to connect to VTS")
            return

        print(f"\nRunning correction loop (every {interval}s)")
        print("=" * 50)
        print("Watching avatar and applying corrections...")
        print("Press Ctrl+C to stop")
        print("=" * 50)

        try:
            while True:
                loop_start = time.time()

                # Capture current frame
                frame_path = self.capture_frame()

                # Get correction from VLM
                correction = self.corrector.get_correction(frame_path)

                # Apply to VTS
                await self.send_corrections(correction["vts_deltas"])

                # Display
                match = correction.get("match_score", 0)
                issue = correction.get("issue", "")[:40]
                deltas = correction.get("vts_deltas", {})
                print(f"\rMatch: {match:.0%} | Issue: {issue:<40} | Deltas: {deltas}", end="")

                # Cleanup temp file
                try:
                    os.remove(frame_path)
                except:
                    pass

                # Wait for next iteration
                elapsed = time.time() - loop_start
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)

        except KeyboardInterrupt:
            print("\n\nStopped.")

        if self.ws:
            await self.ws.close()


def analyze_session(corrector: VLMCorrector, frames_dir: str):
    """Analyze a recorded session (not live)."""
    frames_path = Path(frames_dir)
    frames = sorted(frames_path.glob("*.jpg")) + sorted(frames_path.glob("*.png"))

    print(f"Analyzing {len(frames)} frames from {frames_dir}")

    results = []
    for i, frame in enumerate(frames[:20]):  # Limit for speed
        print(f"\rAnalyzing frame {i+1}/{min(len(frames), 20)}...", end="")
        result = corrector.analyze_frame(str(frame))
        result["frame"] = frame.name
        results.append(result)

    print("\n\nResults:")
    for r in results:
        print(f"  {r.get('frame', '?')}: {r.get('expression', '?')} - energy: {r.get('energy', '?')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="VLM Motion Corrector")
    parser.add_argument("--references", "-r", required=True,
                        help="Path to reference frames (from your piloting)")
    parser.add_argument("--live", action="store_true",
                        help="Run live correction loop")
    parser.add_argument("--analyze", help="Analyze recorded frames instead of live")
    parser.add_argument("--region", help="Screen capture region: x,y,w,h")
    parser.add_argument("--interval", type=float, default=1.5,
                        help="Seconds between corrections (default: 1.5)")
    parser.add_argument("--host", default="localhost", help="VTS host")
    parser.add_argument("--port", type=int, default=8001, help="VTS port")
    args = parser.parse_args()

    if not MLX_AVAILABLE:
        print("Error: mlx-vlm required. Install: pip3 install mlx-vlm")
        return

    # Load VLM and references
    corrector = VLMCorrector(args.references)

    if args.analyze:
        # Analyze recorded session
        analyze_session(corrector, args.analyze)

    elif args.live:
        # Live correction
        if not MSS_AVAILABLE or not CV2_AVAILABLE or not WS_AVAILABLE:
            print("Missing deps: pip3 install mss opencv-python websockets")
            return

        region = None
        if args.region:
            region = tuple(int(x) for x in args.region.split(","))

        live = LiveCorrector(corrector, region, args.host, args.port)
        asyncio.run(live.run(args.interval))

    else:
        print("Specify --live or --analyze")
        parser.print_help()


if __name__ == "__main__":
    main()
