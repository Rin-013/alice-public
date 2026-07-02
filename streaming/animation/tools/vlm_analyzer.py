#!/usr/bin/env python3
"""
VLM Animation Analyzer (MLX - Apple Silicon)
=============================================
Uses a vision-language model via MLX to analyze avatar videos and improve motion quality.

Install:
    pip install mlx-vlm opencv-python

Usage:
    # Analyze a video for movement quality
    python vlm_analyzer.py analyze video.mp4

    # Label emotion/intensity from frame or video
    python vlm_analyzer.py label frame.png

    # Compare generated vs reference animation
    python vlm_analyzer.py compare generated.mp4 reference.mp4

    # Extract movement descriptions for training data
    python vlm_analyzer.py describe video.mp4 --output descriptions.json
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Check MLX dependencies
try:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_image
    MLX_AVAILABLE = True
except ImportError as e:
    MLX_AVAILABLE = False
    MLX_ERROR = str(e)

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


MODEL_ID = "vision-language-model"


class VLMAnalyzer:
    """VLM-based analyzer for avatar animation using MLX on Apple Silicon."""

    def __init__(self, model_id: str = MODEL_ID):
        if not MLX_AVAILABLE:
            raise RuntimeError(f"mlx-vlm not available: {MLX_ERROR}\nInstall with: pip install mlx-vlm")
        if not CV2_AVAILABLE:
            raise RuntimeError("opencv not available. Install with: pip install opencv-python")

        print(f"Loading {model_id}...")
        self.model, self.processor = load(model_id)
        self.model_id = model_id
        print("Model loaded!")

    def _extract_frames(self, video_path: str, num_frames: int = 4,
                        start_sec: float = 0, end_sec: float = None) -> List[str]:
        """Extract frames from video, save as temp files, return paths."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if end_sec is None:
            end_sec = duration

        start_frame = int(start_sec * fps)
        end_frame = int(min(end_sec, duration) * fps)

        frame_indices = np.linspace(start_frame, max(start_frame + 1, end_frame - 1),
                                     num_frames, dtype=int)

        frame_paths = []
        temp_dir = tempfile.mkdtemp()

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
                cv2.imwrite(path, frame)
                frame_paths.append(path)

        cap.release()
        return frame_paths

    def _query(self, image_path: str, prompt: str, max_tokens: int = 512) -> str:
        """Query VLM with single image."""
        image = load_image(image_path)

        formatted_prompt = apply_chat_template(
            self.processor,
            config=self.model.config,
            prompt=prompt,
            num_images=1
        )

        output = generate(
            self.model,
            self.processor,
            image,
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            verbose=False
        )
        return output.strip()

    def _query_multi(self, image_paths: List[str], prompt: str, max_tokens: int = 512) -> str:
        """Query VLM with multiple images (processes sequentially, combines context)."""
        # MLX-VLM handles one image at a time, so we'll describe each and combine
        # For now, use the middle frame as representative
        if not image_paths:
            return ""

        mid_idx = len(image_paths) // 2
        return self._query(image_paths[mid_idx], prompt, max_tokens)

    def analyze_movement(self, video_path: str, num_samples: int = 3) -> Dict:
        """
        Analyze video for movement quality characteristics.

        Returns analysis of naturalness, expression variety, timing.
        """
        print(f"Analyzing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        print(f"  Duration: {duration:.1f}s, FPS: {fps:.1f}")

        results = []
        segment_len = min(3.0, duration / num_samples)

        for i in range(num_samples):
            start = i * (duration / num_samples)
            end = start + segment_len

            frames = self._extract_frames(video_path, num_frames=1,
                                          start_sec=start, end_sec=end)
            if not frames:
                continue

            prompt = """Analyze this avatar avatar frame. Describe:

1. HEAD POSITION: Where is the head tilted/rotated?
2. FACIAL EXPRESSION: What emotion is shown? How intense?
3. EYE STATE: Open/closed? Where looking?
4. MOUTH: Open/closed? Speaking?
5. OVERALL ENERGY: Does this look natural and alive?

Be specific and concise."""

            print(f"  Analyzing segment {i+1}/{num_samples}...")
            response = self._query(frames[0], prompt)

            results.append({
                "segment": i + 1,
                "time": f"{start:.1f}s",
                "analysis": response
            })

            # Cleanup temp files
            for f in frames:
                try:
                    os.remove(f)
                except:
                    pass

        # Summary
        print("  Generating summary...")
        mid_frames = self._extract_frames(video_path, num_frames=1,
                                          start_sec=duration * 0.5,
                                          end_sec=duration * 0.5 + 1)

        summary_prompt = """Looking at this avatar frame, what are the TOP 3 things that would make AI-generated animation look natural and alive like this?

Be specific and actionable for training an animation model."""

        summary = self._query(mid_frames[0], summary_prompt) if mid_frames else ""

        for f in mid_frames:
            try:
                os.remove(f)
            except:
                pass

        return {
            "video": video_path,
            "duration": duration,
            "segments": results,
            "summary": summary
        }

    def label_frame(self, image_path: str) -> Dict:
        """
        Label emotion and intensity from a single frame.

        Returns structured data about the avatar's state.
        """
        prompt = """Analyze this avatar avatar and provide a JSON response:

{
    "emotion": "primary emotion (happy/neutral/excited/thinking/annoyed/sad/smug)",
    "intensity": 0.0 to 1.0,
    "head_x": estimated head rotation left(-30) to right(+30) degrees,
    "head_y": estimated head tilt up(-20) to down(+20) degrees,
    "head_z": estimated head roll degrees,
    "eyes_open": 0.0 to 1.0,
    "mouth_open": 0.0 to 1.0,
    "energy": 0.0 to 1.0 (how animated/alive)
}

Respond with ONLY the JSON."""

        response = self._query(image_path, prompt, max_tokens=200)

        # Parse JSON
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        return {"raw": response, "parse_error": True}

    def label_video(self, video_path: str, sample_interval: float = 1.0) -> List[Dict]:
        """Label frames throughout a video at regular intervals."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        labels = []
        for t in np.arange(0, duration, sample_interval):
            frames = self._extract_frames(video_path, num_frames=1,
                                          start_sec=t, end_sec=t + 0.1)
            if frames:
                print(f"  Labeling t={t:.1f}s...")
                label = self.label_frame(frames[0])
                label["timestamp"] = t
                labels.append(label)

                for f in frames:
                    try:
                        os.remove(f)
                    except:
                        pass

        return labels

    def compare(self, generated_path: str, reference_path: str) -> Dict:
        """Compare generated animation against reference."""
        print(f"Comparing:")
        print(f"  Generated: {generated_path}")
        print(f"  Reference: {reference_path}")

        # Get representative frames
        gen_frames = self._extract_frames(generated_path, num_frames=1)
        ref_frames = self._extract_frames(reference_path, num_frames=1)

        if not gen_frames or not ref_frames:
            return {"error": "Could not extract frames"}

        # Analyze generated
        gen_prompt = """Analyze this avatar avatar animation frame. Rate 1-10:
- Naturalness of pose
- Expression quality
- Does it look alive or robotic?

List any issues that make it look unnatural or AI-generated."""

        gen_analysis = self._query(gen_frames[0], gen_prompt)

        # Analyze reference
        ref_prompt = """Analyze this avatar avatar. What makes this look natural and alive?
What specific qualities should AI animation replicate?"""

        ref_analysis = self._query(ref_frames[0], ref_prompt)

        # Cleanup
        for f in gen_frames + ref_frames:
            try:
                os.remove(f)
            except:
                pass

        return {
            "generated_analysis": gen_analysis,
            "reference_analysis": ref_analysis,
        }

    def describe_movement(self, video_path: str,
                          segment_sec: float = 2.0) -> List[Dict]:
        """
        Extract movement descriptions for training data augmentation.
        Creates text descriptions of avatar state at each segment.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        print(f"Extracting descriptions from {duration:.1f}s video...")

        descriptions = []

        for start in np.arange(0, duration - segment_sec, segment_sec):
            frames = self._extract_frames(video_path, num_frames=1,
                                          start_sec=start, end_sec=start + 0.5)
            if not frames:
                continue

            prompt = """Describe this avatar's pose and expression in ONE sentence.
Include: head position, facial expression, eye state, mouth state, energy level.
Example: "Head tilted right, smiling warmly, eyes half-open, mouth slightly open as if speaking, relaxed energy."

Your description:"""

            desc = self._query(frames[0], prompt, max_tokens=100)

            descriptions.append({
                "time": start,
                "description": desc.strip()
            })

            print(f"  t={start:.1f}s: {desc[:60]}...")

            for f in frames:
                try:
                    os.remove(f)
                except:
                    pass

        return descriptions


def main():
    parser = argparse.ArgumentParser(
        description="VLM Animation Analyzer (MLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vlm_analyzer.py analyze video.mp4
  python vlm_analyzer.py label frame.png
  python vlm_analyzer.py label video.mp4 --interval 0.5
  python vlm_analyzer.py compare generated.mp4 reference.mp4
  python vlm_analyzer.py describe video.mp4 -o descriptions.json
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze movement quality")
    p_analyze.add_argument("video", help="Video file")
    p_analyze.add_argument("-n", "--samples", type=int, default=3, help="Number of segments")
    p_analyze.add_argument("-o", "--output", help="Output JSON file")

    # Label
    p_label = subparsers.add_parser("label", help="Label emotion/pose")
    p_label.add_argument("input", help="Image or video file")
    p_label.add_argument("-i", "--interval", type=float, default=1.0,
                         help="Sample interval for videos (seconds)")
    p_label.add_argument("-o", "--output", help="Output JSON file")

    # Compare
    p_compare = subparsers.add_parser("compare", help="Compare generated vs reference")
    p_compare.add_argument("generated", help="Generated animation")
    p_compare.add_argument("reference", help="Reference video")
    p_compare.add_argument("-o", "--output", help="Output JSON file")

    # Describe
    p_describe = subparsers.add_parser("describe", help="Extract movement descriptions")
    p_describe.add_argument("video", help="Video file")
    p_describe.add_argument("-s", "--segment", type=float, default=2.0,
                            help="Segment length (seconds)")
    p_describe.add_argument("-o", "--output", help="Output JSON file")

    args = parser.parse_args()

    if not MLX_AVAILABLE:
        print(f"Error: {MLX_ERROR}")
        print("\nInstall with: pip install mlx-vlm opencv-python")
        sys.exit(1)

    # Initialize
    analyzer = VLMAnalyzer()

    # Run command
    if args.command == "analyze":
        result = analyzer.analyze_movement(args.video, args.samples)

    elif args.command == "label":
        is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        if is_video:
            result = analyzer.label_video(args.input, args.interval)
        else:
            result = analyzer.label_frame(args.input)

    elif args.command == "compare":
        result = analyzer.compare(args.generated, args.reference)

    elif args.command == "describe":
        result = analyzer.describe_movement(args.video, args.segment)

    # Output
    output_str = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_str)
        print(f"\nSaved to: {args.output}")
    else:
        print("\n" + "=" * 50)
        print("RESULTS:")
        print("=" * 50)
        print(output_str)


if __name__ == "__main__":
    main()
