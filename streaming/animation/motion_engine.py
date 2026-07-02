#!/usr/bin/env python3
"""
Motion Engine - Hybrid clip-based animation with blending.

Plays recorded motion clips with:
- Smooth blending between clips
- Micro-movement noise layer
- Emotion-based clip selection
- Orchestrator for automatic transitions

Usage:
    # Run with manual emotion control (keyboard)
    python3 motion_engine.py

    # Run with specific starting emotion
    python3 motion_engine.py --emotion happy

    # API mode (receive emotion commands via websocket)
    python3 motion_engine.py --api-port 8765
"""

import asyncio
import argparse
import json
import random
import time
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

try:
    import websockets
except ImportError:
    print("Install: pip3 install websockets")
    exit(1)

# Import parameter mapper
from param_mapper import map_params, get_core_animation_params

# Import expression engine and emotion state
from expression_engine import ExpressionEngine, EXPRESSION_DISPLAY_NAMES
from emotion_state import get_emotion, EmotionState

# Import easing functions
from easing import ease_in_out_sine, ease_in_out_cubic


CLIPS_DIR = Path(__file__).parent / "clips"

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "thinking", "excited", "tired"]

# Expressions to exclude (these would break the model if they existed as clips)
BLOCKED_EXPRESSIONS = [
    'expr_flying_head',  # Would hide body
    'expr_shrink',       # Would break scaling
    # Note: Accessories, toggles, and hides aren't recorded as clips, just VTS hotkeys
    # All recorded expr_ clips (gestures, eyes, faces) are allowed
]


@dataclass
class Clip:
    name: str
    fps: int
    frames: List[Dict[str, float]]
    params: List[str]
    loopable: bool = True
    duration: float = 0

    @classmethod
    def load(cls, path: Path) -> 'Clip':
        with open(path) as f:
            data = json.load(f)
        return cls(
            name=data['name'],
            fps=data['fps'],
            frames=data['frames'],
            params=data['params'],
            loopable=data.get('loopable', True),
            duration=data.get('duration', len(data['frames']) / data['fps'])
        )


@dataclass
class ClipLibrary:
    """Manages all recorded clips organized by type and emotion."""
    clips: Dict[str, Clip] = field(default_factory=dict)
    by_emotion: Dict[str, List[Clip]] = field(default_factory=dict)

    def load_all(self, clips_dir: Path):
        """Load all clips from directory."""
        skipped = []
        for path in clips_dir.glob("*.json"):
            clip = Clip.load(path)

            # Filter out problematic expressions
            if clip.name in BLOCKED_EXPRESSIONS:
                skipped.append(clip.name)
                continue

            self.clips[clip.name] = clip

            # Parse emotion from name (for idle/talk/react clips)
            parts = clip.name.split('_')
            if len(parts) >= 2:
                emotion = parts[1]
                if emotion in EMOTIONS:
                    if emotion not in self.by_emotion:
                        self.by_emotion[emotion] = []
                    self.by_emotion[emotion].append(clip)

        print(f"Loaded {len(self.clips)} clips")
        if skipped:
            print(f"  Skipped {len(skipped)} problematic clips: {', '.join(skipped)}")
        for emotion, clips in self.by_emotion.items():
            print(f"  {emotion}: {len(clips)} clips")

    def _get_motion_clips_for_emotion(self, emotion: str) -> List[Clip]:
        """
        Get all MOTION clips for the given emotion (with related fallbacks).
        NEVER returns expression clips (expr_*) - those are for hotkey triggers only.
        """
        motion_clips = []
        seen = set()

        clips = self.by_emotion.get(emotion, [])

        # Filter OUT all expr_* clips
        for clip in clips:
            if not clip.name.startswith('expr_') and clip.name not in seen:
                motion_clips.append(clip)
                seen.add(clip.name)

        # If no exact match, include related emotions for variety
        if not motion_clips or len(motion_clips) < 5:
            emotion_groups = {
                'neutral': ['neutral', 'bored', 'confused', 'thinking'],
                'happy': ['happy', 'excited'],
                'sad': ['sad', 'tired'],
                'angry': ['angry'],
                'surprised': ['surprised'],
                'thinking': ['thinking', 'confused', 'neutral'],
                'excited': ['excited', 'happy'],
                'tired': ['tired', 'sad', 'bored'],
            }

            group = emotion_groups.get(emotion, [emotion])
            for related_emotion in group:
                for clip in self.by_emotion.get(related_emotion, []):
                    if not clip.name.startswith('expr_') and clip.name not in seen:
                        motion_clips.append(clip)
                        seen.add(clip.name)

        # Fallback: any non-expression clip
        if not motion_clips:
            motion_clips = [c for c in self.clips.values() if not c.name.startswith('expr_')]

        return motion_clips

    def get_random_for_emotion(self, emotion: str) -> Optional[Clip]:
        """Get a random MOTION clip for the given emotion."""
        motion_clips = self._get_motion_clips_for_emotion(emotion)
        return random.choice(motion_clips) if motion_clips else None

    def get_weighted_for_emotion(self, emotion: str, recent_clips: List[str]) -> Optional[Clip]:
        """
        Get a clip for the given emotion, weighted against recently played clips.
        Clips in recent_clips get much lower weight to encourage variety.
        """
        motion_clips = self._get_motion_clips_for_emotion(emotion)
        if not motion_clips:
            return None

        # Build weights - recent clips get heavily penalized
        weights = []
        for clip in motion_clips:
            if clip.name in recent_clips:
                # Recently played - very low weight (but not zero in case it's all we have)
                weights.append(0.05)
            else:
                weights.append(1.0)

        # Weighted random selection
        total = sum(weights)
        if total == 0:
            return random.choice(motion_clips)

        r = random.uniform(0, total)
        cumulative = 0
        for clip, w in zip(motion_clips, weights):
            cumulative += w
            if r <= cumulative:
                return clip

        return motion_clips[-1]


class BlendLayer:
    """Adds micro-movements and smooths transitions."""

    def __init__(self):
        self.time = 0
        # Breathing parameters (now targets visible params, not filtered Body)
        self.breath_rate = 0.15  # Hz - natural breathing rate
        self.breath_amount = 0.3  # Subtle chest/head rise

        # Micro-movement noise - DISABLED for clip playback
        self.noise_scale = 0.0
        self.noise_speed = 0.2

        # Eye saccades - small involuntary eye movements
        self.next_saccade_time = 0
        self.saccade_target_x = 0.0
        self.saccade_target_y = 0.0
        self.saccade_current_x = 0.0
        self.saccade_current_y = 0.0
        self.saccade_speed = 12.0  # Fast snap to target

        # Subtle head drift - very slow wandering
        self.head_drift_speed = 0.04  # Very slow Hz

        # Per-parameter smoothing weights (0.0 = instant, 1.0 = never changes)
        # Different body parts need different smoothing for natural motion
        self.param_smoothing = {
            # Head - medium smoothing (natural head drift)
            'FaceAngleX': 0.85,  # Yaw
            'FaceAngleY': 0.85,  # Pitch
            'FaceAngleZ': 0.85,  # Roll

            # Eyes - less smoothing (eyes dart quickly)
            'EyeOpenLeft': 0.7,   # Blinks are quick
            'EyeOpenRight': 0.7,
            'EyeRightX': 0.75,    # Gaze shifts are responsive
            'EyeRightY': 0.75,
            'EyeLeftX': 0.75,
            'EyeLeftY': 0.75,

            # Mouth - high smoothing (mouth should flow)
            'MouthOpen': 0.9,     # Very smooth mouth
            'MouthSmile': 0.9,

            # Brows - medium smoothing
            'BrowLeftY': 0.8,
            'BrowRightY': 0.8,
        }

        # Default smoothing for unknown params
        self.default_smoothing = 0.8

        # Max velocity per frame (units per second) - prevents teleporting
        # Higher = more responsive, Lower = more sluggish
        self.max_velocity = {
            # Head - moderate speed (natural head movement)
            'FaceAngleX': 120.0,   # degrees/sec yaw
            'FaceAngleY': 100.0,   # degrees/sec pitch
            'FaceAngleZ': 80.0,    # degrees/sec roll

            # Eyes - fast (eyes move quickly)
            'EyeOpenLeft': 8.0,    # blinks are fast
            'EyeOpenRight': 8.0,
            'EyeRightX': 6.0,     # gaze shifts
            'EyeRightY': 6.0,
            'EyeLeftX': 6.0,
            'EyeLeftY': 6.0,

            # Mouth - medium
            'MouthOpen': 5.0,
            'MouthSmile': 4.0,

            # Brows - medium
            'BrowLeftY': 5.0,
            'BrowRightY': 5.0,
        }
        self.default_max_velocity = 80.0  # degrees/sec for unknown params

        self.prev_output = {}

    def apply(self, params: Dict[str, float], dt: float, is_speaking: bool = False) -> Dict[str, float]:
        """
        Apply micro-movements and smoothing to params.

        Args:
            params: Raw parameter dict
            dt: Delta time
            is_speaking: If True, completely zero out mouth params (TTS takes over)
        """
        self.time += dt
        output = dict(params)

        # MOUTH CONTROL: If speaking, motion clips NEVER control mouth
        # TTS audio detection in VTS will handle mouth movement
        if is_speaking:
            for key in list(output.keys()):
                if 'Mouth' in key:
                    output[key] = 0.0  # Complete override - TTS controls mouth
        else:
            # Dampen mouth when NOT speaking (subtle idle movement)
            for key in output:
                if 'Mouth' in key:
                    output[key] *= 0.2  # Increased from 0.1 - slightly more mouth movement

        # Dampen eye gaze movement
        for key in output:
            if 'Eye' in key and 'Open' not in key:  # Gaze movement
                output[key] *= 0.7  # Increased from 0.5 - more eye movement

        # === MICRO-MOVEMENTS (life-like idle overlay) ===

        # Breathing - subtle head pitch oscillation (visible, unlike filtered Body params)
        breath = math.sin(self.time * 2 * math.pi * self.breath_rate) * self.breath_amount
        if 'FaceAngleY' in output:
            output['FaceAngleY'] += breath * 0.4  # Tiny head nod with breath

        # Subtle head drift - very slow wandering movement
        drift_t = self.time * self.head_drift_speed
        if 'FaceAngleX' in output:
            # Multi-frequency drift for organic feel
            drift_x = (
                math.sin(drift_t * 2 * math.pi * 1.0) * 0.6 +
                math.sin(drift_t * 2 * math.pi * 2.7) * 0.3 +
                math.sin(drift_t * 2 * math.pi * 0.3) * 0.4
            )
            output['FaceAngleX'] += drift_x
        if 'FaceAngleZ' in output:
            drift_z = math.sin(drift_t * 2 * math.pi * 0.7 + 1.2) * 0.25
            output['FaceAngleZ'] += drift_z

        # Eye saccades - small involuntary eye jumps every 2-6 seconds
        if self.time >= self.next_saccade_time:
            # New saccade target
            self.saccade_target_x = random.gauss(0, 0.15)  # Small random offset
            self.saccade_target_y = random.gauss(0, 0.08)
            # Clamp so eyes don't go wild
            self.saccade_target_x = max(-0.4, min(0.4, self.saccade_target_x))
            self.saccade_target_y = max(-0.2, min(0.2, self.saccade_target_y))
            self.next_saccade_time = self.time + random.uniform(2.0, 6.0)

        # Smoothly move toward saccade target (fast snap, like real eyes)
        saccade_lerp = min(1.0, self.saccade_speed * dt)
        self.saccade_current_x += (self.saccade_target_x - self.saccade_current_x) * saccade_lerp
        self.saccade_current_y += (self.saccade_target_y - self.saccade_current_y) * saccade_lerp

        for key in output:
            if key in ('EyeRightX', 'EyeLeftX'):
                output[key] += self.saccade_current_x
            elif key in ('EyeRightY', 'EyeLeftY'):
                output[key] += self.saccade_current_y

        # Per-parameter smoothing with different weights
        if self.prev_output:
            for key in output:
                if key in self.prev_output:
                    # Get parameter-specific smoothing weight
                    smoothing = self.param_smoothing.get(key, self.default_smoothing)

                    # Apply EMA with parameter-specific weight
                    output[key] = (
                        smoothing * self.prev_output[key] +
                        (1 - smoothing) * output[key]
                    )

        # Velocity limiting - cap max change per frame
        if self.prev_output and dt > 0:
            for key in output:
                if key in self.prev_output:
                    max_vel = self.max_velocity.get(key, self.default_max_velocity)
                    max_delta = max_vel * dt  # Scale by frame time
                    delta = output[key] - self.prev_output[key]
                    if abs(delta) > max_delta:
                        output[key] = self.prev_output[key] + math.copysign(max_delta, delta)

        self.prev_output = dict(output)
        return output


class ClipPlayer:
    """Plays clips with crossfade blending."""

    def __init__(self):
        self.current_clip: Optional[Clip] = None
        self.current_frame: float = 0  # Float for interpolation
        self.next_clip: Optional[Clip] = None
        self.blend_progress: float = 1.0  # 1.0 = fully on current
        self.blend_duration: float = 0.5  # seconds

    def play(self, clip: Clip, crossfade: float = 0.5):
        """Start playing a clip, crossfading from current."""
        if self.current_clip is None:
            self.current_clip = clip
            self.current_frame = 0
            self.blend_progress = 1.0
        else:
            self.next_clip = clip
            self.blend_progress = 0.0
            self.blend_duration = crossfade

    def get_frame(self, dt: float) -> Dict[str, float]:
        """Get the current blended frame, advance playhead."""
        if self.current_clip is None:
            return {}

        # Advance frame - playback speed tuning
        playback_speed = 0.9  # 90% speed - slightly slower than recorded
        self.current_frame += dt * self.current_clip.fps * playback_speed

        # Loop or stop
        if self.current_frame >= len(self.current_clip.frames):
            if self.current_clip.loopable:
                self.current_frame = 0
            else:
                self.current_frame = len(self.current_clip.frames) - 1

        # Get current frame (interpolated with subtle easing)
        frame_idx = int(self.current_frame)
        frame_frac = self.current_frame - frame_idx
        frame_idx = min(frame_idx, len(self.current_clip.frames) - 1)
        next_idx = min(frame_idx + 1, len(self.current_clip.frames) - 1)

        # Apply subtle easing to frame interpolation for smoother motion
        # Use quadratic for within-clip (less aggressive than cubic)
        eased_frac = ease_in_out_cubic(frame_frac) if frame_frac > 0 else 0

        current_params = self._lerp_frames(
            self.current_clip.frames[frame_idx],
            self.current_clip.frames[next_idx],
            eased_frac
        )

        # Handle crossfade to next clip
        if self.next_clip is not None:
            self.blend_progress += dt / self.blend_duration

            if self.blend_progress >= 1.0:
                # Transition complete
                self.current_clip = self.next_clip
                self.current_frame = 0
                self.next_clip = None
                self.blend_progress = 1.0
            else:
                # Blend between current and next with EASING
                # Apply easing curve for smooth acceleration/deceleration
                eased_progress = ease_in_out_sine(self.blend_progress)

                next_frame = self.next_clip.frames[0]
                current_params = self._lerp_frames(
                    current_params, next_frame, eased_progress
                )

        return current_params

    def _lerp_frames(self, a: Dict, b: Dict, t: float) -> Dict[str, float]:
        """Linear interpolation between two frames."""
        result = {}
        all_keys = set(a.keys()) | set(b.keys())
        for key in all_keys:
            va = a.get(key, 0)
            vb = b.get(key, 0)
            result[key] = va + (vb - va) * t
        return result


class Orchestrator:
    """Decides when to switch clips based on emotion and timing."""

    def __init__(self, library: ClipLibrary, player: ClipPlayer):
        self.library = library
        self.player = player
        self.current_emotion = "neutral"
        self.time_in_clip = 0
        self.min_clip_time = 8.0  # Increased from 5.0 - more stable
        self.max_clip_time = 30.0  # Increased from 20.0 - longer clips
        self.recent_clips: List[str] = []  # Track recently played clip names
        self.recent_clip_limit = 6  # Don't repeat within last N clips

    def set_emotion(self, emotion: str, force: bool = False):
        """Change emotion, triggering a clip switch."""
        if force or emotion != self.current_emotion:
            self.current_emotion = emotion
            self._pick_new_clip()

    def update(self, dt: float):
        """Update orchestrator, possibly switch clips."""
        self.time_in_clip += dt

        # Random chance to switch after min time
        if self.time_in_clip > self.min_clip_time:
            switch_chance = (self.time_in_clip - self.min_clip_time) / (self.max_clip_time - self.min_clip_time)
            if random.random() < switch_chance * dt:
                self._pick_new_clip()

        # Force switch after max time
        if self.time_in_clip > self.max_clip_time:
            self._pick_new_clip()

    def _pick_new_clip(self):
        """Pick a new clip for current emotion, avoiding recent repeats."""
        clip = self.library.get_weighted_for_emotion(
            self.current_emotion, self.recent_clips
        )
        if clip and (self.player.current_clip is None or clip.name != self.player.current_clip.name):
            self.player.play(clip, crossfade=0.8)
            self.time_in_clip = 0
            # Track in recent history
            self.recent_clips.append(clip.name)
            if len(self.recent_clips) > self.recent_clip_limit:
                self.recent_clips.pop(0)
            print(f"\n→ Playing: {clip.name}")


class VTSConnection:
    def __init__(self, host="localhost", port=8001):
        self.host = host
        self.port = port
        self.ws = None

    async def connect(self):
        uri = f"ws://{self.host}:{self.port}"
        print(f"Connecting to VTS at {uri}...")
        self.ws = await websockets.connect(uri)

        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth-token",
            "messageType": "AuthenticationTokenRequest",
            "data": {"pluginName": "Motion Engine", "pluginDeveloper": "Alice"}
        }))
        resp = json.loads(await self.ws.recv())
        token = resp.get("data", {}).get("authenticationToken")

        if not token:
            return False

        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth",
            "messageType": "AuthenticationRequest",
            "data": {"pluginName": "Motion Engine", "pluginDeveloper": "Alice",
                     "authenticationToken": token}
        }))
        resp = json.loads(await self.ws.recv())
        return resp.get("data", {}).get("authenticated", False)

    async def ensure_connected(self):
        """Reconnect if the websocket died (e.g. after blocking input)."""
        try:
            if self.ws and self.ws.open:
                return True
        except Exception:
            pass
        print("  VTS connection lost, reconnecting...")
        return await self.connect()

    async def set_params(self, params: Dict[str, float], weight: float = 1.0):
        param_list = [{"id": k, "value": float(v), "weight": weight} for k, v in params.items()]
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "inject",
            "messageType": "InjectParameterDataRequest",
            "data": {"parameterValues": param_list}
        }))
        await self.ws.recv()

    async def trigger_hotkey(self, hotkey_id: str) -> bool:
        """
        Trigger a VTS hotkey by its ID.

        Args:
            hotkey_id: The actual VTS hotkey UUID

        Returns:
            True if successful
        """
        try:
            await self.ws.send(json.dumps({
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "hotkey",
                "messageType": "HotkeyTriggerRequest",
                "data": {"hotkeyID": hotkey_id}
            }))
            resp = json.loads(await self.ws.recv())
            # VTS returns hotkeyID in data on success, APIError on failure
            return "hotkeyID" in resp.get("data", {})
        except Exception as e:
            print(f"Failed to trigger hotkey: {e}")
            return False

    async def list_hotkeys(self) -> list:
        """Query all hotkeys from the current VTS model. Returns list of hotkey dicts with name, hotkeyID, type, etc."""
        try:
            await self.ws.send(json.dumps({
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "hotkey-list",
                "messageType": "HotkeysInCurrentModelRequest",
                "data": {}
            }))
            resp = json.loads(await self.ws.recv())
            return resp.get("data", {}).get("availableHotkeys", [])
        except Exception as e:
            print(f"Failed to list hotkeys: {e}")
            return []

    async def get_expression_states(self) -> list:
        """Query all expressions and their active states from VTS."""
        try:
            await self.ws.send(json.dumps({
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "expr-state",
                "messageType": "ExpressionStateRequest",
                "data": {"details": True}
            }))
            resp = json.loads(await self.ws.recv())
            return resp.get("data", {}).get("expressions", [])
        except Exception as e:
            print(f"Failed to get expression states: {e}")
            return []

    async def set_expression_active(self, expression_file: str, active: bool) -> bool:
        """Activate or deactivate an expression by its file name."""
        try:
            await self.ws.send(json.dumps({
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "expr-set",
                "messageType": "ExpressionActivationRequest",
                "data": {"expressionFile": expression_file, "active": active}
            }))
            resp = json.loads(await self.ws.recv())
            return "error" not in resp.get("messageType", "").lower()
        except Exception as e:
            print(f"Failed to set expression state: {e}")
            return False

    async def reset_dangerous_expressions(self):
        """
        On startup, ONLY disable expressions that break the model.
        Leaves everything else exactly as-is (watermark, tongue, etc.)
        """
        # Only these specific expressions get reset - nothing else
        DANGEROUS_KEYWORDS = ['flying head', 'shrink']

        expressions = await self.get_expression_states()
        if not expressions:
            print("  Could not query expression states (may not be supported)")
            return

        reset_count = 0
        for expr in expressions:
            name = expr.get("name", "").lower()
            file = expr.get("file", "")
            active = expr.get("active", False)

            if active and any(kw in name for kw in DANGEROUS_KEYWORDS):
                print(f"  Resetting dangerous expression: {expr.get('name')} ({file})")
                await self.set_expression_active(file, False)
                reset_count += 1

        if reset_count:
            print(f"  Reset {reset_count} dangerous expression(s)")
        else:
            print("  No dangerous expressions active - all good")


async def run_engine(args):
    """Main engine loop."""
    import signal

    # Load clips
    library = ClipLibrary()
    library.load_all(CLIPS_DIR)

    if not library.clips:
        print("\nNo clips found! Record some first:")
        print("  python3 capture_clip.py idle_neutral_01")
        print("  python3 capture_clip.py idle_happy_01")
        return

    # Setup components
    player = ClipPlayer()
    blend = BlendLayer()
    orchestrator = Orchestrator(library, player)

    # Start with initial emotion
    orchestrator.set_emotion(args.emotion, force=True)

    # Show parameter mapping info
    if player.current_clip and len(player.current_clip.frames) > 0:
        sample_frame = player.current_clip.frames[0]
        print(f"\n=== Parameter Mapping Test ===")
        print(f"Clip has {len(sample_frame)} parameters")
        vts_mapped = map_params(sample_frame)
        core = get_core_animation_params(vts_mapped)
        print(f"Mapped to {len(vts_mapped)} VTS params")
        print(f"Filtered to {len(core)} core params: {list(core.keys())}")
        print()

    # Connect to VTS
    vts = VTSConnection(args.host, args.port)
    if not await vts.connect():
        print("Failed to connect to VTS")
        return

    # Initialize expression engine - queries VTS for real hotkey IDs
    expression_engine = ExpressionEngine(vts)
    await expression_engine.init_hotkeys()

    # Expression test cycling
    test_expressions = list(expression_engine.get_available_expressions())
    test_expr_idx = 0

    print(f"\nMotion Engine running!")
    print(f"Starting emotion: {args.emotion}")
    print("=" * 50)
    print("Controls:")
    print("  1-8: Change emotion (triggers motion + expression)")
    print("  e: Cycle test expression (manual trigger)")
    print("  c: Clear all expressions")
    print("  q: Quit")
    print(f"  Available expressions: {len(test_expressions)}")
    print("=" * 50)

    # Handle stop
    stop_flag = False
    def handle_stop(sig, frame):
        nonlocal stop_flag
        stop_flag = True
    signal.signal(signal.SIGINT, handle_stop)

    # Keyboard input (non-blocking, cross-platform)
    import sys
    _is_windows = sys.platform == "win32"
    _old_settings = None
    if _is_windows:
        import msvcrt
    else:
        import select
        if sys.stdin.isatty():
            import tty
            import termios
            _old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    fps = 40  # Higher FPS for smoother motion
    frame_time = 1.0 / fps
    frame_count = 0
    last_emotion_check = 0
    emotion_check_interval = 0.1  # Check emotion state every 100ms

    try:
        while not stop_flag:
            loop_start = time.time()

            # Check emotion state from file (EmotionBERT integration)
            if time.time() - last_emotion_check > emotion_check_interval:
                emotion_state = get_emotion()

                # Update motion clips if emotion changed
                if emotion_state.emotion != orchestrator.current_emotion:
                    orchestrator.set_emotion(emotion_state.emotion)
                    # Update expressions
                    await expression_engine.update_emotion(emotion_state.emotion)

                last_emotion_check = time.time()
                is_speaking = emotion_state.is_speaking
            else:
                # Use last known state
                is_speaking = False  # Default if state not checked yet

            # Check keyboard (manual override)
            if _is_windows:
                _has_key = msvcrt.kbhit()
            elif sys.stdin.isatty():
                _has_key = select.select([sys.stdin], [], [], 0)[0]
            else:
                _has_key = False
            if _has_key:
                    key = msvcrt.getwch() if _is_windows else sys.stdin.read(1)
                    if key == 'q':
                        break
                    elif key == 'e' and test_expressions:
                        # Cycle through expressions for testing
                        expr_name = test_expressions[test_expr_idx % len(test_expressions)]
                        print(f"\n[TEST] Expression: {expr_name}")
                        await expression_engine.trigger_by_name(expr_name)
                        test_expr_idx += 1
                    elif key == 'c':
                        # Clear all expressions
                        print(f"\n[TEST] Clearing all expressions")
                        await expression_engine._clear_all_expressions()
                    elif key in '12345678':
                        idx = int(key) - 1
                        if idx < len(EMOTIONS):
                            orchestrator.set_emotion(EMOTIONS[idx])
                            await expression_engine.update_emotion(EMOTIONS[idx])
                            print(f"\nEmotion: {EMOTIONS[idx]}")

            # Update orchestrator
            orchestrator.update(frame_time)

            # Get frame from player
            raw_params = player.get_frame(frame_time)

            # Apply blend layer (with speaking flag for mouth control)
            final_params = blend.apply(raw_params, frame_time, is_speaking=is_speaking)

            # Map Live2D params to VTS params
            vts_params = map_params(final_params)
            core_params = get_core_animation_params(vts_params)

            # Send to VTS
            if core_params:
                await vts.set_params(core_params)

            frame_count += 1
            if frame_count % 15 == 0:  # Print every second at 15 FPS
                clip_name = player.current_clip.name if player.current_clip else "none"
                param_count = len(core_params) if 'core_params' in locals() else 0
                print(f"\r[{orchestrator.current_emotion}] {clip_name} | Params: {param_count} | Frame {frame_count}", end="")

            # Maintain FPS
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                await asyncio.sleep(frame_time - elapsed)

    finally:
        if _old_settings is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _old_settings)

    print(f"\nStopped. Total frames: {frame_count}")


def main():
    parser = argparse.ArgumentParser(description="Motion Engine")
    parser.add_argument("--emotion", "-e", default="neutral", choices=EMOTIONS,
                        help="Starting emotion")
    parser.add_argument("--host", default="localhost", help="VTS host")
    parser.add_argument("--port", type=int, default=8001, help="VTS port")
    parser.add_argument("--api-port", type=int, help="Enable API on this port for external emotion control")
    args = parser.parse_args()

    asyncio.run(run_engine(args))


if __name__ == "__main__":
    main()
