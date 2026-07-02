#!/usr/bin/env python3
"""
Capture named motion clips for the hybrid animation system.

Records VTS parameters as reusable motion clips.

Usage:
    python3 capture_clip.py idle_neutral_01
    python3 capture_clip.py thinking_loop --duration 10
    python3 capture_clip.py laugh_reaction --duration 3

    # List all clips
    python3 capture_clip.py --list

    # Preview a clip
    python3 capture_clip.py --preview idle_neutral_01

Clip naming convention:
    {type}_{emotion}_{variant}

    Types: idle, react, talk, transition
    Emotions: neutral, happy, sad, angry, surprised, thinking, excited, tired
    Variant: 01, 02, 03... or descriptive (loop, burst, slow)
"""

import asyncio
import argparse
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Install: pip3 install websockets")
    exit(1)


CLIPS_DIR = Path(__file__).parent / "clips"


class VTSConnection:
    def __init__(self, host="localhost", port=8001):
        self.host = host
        self.port = port
        self.ws = None
        self.param_names = []

    async def connect(self):
        uri = f"ws://{self.host}:{self.port}"
        self.ws = await websockets.connect(uri)

        # Auth
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth-token",
            "messageType": "AuthenticationTokenRequest",
            "data": {"pluginName": "Clip Capture", "pluginDeveloper": "Alice"}
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
            "data": {"pluginName": "Clip Capture", "pluginDeveloper": "Alice",
                     "authenticationToken": token}
        }))
        resp = json.loads(await self.ws.recv())
        return resp.get("data", {}).get("authenticated", False)

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
            if p["name"] not in self.param_names:
                self.param_names.append(p["name"])
        return params

    async def set_params(self, param_dict: dict, weight: float = 1.0):
        params = [{"id": k, "value": float(v), "weight": weight} for k, v in param_dict.items()]
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "inject",
            "messageType": "InjectParameterDataRequest",
            "data": {"parameterValues": params}
        }))
        await self.ws.recv()


def find_important_params(frames: list, threshold: float = 0.05) -> list:
    """Find params that actually move in this clip."""
    if not frames:
        return []

    all_params = list(frames[0].keys())
    important = []

    for param in all_params:
        values = [f.get(param, 0) for f in frames]
        std = np.std(values)
        if std > threshold:
            important.append(param)

    return important


async def capture_clip(name: str, duration: float, fps: int, host: str, port: int):
    """Capture a motion clip."""
    import signal

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    vts = VTSConnection(host, port)
    if not await vts.connect():
        print("Failed to connect to VTS")
        return

    print(f"Connected! Recording clip: {name}")
    print(f"Duration: {duration}s at {fps}fps")
    print("=" * 50)
    print("Start moving! Recording in 3...")
    await asyncio.sleep(1)
    print("2...")
    await asyncio.sleep(1)
    print("1...")
    await asyncio.sleep(1)
    print("RECORDING!")
    print("=" * 50)

    frames = []
    frame_time = 1.0 / fps
    start_time = time.time()

    stop_flag = False
    def handle_stop(sig, frame):
        nonlocal stop_flag
        stop_flag = True
    signal.signal(signal.SIGINT, handle_stop)

    while not stop_flag:
        loop_start = time.time()
        elapsed = loop_start - start_time

        if elapsed >= duration:
            break

        params = await vts.get_params()
        frames.append(params)

        remaining = duration - elapsed
        print(f"\r[{elapsed:.1f}s / {duration:.1f}s] Frames: {len(frames)} | {remaining:.1f}s remaining", end="")

        loop_time = time.time() - loop_start
        if loop_time < frame_time:
            await asyncio.sleep(frame_time - loop_time)

    print("\n\nProcessing...")

    # Find which params actually moved
    important = find_important_params(frames)
    print(f"Active params: {len(important)}")

    # Compress: only store important params
    compressed_frames = []
    for f in frames:
        compressed_frames.append({p: f[p] for p in important})

    # Save clip
    clip_data = {
        "name": name,
        "fps": fps,
        "duration": len(frames) / fps,
        "n_frames": len(frames),
        "params": important,
        "frames": compressed_frames,
        "recorded": datetime.now().isoformat(),
        "loopable": name.endswith("_loop") or "idle" in name,
    }

    clip_path = CLIPS_DIR / f"{name}.json"
    with open(clip_path, 'w') as f:
        json.dump(clip_data, f)

    print(f"Saved: {clip_path}")
    print(f"  {len(frames)} frames, {len(important)} active params")

    # Analyze for loopability
    if len(frames) > 10:
        first = np.array([frames[0].get(p, 0) for p in important])
        last = np.array([frames[-1].get(p, 0) for p in important])
        loop_error = np.mean(np.abs(first - last))
        print(f"  Loop error: {loop_error:.3f} (lower = better loop)")


async def preview_clip(name: str, host: str, port: int):
    """Play back a recorded clip."""
    clip_path = CLIPS_DIR / f"{name}.json"
    if not clip_path.exists():
        print(f"Clip not found: {clip_path}")
        return

    with open(clip_path) as f:
        clip = json.load(f)

    vts = VTSConnection(host, port)
    if not await vts.connect():
        return

    print(f"Playing: {name}")
    print(f"  {clip['n_frames']} frames at {clip['fps']}fps")
    print("Press Ctrl+C to stop")

    import signal
    stop_flag = False
    def handle_stop(sig, frame):
        nonlocal stop_flag
        stop_flag = True
    signal.signal(signal.SIGINT, handle_stop)

    frame_time = 1.0 / clip['fps']
    frame_idx = 0

    while not stop_flag:
        frame = clip['frames'][frame_idx]
        await vts.set_params(frame)

        frame_idx = (frame_idx + 1) % len(clip['frames'])

        if frame_idx == 0:
            print("\r[looping]", end="")
        else:
            print(f"\rFrame {frame_idx}/{clip['n_frames']}", end="")

        await asyncio.sleep(frame_time)

    print("\nStopped")


def list_clips():
    """List all recorded clips."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    clips = list(CLIPS_DIR.glob("*.json"))

    if not clips:
        print("No clips recorded yet.")
        print(f"Clips directory: {CLIPS_DIR}")
        return

    print(f"Clips in {CLIPS_DIR}:\n")

    # Group by type
    by_type = {}
    for clip_path in sorted(clips):
        with open(clip_path) as f:
            clip = json.load(f)

        name = clip['name']
        parts = name.split('_')
        clip_type = parts[0] if parts else 'other'

        if clip_type not in by_type:
            by_type[clip_type] = []
        by_type[clip_type].append(clip)

    for clip_type, type_clips in sorted(by_type.items()):
        print(f"{clip_type.upper()}:")
        for clip in type_clips:
            loop = "🔄" if clip.get('loopable') else "  "
            print(f"  {loop} {clip['name']}: {clip['duration']:.1f}s, {clip['n_frames']} frames")
        print()


DEFAULT_SESSION = []

# Helper to generate 3 variations
def _add(base_name, duration, instruction):
    for i in range(1, 4):
        DEFAULT_SESSION.append((f"{base_name}_{i:02d}", duration, f"{instruction} (variation {i})"))

# =============================================================================
# IDLE STATES (longer, loopable) - 10 sec each, 3 variations
# =============================================================================
_add("idle_neutral", 10, "NEUTRAL - Relaxed, natural small movements, breathing")
_add("idle_happy", 10, "HAPPY - Smiling, upbeat energy, slight bounce")
_add("idle_sad", 10, "SAD - Droopy, slow, looking down sometimes")
_add("idle_angry", 10, "ANGRY - Tense, furrowed brow, sharp small movements")
_add("idle_excited", 10, "EXCITED - High energy! Bouncy, wide eyes, can't sit still")
_add("idle_tired", 10, "TIRED - Slow, heavy eyelids, low energy, maybe swaying")
_add("idle_thinking", 10, "THINKING - Contemplative, looking up/away, slower")
_add("idle_confused", 10, "CONFUSED - Puzzled expression, head tilts, squinting")
_add("idle_seductive", 10, "SEDUCTIVE - Flirty, lidded eyes, slow movements, smirk")
_add("idle_shy", 10, "SHY - Looking down/away, small movements, maybe blushing pose")
_add("idle_smug", 10, "SMUG - Self-satisfied smirk, confident posture")
_add("idle_bored", 10, "BORED - Disinterested, maybe looking around, sighing energy")

# =============================================================================
# TALKING STATES (mouth moving, different energies) - 10 sec each
# =============================================================================
_add("talk_neutral", 10, "TALKING NEUTRAL - Normal speaking, mouth moving naturally")
_add("talk_happy", 10, "TALKING HAPPY - Animated speaking, smiling while talking")
_add("talk_excited", 10, "TALKING EXCITED - Fast, energetic speaking, expressive")
_add("talk_sad", 10, "TALKING SAD - Slower speaking, subdued energy")
_add("talk_angry", 10, "TALKING ANGRY - Intense speaking, sharp movements")
_add("talk_thinking", 10, "TALKING THINKING - Speaking slowly, pausing, looking up")
_add("talk_seductive", 10, "TALKING SEDUCTIVE - Slow, deliberate, flirty speaking")
_add("talk_whisper", 10, "WHISPERING - Quiet, leaning in slightly, softer movements")

# =============================================================================
# REACTIONS (shorter, punchy) - 4-6 sec each
# =============================================================================
_add("react_surprised", 5, "SURPRISED - Quick shock! Eyes wide, maybe gasp")
_add("react_laugh", 6, "LAUGHING - Genuine laugh, eyes squint, shaking")
_add("react_giggle", 5, "GIGGLE - Smaller laugh, maybe covering mouth energy")
_add("react_gasp", 4, "GASP - Sharp intake, shock or awe")
_add("react_sigh", 5, "SIGH - Exhale, shoulders drop, relief or exasperation")
_add("react_eyeroll", 4, "EYE ROLL - Dramatic eye roll, maybe head tilt")
_add("react_pout", 5, "POUT - Sad pout, puppy eyes, pleading")
_add("react_smirk", 4, "SMIRK - One-sided smile, knowing look")
_add("react_cringe", 5, "CRINGE - Uncomfortable, squinting, pulling back")
_add("react_uwu", 5, "UWU/CUTE - Maximum cute energy, soft expression")

# =============================================================================
# HEAD MOVEMENTS - 5 sec each
# =============================================================================
_add("head_nod", 5, "NODDING YES - Clear agreeing nods, a few times")
_add("head_shake", 5, "SHAKING NO - Clear disagreeing shakes")
_add("head_tilt_left", 5, "HEAD TILT LEFT - Cute/questioning tilt to your left")
_add("head_tilt_right", 5, "HEAD TILT RIGHT - Tilt to your right")
_add("head_tilt_curious", 5, "CURIOUS TILT - Questioning head tilt with raised brow")
_add("head_look_up", 5, "LOOKING UP - Thinking, remembering, eyes and head up")
_add("head_look_down", 5, "LOOKING DOWN - Shy, sad, or reading something")
_add("head_look_left", 5, "LOOKING LEFT - Glancing to your left, maybe suspicious")
_add("head_look_right", 5, "LOOKING RIGHT - Glancing to your right")
_add("head_look_around", 6, "LOOKING AROUND - Scanning, searching, shifting gaze")

# =============================================================================
# EYE/FACE SPECIFIC - 4-5 sec each
# =============================================================================
_add("face_blink", 5, "NATURAL BLINKS - Just natural blinking pattern")
_add("face_wink_left", 4, "WINK LEFT - Flirty wink with left eye")
_add("face_wink_right", 4, "WINK RIGHT - Wink with right eye")
_add("face_wink_tilt", 5, "WINK + HEAD TILT - Wink with cute head tilt combo")
_add("face_tongue_out", 5, "TONGUE OUT - Playful tongue sticking out")
_add("face_tongue_tease", 5, "TONGUE TEASE - Teasing tongue, maybe licking lips")
_add("face_kiss", 4, "KISS/SMOOCH - Blowing a kiss or kissy face")
_add("face_angry_eyes", 5, "ANGRY EYES - Glaring, intense stare, furrowed")
_add("face_pleading", 5, "PLEADING EYES - Big puppy eyes, begging")
_add("face_sleepy", 5, "SLEEPY EYES - Drooping, heavy lids, yawning")
_add("face_wide_eyes", 4, "WIDE EYES - Surprised or shocked wide eyes")
_add("face_squint", 4, "SQUINTING - Suspicious, scrutinizing, or bright light")

# =============================================================================
# FILLER/HESITATION - 4-6 sec each
# =============================================================================
_add("filler_umm", 5, "UMMMM - Hesitating, thinking of what to say, 'umm'")
_add("filler_uhh", 5, "UHHH - Confused hesitation, 'uhhhh'")
_add("filler_hmm", 5, "HMMM - Pondering, considering, 'hmmmm'")
_add("filler_well", 4, "WELL... - Starting to explain, 'well...'")
_add("filler_sooo", 5, "SOOO - Awkward trailing off, 'sooo...'")

# =============================================================================
# SPECIAL/FUN - 5 sec each
# =============================================================================
_add("special_yawn", 6, "YAWN - Big tired yawn, stretching maybe")
_add("special_stretch", 6, "STRETCH - Stretching, loosening up")
_add("special_dance", 8, "LITTLE DANCE - Grooving, vibing, small dance moves")
_add("special_celebrate", 6, "CELEBRATE - Victory! Excitement, maybe fist pump energy")
_add("special_scared", 5, "SCARED - Frightened, shrinking back, worried")
_add("special_disgust", 5, "DISGUST - Eww face, grossed out")
_add("special_impressed", 5, "IMPRESSED - Wow, raised eyebrows, nodding approval")
_add("special_mischief", 6, "MISCHIEVOUS - Up to something, sneaky smile, shifty eyes")
_add("special_zoned_out", 8, "ZONED OUT - Spacing out, unfocused, daydreaming")


async def capture_expressions(fps: int, host: str, port: int, duration: float):
    """Interactive expression toggle capture.

    You trigger VTS expression hotkeys, name them, and it records.
    """
    import signal
    import sys
    import select

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    vts = VTSConnection(host, port)
    if not await vts.connect():
        print("Failed to connect to VTS")
        return

    print("=" * 60)
    print("  EXPRESSION TOGGLE CAPTURE")
    print("=" * 60)
    print()
    print("  How it works:")
    print("  1. Trigger your VTS expression hotkey")
    print("  2. Type a name for it (e.g., 'heart_eyes')")
    print("  3. Recording starts for 8 seconds")
    print("  4. Repeat for all expressions")
    print()
    print("  Commands:")
    print("    [name]  - Start recording with that name")
    print("    q       - Quit")
    print("    list    - Show recorded expressions")
    print("=" * 60)

    expr_count = 0

    while True:
        print(f"\n[{expr_count} recorded] Trigger expression, then type name (or 'q' to quit):")

        try:
            name = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not name:
            continue
        if name.lower() == 'q':
            break
        if name.lower() == 'list':
            expr_clips = [c for c in CLIPS_DIR.glob("expr_*.json")]
            print(f"\nRecorded expressions ({len(expr_clips)}):")
            for c in sorted(expr_clips):
                print(f"  {c.stem}")
            continue

        # Sanitize name
        clean_name = "expr_" + name.replace(" ", "_").replace("-", "_").lower()

        print(f"\n  Recording: {clean_name}")
        print("  3...")
        await asyncio.sleep(1)
        print("  2...")
        await asyncio.sleep(1)
        print("  1...")
        await asyncio.sleep(1)
        print("  🔴 RECORDING!")

        # Record
        frames = []
        frame_time = 1.0 / fps
        start_time = time.time()

        while True:
            loop_start = time.time()
            elapsed = loop_start - start_time

            if elapsed >= duration:
                break

            params = await vts.get_params()
            frames.append(params)

            remaining = duration - elapsed
            bar_len = 20
            bar_fill = int((elapsed / duration) * bar_len)
            bar = "█" * bar_fill + "░" * (bar_len - bar_fill)
            print(f"\r  [{bar}] {remaining:.1f}s", end="", flush=True)

            loop_time = time.time() - loop_start
            if loop_time < frame_time:
                await asyncio.sleep(frame_time - loop_time)

        print("\n")

        if len(frames) < 10:
            print("  Too short, skipped")
            continue

        # Save
        important = find_important_params(frames)
        compressed = [{p: f[p] for p in important} for f in frames]

        clip_data = {
            "name": clean_name,
            "fps": fps,
            "duration": len(frames) / fps,
            "n_frames": len(frames),
            "params": important,
            "frames": compressed,
            "recorded": datetime.now().isoformat(),
            "loopable": True,
            "is_expression": True,
        }

        clip_path = CLIPS_DIR / f"{clean_name}.json"
        with open(clip_path, 'w') as f:
            json.dump(clip_data, f)

        print(f"  ✓ Saved: {clean_name}")
        expr_count += 1

    print(f"\nDone! Recorded {expr_count} expressions.")

    # List all
    expr_clips = [c for c in CLIPS_DIR.glob("expr_*.json")]
    if expr_clips:
        print(f"\nAll expressions ({len(expr_clips)}):")
        for c in sorted(expr_clips):
            print(f"  {c.stem}")


CATEGORIES = {
    "idle": "Idle states (neutral, happy, sad, etc.)",
    "talk": "Talking animations",
    "react": "Reactions (laugh, gasp, etc.)",
    "head": "Head movements (nod, shake, tilt)",
    "face": "Face/eye specific (blink, wink, tongue)",
    "filler": "Filler/hesitation (umm, uhh)",
    "special": "Special/fun (yawn, dance, celebrate)",
    "expr": "VTS expression hotkeys (accessories, poses, effects)",
}

# Expression hotkeys - predefined session (16 total)
EXPRESSION_SESSION = [
    # Poses (4)
    ("expr_holding_star", 8, "HOLDING STAR - Press: Left Shift + 3"),
    ("expr_heart_gesture", 8, "HEART GESTURE - Press: Left Shift + 4"),
    ("expr_covering_chest", 8, "COVERING CHEST - Press: Left Shift + 5"),
    ("expr_praying", 8, "PRAYING - Press: Left Shift + 6"),
    # Eye expressions (5)
    ("expr_black_eyes", 8, "BLACK EYES - Press: Left Ctrl + 1"),
    ("expr_white_eyes", 8, "WHITE EYES - Press: Left Ctrl + 2"),
    ("expr_dizzy", 8, "DIZZY - Press: Right Shift + 6"),
    ("expr_star_eyes", 8, "STAR EYES - Press: Right Shift + 8"),
    ("expr_heart_eyes", 8, "HEART EYES - Press: Right Shift + 9"),
    # Face expressions (7)
    ("expr_tongue_out", 8, "TONGUE OUT - Press: Left Ctrl + 0"),
    ("expr_angry", 8, "ANGRY - Press: Right Shift + 1"),
    ("expr_speechless", 8, "SPEECHLESS - Press: Right Shift + 2"),
    ("expr_zzz_sleepy", 8, "ZZZ (SLEEPY) - Press: Right Shift + 3"),
    ("expr_dark_face", 8, "DARK FACE - Press: Right Shift + 4"),
    ("expr_blushing", 8, "BLUSHING - Press: Right Shift + 5"),
    ("expr_crying", 8, "CRYING - Press: Right Shift + 7"),
]


def filter_session(prompts: list, category: str = None) -> list:
    """Filter prompts by category prefix."""
    if category == "expr":
        return EXPRESSION_SESSION
    if category == "remaining":
        # Everything not yet done: head (from 22), face, filler, special, expr
        remaining = []
        remaining += [p for p in prompts if p[0].startswith("head")][22:]  # head from 22
        remaining += [p for p in prompts if p[0].startswith("face")]
        remaining += [p for p in prompts if p[0].startswith("filler")]
        remaining += [p for p in prompts if p[0].startswith("special")]
        remaining += EXPRESSION_SESSION
        return remaining
    if not category or category == "all":
        return prompts
    return [p for p in prompts if p[0].startswith(category)]


async def run_expression_session(prompts: list, fps: int, host: str, port: int, start_at: int = 0):
    """Guided expression capture - auto-advances, just trigger your VTS hotkeys."""
    import sys
    import tty
    import termios

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    vts = VTSConnection(host, port)
    if not await vts.connect():
        print("Failed to connect to VTS")
        return

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    print("=" * 60)
    print("  EXPRESSION CAPTURE SESSION")
    print("=" * 60)
    print(f"  {len(prompts)} expressions to record")
    print()
    print("  How it works:")
    print("    1. See the hotkey instruction")
    print("    2. Press the VTS hotkey during countdown")
    print("    3. Recording happens automatically")
    print("    4. Next expression starts")
    print()
    print("  Controls:")
    print("    n = skip this expression")
    print("    q = quit")
    print("=" * 60)
    print("\nPress any key to begin...")

    try:
        # Wait for any key
        sys.stdin.read(1)

        for i, (name, duration, instruction) in enumerate(prompts):
            if i < start_at:
                continue

            print(f"\n{'='*60}")
            print(f"  EXPRESSION {i+1}/{len(prompts)}")
            print(f"{'='*60}")
            print(f"\n  >>> {instruction} <<<\n")
            print(f"  Press the hotkey NOW, recording in 5 seconds...")
            print(f"  (n=skip, q=quit)")

            # Countdown - they press VTS hotkey during this
            skip = False
            for countdown in [5, 4, 3, 2, 1]:
                key = check_key()
                if key == 'n':
                    print("  [SKIPPED]")
                    skip = True
                    break
                if key == 'q':
                    print(f"\n\nStopped at {i+1}. Resume with: --start-at {i}")
                    return
                print(f"  {countdown}...")
                await asyncio.sleep(1)

            if skip:
                continue

            print("\n  🔴 RECORDING!\n")

            # Record
            frames = []
            frame_time = 1.0 / fps
            start_time = time.time()
            skipped = False

            while True:
                loop_start = time.time()
                elapsed = loop_start - start_time

                key = check_key()
                if key == 'n':
                    print("\n  [SKIPPED]")
                    skipped = True
                    break
                if key == 'q':
                    print(f"\n\nStopped at {i+1}. Resume with: --start-at {i}")
                    return

                if elapsed >= duration:
                    break

                params = await vts.get_params()
                frames.append(params)

                remaining = duration - elapsed
                bar_len = 30
                bar_fill = int((elapsed / duration) * bar_len)
                bar = "█" * bar_fill + "░" * (bar_len - bar_fill)
                print(f"\r  [{bar}] {remaining:.1f}s", end="", flush=True)

                loop_time = time.time() - loop_start
                if loop_time < frame_time:
                    await asyncio.sleep(frame_time - loop_time)

            if skipped or len(frames) < 10:
                continue

            print("\n\n  ✓ Saving...")

            important = find_important_params(frames)
            compressed = [{p: f[p] for p in important} for f in frames]

            clip_data = {
                "name": name,
                "fps": fps,
                "duration": len(frames) / fps,
                "n_frames": len(frames),
                "params": important,
                "frames": compressed,
                "recorded": datetime.now().isoformat(),
                "loopable": True,
                "is_expression": True,
            }

            clip_path = CLIPS_DIR / f"{name}.json"
            with open(clip_path, 'w') as f:
                json.dump(clip_data, f)

            print(f"  ✓ Saved: {name}")

        print("\n" + "=" * 60)
        print("  EXPRESSIONS COMPLETE!")
        print("=" * 60)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def check_key():
    """Non-blocking key check. Returns key or None."""
    import sys
    import select
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


async def run_session(prompts: list, fps: int, host: str, port: int, start_at: int = 0):
    """Run a guided capture session with multiple clips."""
    import sys
    import tty
    import termios

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    vts = VTSConnection(host, port)
    if not await vts.connect():
        print("Failed to connect to VTS")
        return

    # Setup terminal for single key input
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    # Estimate time
    total_time = sum(p[1] for p in prompts) + len(prompts) * 6
    mins = int(total_time // 60)
    secs = int(total_time % 60)

    print("=" * 60)
    print("  GUIDED CAPTURE SESSION")
    print("=" * 60)
    print(f"  {len(prompts)} clips to record (starting at {start_at + 1})")
    print(f"  Estimated time: {mins}m {secs}s")
    print()
    print("  Controls:")
    print("    SPACE = start recording")
    print("    n     = skip this clip")
    print("    q     = quit session")
    print("=" * 60)
    print("\nPress SPACE to begin...")

    try:
        # Wait for space to start
        while True:
            key = check_key()
            if key == ' ':
                break
            if key == 'q':
                print("\nQuitting.")
                return
            await asyncio.sleep(0.05)

        for i, (name, duration, instruction) in enumerate(prompts):
            if i < start_at:
                continue

            print(f"\n{'='*60}")
            print(f"  CLIP {i+1}/{len(prompts)}: {name}")
            print(f"{'='*60}")
            print(f"\n  >>> {instruction} <<<\n")
            print(f"  Duration: {duration} seconds")
            print(f"  SPACE=record  n=skip  q=quit")

            # Wait for input
            while True:
                key = check_key()
                if key == ' ':
                    break
                if key == 'n':
                    print("  [SKIPPED]")
                    break
                if key == 'q':
                    print(f"\n\nStopped at clip {i+1}. Resume with: --start-at {i}")
                    return
                await asyncio.sleep(0.05)

            if key == 'n':
                continue

            # Quick countdown
            for countdown in [3, 2, 1]:
                print(f"  {countdown}...")
                await asyncio.sleep(0.7)

            print("\n  🔴 RECORDING!\n")

            # Record
            frames = []
            frame_time = 1.0 / fps
            start_time = time.time()
            skipped = False

            while True:
                loop_start = time.time()
                elapsed = loop_start - start_time

                # Check for skip/quit during recording
                key = check_key()
                if key == 'n':
                    print("\n  [SKIPPED MID-RECORD]")
                    skipped = True
                    break
                if key == 'q':
                    print(f"\n\nStopped at clip {i+1}. Resume with: --start-at {i}")
                    return

                if elapsed >= duration:
                    break

                params = await vts.get_params()
                frames.append(params)

                remaining = duration - elapsed
                bar_len = 30
                bar_fill = int((elapsed / duration) * bar_len)
                bar = "█" * bar_fill + "░" * (bar_len - bar_fill)
                print(f"\r  [{bar}] {remaining:.1f}s", end="", flush=True)

                loop_time = time.time() - loop_start
                if loop_time < frame_time:
                    await asyncio.sleep(frame_time - loop_time)

            if skipped or len(frames) < 10:
                continue

            print("\n\n  ✓ Saving...")

            # Process and save
            important = find_important_params(frames)
            compressed = [{p: f[p] for p in important} for f in frames]

            clip_data = {
                "name": name,
                "fps": fps,
                "duration": len(frames) / fps,
                "n_frames": len(frames),
                "params": important,
                "frames": compressed,
                "recorded": datetime.now().isoformat(),
                "loopable": "idle" in name or "loop" in name,
            }

            clip_path = CLIPS_DIR / f"{name}.json"
            with open(clip_path, 'w') as f:
                json.dump(clip_data, f)

            print(f"  ✓ Saved: {name} ({len(frames)} frames, {len(important)} params)")

        print("\n" + "=" * 60)
        print("  SESSION COMPLETE!")
        print("=" * 60)
        list_clips()

    finally:
        # Restore terminal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main():
    parser = argparse.ArgumentParser(description="Capture motion clips")
    parser.add_argument("name", nargs="?", help="Clip name (e.g. idle_happy_01)")
    parser.add_argument("--duration", "-d", type=float, default=15, help="Recording duration (seconds)")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument("--host", default="localhost", help="VTS host")
    parser.add_argument("--port", type=int, default=8001, help="VTS port")
    parser.add_argument("--list", "-l", action="store_true", help="List all clips")
    parser.add_argument("--preview", "-p", action="store_true", help="Preview a clip")
    parser.add_argument("--session", "-s", action="store_true", help="Run guided capture session")
    parser.add_argument("--category", "-c", choices=["all", "idle", "talk", "react", "head", "face", "filler", "special", "expr", "remaining"],
                        default="all", help="Which category to record")
    parser.add_argument("--start-at", type=int, default=0, help="Skip to clip N in session")
    parser.add_argument("--show-categories", action="store_true", help="Show all categories and clip counts")
    parser.add_argument("--expressions", "-e", action="store_true", help="Interactive expression toggle capture")
    parser.add_argument("--expr-duration", type=float, default=8, help="Duration for expression captures")
    args = parser.parse_args()

    if args.expressions:
        asyncio.run(capture_expressions(args.fps, args.host, args.port, args.expr_duration))
    elif args.show_categories:
        print("CAPTURE SESSION CATEGORIES:\n")
        for cat, desc in CATEGORIES.items():
            clips = filter_session(DEFAULT_SESSION, cat)
            time_est = sum(p[1] for p in clips) + len(clips) * 6
            mins = int(time_est // 60)
            print(f"  {cat:8s} - {len(clips):3d} clips (~{mins}m) - {desc}")
        print(f"\n  {'all':8s} - {len(DEFAULT_SESSION):3d} clips total (not including expr)")
        print("\nRun with: python3 capture_clip.py --session --category idle")
        print("          python3 capture_clip.py --session --category expr")
    elif args.list:
        list_clips()
    elif args.session:
        if args.category == "expr":
            asyncio.run(run_expression_session(EXPRESSION_SESSION, args.fps, args.host, args.port, args.start_at))
        else:
            prompts = filter_session(DEFAULT_SESSION, args.category)
            asyncio.run(run_session(prompts, args.fps, args.host, args.port, args.start_at))
    elif args.preview and args.name:
        asyncio.run(preview_clip(args.name, args.host, args.port))
    elif args.name:
        asyncio.run(capture_clip(args.name, args.duration, args.fps, args.host, args.port))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 capture_clip.py --session          # Guided session (recommended)")
        print("  python3 capture_clip.py idle_neutral_01    # Single clip")
        print("  python3 capture_clip.py --list             # List clips")
        print("  python3 capture_clip.py --preview idle_neutral_01")


if __name__ == "__main__":
    main()
