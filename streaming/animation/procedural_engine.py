#!/usr/bin/env python3
"""
Procedural engine - generative animation runner.

Parallel to motion_engine.py. Same VTS plumbing (VTSConnection + ExpressionEngine
imported from the existing module - no duplication), but ticks ProceduralMotion
every frame instead of pulling pre-recorded clips. No clip files needed.

motion_engine.py is unchanged and remains the production path auto-launched
by chat.py. Run this one manually to test the generative motion:

    python streaming/animation/procedural_engine.py
    python streaming/animation/procedural_engine.py --emotion excited

Keyboard:
    1-9   set emotion (neutral/happy/excited/sassy/sad/angry/surprised/thinking/tired)
    e     trigger an audio onset pulse (preview brow pop)
    q     quit

Speech reactivity:
    By default this taps Alice's voice off VB-Cable ("CABLE Output") so her
    head nods/leans/pops *with what she says* instead of drifting randomly.
    Run chat.py alongside this to see it. To test by talking into your own
    mic instead, pass --audio-device "Microphone". --no-audio disables it.
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from dataclasses import asdict

from audio_listener import AudioListener, resolve_device
from emotion_state import EMOTION_STATE_FILE, EMOTIONS, EmotionState, get_emotion
from expression_engine import EXPRESSION_DISPLAY_NAMES, ExpressionEngine
from motion_engine import VTSConnection   # reuse the existing VTS client
from param_mapper import get_core_animation_params, map_params
from procedural_motion import ProceduralMotion


def _force_emotion(emotion: str) -> None:
    """Reset emotion_state.json to a clean state with `emotion` dominant.

    Bypasses the accumulator in set_emotion(). Used at startup and on
    keyboard override so this test runner doesn't inherit stale state
    from a prior Alice session (which is why she defaulted to crying).
    """
    state = EmotionState(emotion=emotion, confidence=1.0, timestamp=time.time())
    state.scores = {e: 0.0 for e in EMOTIONS}
    state.scores[emotion] = 1.0
    with open(EMOTION_STATE_FILE, 'w') as f:
        json.dump(asdict(state), f, indent=2)


async def run(args):
    motion = ProceduralMotion(seed=args.seed)

    # Clobber any stale emotion left over from a prior chat.py session.
    _force_emotion(args.emotion)

    # Speech tap: the wire that makes motion react to Alice's voice instead
    # of drifting randomly. Falls back to idle-only motion if it can't open.
    listener = None
    if not args.no_audio:
        dev = resolve_device(args.audio_device) if args.audio_device else None
        listener = AudioListener(device=dev)
        if not listener.start():
            listener = None
            print("Audio reactivity OFF - motion will be idle-only "
                  "(start chat.py so Alice speaks into VB-Cable, or pass --audio-device \"Microphone\")")

    vts = VTSConnection(args.host, args.port)
    if not await vts.connect():
        print("Failed to connect to VTS")
        return

    # Wipe stale emotion-driven expressions (crying, heart_eyes, etc.) left
    # active from a prior session. Only touches expressions ExpressionEngine
    # manages - leaves user toggles like "hide watermark" alone.
    managed_names = {n.lower() for n in EXPRESSION_DISPLAY_NAMES.values()}
    expressions = await vts.get_expression_states()
    cleared = 0
    for expr in expressions:
        if not expr.get("active"):
            continue
        name = expr.get("name", "").lower().strip()
        if name in managed_names or any(m in name for m in managed_names):
            await vts.set_expression_active(expr.get("file", ""), False)
            cleared += 1
    if cleared:
        print(f"Cleared {cleared} stale emotion expression(s) from prior session")

    expression_engine = None
    if args.expressions:
        expression_engine = ExpressionEngine(vts)
        await expression_engine.init_hotkeys()
        await expression_engine.update_emotion(args.emotion)
    else:
        print("Expressions DISABLED (pass --expressions to enable VTS hotkey overlays)")

    print(f"\nProcedural engine running. Starting emotion: {args.emotion}")
    print("=" * 50)
    print("1-9 emotion  |  e onset pulse  |  [ / ] eye openness  |  q quit")
    print("=" * 50)

    stop_flag = False
    def handle_stop(_sig, _frame):
        nonlocal stop_flag
        stop_flag = True
    signal.signal(signal.SIGINT, handle_stop)

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

    fps = 40
    frame_time = 1.0 / fps
    last_emotion_check = 0.0
    emotion_check_interval = 0.1
    last_known_emotion = args.emotion
    frame_count = 0

    try:
        while not stop_flag:
            loop_start = time.time()

            # Pull current emotion from the shared file (Mind writes it).
            if time.time() - last_emotion_check > emotion_check_interval:
                state = get_emotion()
                if state.emotion != last_known_emotion:
                    if expression_engine is not None:
                        await expression_engine.update_emotion(state.emotion)
                    last_known_emotion = state.emotion
                last_emotion_check = time.time()
            else:
                state = get_emotion()  # cheap; file is tiny

            # Keyboard
            if _is_windows:
                has_key = msvcrt.kbhit()
            elif sys.stdin.isatty():
                has_key = select.select([sys.stdin], [], [], 0)[0]
            else:
                has_key = False
            if has_key:
                key = msvcrt.getwch() if _is_windows else sys.stdin.read(1)
                if key == 'q':
                    break
                elif key == 'e':
                    motion.trigger_onset(1.0)
                elif key == '[':
                    motion.eye_open_rest = max(0.20, motion.eye_open_rest - 0.05)
                    print(f"\neye_open_rest = {motion.eye_open_rest:.2f}  (narrower)")
                elif key == ']':
                    motion.eye_open_rest = min(1.20, motion.eye_open_rest + 0.05)
                    print(f"\neye_open_rest = {motion.eye_open_rest:.2f}  (wider)")
                elif key in '123456789':
                    idx = int(key) - 1
                    if idx < len(EMOTIONS):
                        emo = EMOTIONS[idx]
                        _force_emotion(emo)  # bypass accumulator - immediate
                        if expression_engine is not None:
                            await expression_engine.update_emotion(emo)
                        last_known_emotion = emo
                        print(f"\nEmotion: {emo}")

            # Pull Alice's live voice envelope + emphasis onsets.
            audio_env = 0.0
            if listener is not None:
                audio_env = listener.get_env()
                onset = listener.poll_onset()
                if onset:
                    motion.trigger_onset(onset)

            # Generate one frame, driven by her speech.
            raw = motion.tick(frame_time, state, audio_env=audio_env)
            vts_params = map_params(raw)
            core = get_core_animation_params(vts_params)
            if core:
                await vts.set_params(core)

            frame_count += 1
            if frame_count % 40 == 0:  # once a second
                env_bar = "#" * int(audio_env * 20)
                print(f"\r[{state.emotion}] eye={raw['EyeOpenLeft']:.2f} "
                      f"head=({raw['FaceAngleX']:+.1f},{raw['FaceAngleY']:+.1f},{raw['FaceAngleZ']:+.1f})  "
                      f"lean={raw['FacePositionZ']:+.2f}  voice|{env_bar:<20}|",
                      end="", flush=True)

            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                await asyncio.sleep(frame_time - elapsed)

    finally:
        if listener is not None:
            listener.stop()
        if _old_settings is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _old_settings)

    print(f"\nStopped. Frames: {frame_count}")


def main():
    parser = argparse.ArgumentParser(description="Procedural animation runner (parallel to motion_engine.py)")
    parser.add_argument("--emotion", "-e", default="neutral", choices=EMOTIONS,
                        help="Starting emotion")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the RNG for reproducible motion (default: random)")
    parser.add_argument("--expressions", action="store_true",
                        help="Enable VTS hotkey expression overlays (off by default for clean motion testing)")
    parser.add_argument("--audio-device", default=None,
                        help="Audio capture device name substring (default: VB-Cable 'CABLE Output'). "
                             "Use 'Microphone' to drive motion by talking into your mic.")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable speech reactivity (idle motion only)")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
