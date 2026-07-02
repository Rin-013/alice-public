#!/usr/bin/env python3
"""
Test Suite - Thorough test of Movement Engine + Expression Engine.

Connects to VTS and runs through each system with pauses so you can watch Alice.
Press Enter to advance through each test, or 's' + Enter to skip a section.

Tests:
  1. Expression Engine - each expression individually, then clear
  2. Movement Engine - each emotion's clips, verify variety
  3. Combined - emotion changes trigger both motion + expressions
"""

import asyncio
import json
import time
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Install: pip3 install websockets")
    exit(1)

from motion_engine import (
    VTSConnection, ClipLibrary, ClipPlayer, BlendLayer,
    Orchestrator, CLIPS_DIR, EMOTIONS
)
from expression_engine import (
    ExpressionEngine, EMOTION_EXPRESSIONS, EXPRESSION_DISPLAY_NAMES
)
from param_mapper import map_params, get_core_animation_params


HOLD_TIME = 4  # Seconds to hold each test state


async def prompt(msg, vts=None, expression_engine=None):
    """Wait for user input. Reconnects VTS after since input() blocks the event loop."""
    print(f"\n{msg}")
    print("  [Enter] continue  |  [s] skip section  |  [q] quit")
    resp = await asyncio.get_event_loop().run_in_executor(
        None, lambda: input("  > ").strip().lower()
    )
    if resp == 'q':
        print("Quitting.")
        sys.exit(0)
    # Reconnect VTS - input() blocks event loop so websocket pings time out
    if vts:
        await vts.ensure_connected()
        # Re-init hotkeys since they're tied to the VTS session
        if expression_engine:
            await expression_engine.init_hotkeys()
    return resp != 's'


async def play_motion(player, blend, vts, seconds):
    """Run the motion loop for N seconds, sending frames to VTS."""
    fps = 40
    frame_time = 1.0 / fps
    frames = int(seconds * fps)
    for _ in range(frames):
        start = time.time()
        raw = player.get_frame(frame_time)
        final = blend.apply(raw, frame_time)
        vts_params = map_params(final)
        core = get_core_animation_params(vts_params)
        if core:
            await vts.set_params(core)
        elapsed = time.time() - start
        if elapsed < frame_time:
            await asyncio.sleep(frame_time - elapsed)


async def run_tests():
    # --- Setup ---
    print("=" * 60)
    print("  MOTION + EXPRESSION ENGINE TEST SUITE")
    print("=" * 60)

    # Load clips
    library = ClipLibrary()
    library.load_all(CLIPS_DIR)
    if not library.clips:
        print("\nNo clips found!")
        return

    player = ClipPlayer()
    blend = BlendLayer()
    orchestrator = Orchestrator(library, player)

    # Connect to VTS
    vts = VTSConnection()
    if not await vts.connect():
        print("Failed to connect to VTS!")
        return
    print("Connected to VTS")

    # Init expression engine
    expression_engine = ExpressionEngine(vts)
    await expression_engine.init_hotkeys()
    available = expression_engine.get_available_expressions()

    if not available:
        print("\nWARNING: No expressions matched VTS hotkeys!")
        print("Expression tests will be skipped.")

    # Start neutral motion so Alice isn't frozen during tests
    orchestrator.set_emotion("neutral")

    # =============================================
    # TEST 1: Expression Engine - Individual
    # =============================================
    print("\n" + "=" * 60)
    print("  TEST 1: EXPRESSION ENGINE - Individual Expressions")
    print("=" * 60)
    print(f"  {len(available)} expressions matched VTS hotkeys")
    print(f"  Each expression will show for {HOLD_TIME}s then clear")

    if available and await prompt("Start expression test?", vts, expression_engine):
        # Refresh available list after possible reconnect
        available = expression_engine.get_available_expressions()
        for i, expr_name in enumerate(available):
            print(f"\n  [{i+1}/{len(available)}] Triggering: {expr_name}")

            # Trigger expression
            await expression_engine.trigger_by_name(expr_name)

            # Keep motion running while expression shows
            await play_motion(player, blend, vts, HOLD_TIME)

            # Clear
            await expression_engine._clear_all_expressions()
            print(f"    Cleared: {expr_name}")

            # Brief pause between expressions
            await play_motion(player, blend, vts, 1.0)

        print("\n  Expression individual test complete!")

    # =============================================
    # TEST 2: Expression Engine - Emotion Mapping
    # =============================================
    print("\n" + "=" * 60)
    print("  TEST 2: EXPRESSION ENGINE - Emotion Triggers")
    print("=" * 60)
    print("  Testing emotion → expression mapping")

    emotions_with_expressions = [
        e for e, exprs in EMOTION_EXPRESSIONS.items() if exprs
    ]
    print(f"  Emotions with expressions: {emotions_with_expressions}")

    if await prompt("Start emotion-expression test?", vts, expression_engine):
        for emotion in emotions_with_expressions:
            expected = EMOTION_EXPRESSIONS[emotion]
            avail = [e for e in expected if e in expression_engine.hotkey_ids]
            print(f"\n  Emotion: {emotion}")
            print(f"    Expected: {expected}")
            print(f"    Available: {avail}")

            if not avail:
                print(f"    SKIP - no matched hotkeys")
                continue

            # Trigger via emotion (picks random from available)
            await expression_engine.update_emotion(emotion)

            # Show active
            active = [a.name for a in expression_engine.active_expressions]
            print(f"    Active: {active}")

            # Hold with motion
            await play_motion(player, blend, vts, HOLD_TIME)

            # Return to neutral (clears expression)
            await expression_engine.update_emotion("neutral")
            await play_motion(player, blend, vts, 1.5)

        print("\n  Emotion-expression mapping test complete!")

    # =============================================
    # TEST 3: Movement Engine - Clip Selection
    # =============================================
    print("\n" + "=" * 60)
    print("  TEST 3: MOVEMENT ENGINE - Clip Selection per Emotion")
    print("=" * 60)
    print("  Testing clip variety and no-repeat logic")

    if await prompt("Start movement test?", vts, expression_engine):
        for emotion in EMOTIONS:
            clips = library._get_motion_clips_for_emotion(emotion)
            clip_names = [c.name for c in clips]
            print(f"\n  Emotion: {emotion} ({len(clips)} clips available)")

            # Pick 3 clips to verify variety
            picks = []
            recent = []
            for j in range(min(3, len(clips))):
                clip = library.get_weighted_for_emotion(emotion, recent)
                if clip:
                    picks.append(clip.name)
                    recent.append(clip.name)

            unique = len(set(picks))
            print(f"    Picked: {picks}")
            print(f"    Unique: {unique}/{len(picks)} {'(good variety)' if unique == len(picks) else '(repeats!)'}")

            # Check no expr_ clips leaked through
            expr_leaks = [c for c in clip_names if c.startswith('expr_')]
            if expr_leaks:
                print(f"    BUG: expr_* clips in pool: {expr_leaks}")
            else:
                print(f"    No expr_* clips in pool (correct)")

            # Play this emotion for a few seconds
            orchestrator.set_emotion(emotion)
            current = player.current_clip.name if player.current_clip else "none"
            print(f"    Playing: {current}")
            await play_motion(player, blend, vts, HOLD_TIME)

        # Return to neutral
        orchestrator.set_emotion("neutral")
        print("\n  Movement engine test complete!")

    # =============================================
    # TEST 4: Combined - Full Emotion Cycle
    # =============================================
    print("\n" + "=" * 60)
    print("  TEST 4: COMBINED - Motion + Expressions Together")
    print("=" * 60)
    print(f"  Cycling through all {len(EMOTIONS)} emotions")
    print(f"  {HOLD_TIME+2}s per emotion (motion + expression)")

    if await prompt("Start combined test?", vts, expression_engine):
        for emotion in EMOTIONS:
            has_expr = bool(EMOTION_EXPRESSIONS.get(emotion, []))
            print(f"\n  >>> {emotion.upper()} {'(+expression)' if has_expr else '(motion only)'}")

            # Set both
            orchestrator.set_emotion(emotion)
            await expression_engine.update_emotion(emotion)

            clip_name = player.current_clip.name if player.current_clip else "none"
            active_expr = [a.name for a in expression_engine.active_expressions]
            print(f"      Clip: {clip_name}")
            print(f"      Expression: {active_expr if active_expr else 'none'}")

            await play_motion(player, blend, vts, HOLD_TIME + 2)

        # End on neutral
        orchestrator.set_emotion("neutral")
        await expression_engine.update_emotion("neutral")
        await play_motion(player, blend, vts, 2.0)

        print("\n  Combined test complete!")

    # =============================================
    # TEST 5: Micro-movements (idle)
    # =============================================
    print("\n" + "=" * 60)
    print("  TEST 5: MICRO-MOVEMENTS - Idle Behavior")
    print("=" * 60)
    print("  Neutral emotion, 15 seconds. Watch for:")
    print("    - Eye saccades (small gaze shifts every 2-6s)")
    print("    - Subtle head drift")
    print("    - Breathing (tiny head pitch)")

    if await prompt("Start idle test?", vts, expression_engine):
        orchestrator.set_emotion("neutral")
        await expression_engine.update_emotion("neutral")
        print("\n  Running idle for 15 seconds... watch her eyes and head")
        await play_motion(player, blend, vts, 15.0)
        print("\n  Idle test complete!")

    # =============================================
    # Summary
    # =============================================
    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)
    print(f"  Expressions matched: {len(available)}/{len(EXPRESSION_DISPLAY_NAMES)}")
    print(f"  Total clips loaded: {len(library.clips)}")
    print(f"  Emotions with clips: {len(library.by_emotion)}")
    print(f"  Emotions with expressions: {len(emotions_with_expressions)}")

    if available:
        unmatched = [n for n in EXPRESSION_DISPLAY_NAMES if n not in expression_engine.hotkey_ids]
        if unmatched:
            print(f"\n  Unmatched expressions: {unmatched}")
            print("  (VTS hotkey names may not match our display names)")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(run_tests())
