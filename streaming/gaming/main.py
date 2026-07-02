"""
Alice Gaming - Main Entry Point
================================

GamingSession orchestrates the full gaming pipeline:
- Capture loop (frame acquisition)
- Vision loop (NitroGen + game state extraction)
- Director loop (strategic decisions at ~6 Hz)
- Mixer (blend NitroGen + directives each frame)
- Commentary pipeline (TTS output)
- Input controller (send actions to game)

Usage:
    python gaming/main.py --mock        # Run with all mocks (dev/testing)
    python gaming/main.py --game elden  # Load game-specific config
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import time
from typing import Dict, Optional

from streaming.gaming.capture.base import MockCapture, CaptureSource
from streaming.gaming.capture.frame_buffer import FrameBuffer
from streaming.gaming.cognitive.behavior_modes import BehaviorModeManager
from streaming.gaming.cognitive.director import AliceDirector
from streaming.gaming.cognitive.directive import DirectiveBuffer
from streaming.gaming.cognitive.emotion_bridge import EmotionBridge
from streaming.gaming.comms.gpu_bridge import GPUBridge, LocalBridge
from streaming.gaming.input.base import InputController, MockController
from streaming.gaming.mixer.directive_mixer import DirectiveMixer
from streaming.gaming.stream.commentary import CommentaryPipeline
from streaming.gaming.types import BehaviorMode, CapturedFrame, GamepadAction, GameState
from streaming.gaming.utils.config import GamingConfig
from streaming.gaming.utils.logging_utils import get_logger
from streaming.gaming.utils.timing import FrameTimer
from streaming.gaming.vision.game_state import GameStateExtractor, MockGameStateExtractor
from streaming.gaming.vision.nitrogen_agent import MockNitroGen, NitroGenAgent

logger = get_logger(__name__)


class GamingSession:
    """
    Main orchestrator for Alice's gaming session.

    Manages the dual-loop architecture:
    - Fast loop (~60 Hz): capture → vision → mixer → input
    - Slow loop (~6 Hz): director → directives + commentary
    """

    def __init__(
        self,
        config: Optional[GamingConfig] = None,
        capture: Optional[CaptureSource] = None,
        nitrogen: Optional[NitroGenAgent] = None,
        controller: Optional[InputController] = None,
        bridge: Optional[GPUBridge] = None,
        mock: bool = False,
    ):
        self._config = config or GamingConfig()
        self._mock = mock

        # Components
        self._capture = capture or (MockCapture() if mock else MockCapture())
        self._nitrogen = nitrogen or (MockNitroGen() if mock else MockNitroGen())
        self._controller = controller or (MockController() if mock else MockController())
        self._bridge = bridge or LocalBridge()

        # Frame buffer
        self._frame_buffer = FrameBuffer()

        # State extraction
        self._state_extractor: GameStateExtractor = (
            MockGameStateExtractor() if mock else GameStateExtractor()
        )

        # Cognitive
        self._directive_buffer = DirectiveBuffer()
        self._mode_manager = BehaviorModeManager(
            initial_mode=BehaviorMode(
                self._config.get("behavior.default_mode", "entertainer")
            ),
            transition_cooldown_sec=self._config.get(
                "behavior.transition_cooldown_sec", 5.0
            ),
        )
        self._emotion_bridge = EmotionBridge()
        self._director = AliceDirector(
            mode_manager=self._mode_manager,
            emotion_bridge=self._emotion_bridge,
            directive_buffer=self._directive_buffer,
            config=self._config.section("director"),
        )

        # Mixer
        self._mixer = DirectiveMixer(
            directive_buffer=self._directive_buffer,
            base_weight=self._config.get("mixer.base_weight", 0.3),
            urgency_scale=self._config.get("mixer.urgency_scale", 0.7),
            max_weight=self._config.get("mixer.max_weight", 0.95),
            smoothing=self._config.get("mixer.smoothing", 0.15),
            passthrough_on_empty=self._config.get("mixer.passthrough_on_empty", True),
        )

        # Commentary
        self._commentary = CommentaryPipeline(
            min_gap_sec=self._config.get("commentary.min_gap_sec", 3.0),
            max_queue_size=self._config.get("commentary.max_queue_size", 10),
            ttl_sec=self._config.get("commentary.ttl_sec", 15.0),
            interrupt_priority=self._config.get("commentary.interrupt_priority", 3),
            default_tts_instruct=self._config.get(
                "commentary.default_tts_instruct",
                "Speak naturally with a playful tone",
            ),
        )

        # Timers
        self._main_timer = FrameTimer(
            target_fps=self._config.get("timing.main_loop_fps", 60)
        )
        self._director_timer = FrameTimer(
            target_fps=self._config.get("timing.director_fps", 6)
        )

        # State
        self._running = False
        self._frame_count = 0
        self._current_game_state = GameState()

        logger.info(
            f"GamingSession initialized (mock={mock}, "
            f"main_fps={self._main_timer.target_fps}, "
            f"director_fps={self._director_timer.target_fps})"
        )

    # --- Lifecycle ---

    def start(self) -> bool:
        """Initialize all components and start the session."""
        logger.info("Starting gaming session...")

        # Connect bridge
        self._bridge.connect()

        # Start capture
        if not self._capture.start():
            logger.error("Failed to start capture")
            return False

        # Load NitroGen
        if not self._nitrogen.load():
            logger.warning("NitroGen failed to load — using neutral actions")

        # Connect controller
        if not self._controller.connect():
            logger.warning("Controller failed to connect — actions will be logged only")

        self._running = True
        logger.info("Gaming session started")
        return True

    def stop(self):
        """Stop all components."""
        self._running = False
        self._capture.stop()
        self._controller.disconnect()
        self._bridge.disconnect()
        logger.info("Gaming session stopped")

    # --- Main loop ---

    def run_sync(self, max_frames: int = 0):
        """
        Run the gaming loop synchronously.

        Args:
            max_frames: Stop after this many frames (0 = infinite).
        """
        if not self.start():
            return

        snapshot_interval = self._config.get("vision.snapshot_interval_frames", 10)
        director_interval = int(
            self._main_timer.target_fps / self._director_timer.target_fps
        )

        try:
            while self._running:
                self._main_timer.tick()
                self._frame_count += 1

                # --- Fast loop (every frame) ---

                # 1. Capture
                frame = self._capture.grab()
                if frame is not None:
                    self._frame_buffer.write(frame)

                # 2. Read latest frame
                latest_frame = self._frame_buffer.read()
                if latest_frame is None:
                    self._main_timer.wait()
                    continue

                # 3. NitroGen predict
                nitrogen_action = self._nitrogen.predict(latest_frame)

                # 4. Mix with directives
                mixed_action = self._mixer.mix(
                    nitrogen_action, mode=self._mode_manager.mode
                )

                # 5. Send to controller
                self._controller.send(mixed_action)

                # --- Slow loop (every N frames) ---

                # Game state extraction (every snapshot_interval frames)
                if self._frame_count % snapshot_interval == 0:
                    self._current_game_state = self._state_extractor.extract(
                        latest_frame
                    )

                # Director tick
                if self._frame_count % director_interval == 0:
                    self._director.tick(self._current_game_state)

                    # Feed director commentary into pipeline
                    commentary_items = self._director.pop_commentary()
                    self._commentary.submit_many(commentary_items)

                # Commentary check
                next_line = self._commentary.next()
                if next_line is not None:
                    logger.info(
                        f"[COMMENTARY] ({next_line.priority.name}) {next_line.text}"
                    )
                    # In production: send to TTS
                    self._commentary.mark_done()

                # Frame limit
                if max_frames > 0 and self._frame_count >= max_frames:
                    break

                self._main_timer.wait()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    # --- Status ---

    def get_status(self) -> Dict:
        return {
            "running": self._running,
            "frame_count": self._frame_count,
            "main_fps": round(self._main_timer.fps, 1),
            "mode": self._mode_manager.mode.value,
            "mixer": self._mixer.get_status(),
            "director": self._director.get_status(),
            "commentary": self._commentary.get_status(),
            "mock": self._mock,
        }


# --- CLI entry point ---


def main():
    parser = argparse.ArgumentParser(description="Alice Gaming Session")
    parser.add_argument(
        "--mock", action="store_true", help="Run with all mock components"
    )
    parser.add_argument("--hardware", type=str, default=None, help="Hardware config (pc, xbox)")
    parser.add_argument("--game", type=str, default=None, help="Game config name")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to run (0 = infinite)",
    )
    args = parser.parse_args()

    config = GamingConfig(hardware=args.hardware, game=args.game)
    session = GamingSession(config=config, mock=args.mock or True)

    # Handle Ctrl+C
    def _signal_handler(sig, frame):
        session.stop()

    signal.signal(signal.SIGINT, _signal_handler)

    logger.info("Running gaming session (Ctrl+C to stop)...")
    session.run_sync(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
