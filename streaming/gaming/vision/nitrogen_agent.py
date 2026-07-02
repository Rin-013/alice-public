"""
NitroGen Agent
==============

Wrapper around the NitroGen vision-to-action model (493M params).
Produces GamepadAction from game frames at ~60fps on GPU0.

Fully stubbed with MockNitroGen for dev/testing.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from streaming.gaming.types import CapturedFrame, GamepadAction
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class NitroGenAgent(ABC):
    """Abstract base for NitroGen inference."""

    @abstractmethod
    def load(self) -> bool:
        """Load the model. Returns True on success."""
        ...

    @abstractmethod
    def predict(self, frame: CapturedFrame) -> GamepadAction:
        """Predict gamepad action from a frame."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...


class MockNitroGen(NitroGenAgent):
    """
    Mock NitroGen that returns slightly noisy neutral actions.

    Simulates a real model's output for pipeline testing.
    """

    def __init__(self, noise: float = 0.1):
        self._noise = noise
        self._ready = False
        self._predict_count = 0

    def load(self) -> bool:
        self._ready = True
        logger.info("MockNitroGen loaded (noise=%.2f)", self._noise)
        return True

    def predict(self, frame: CapturedFrame) -> GamepadAction:
        if not self._ready:
            return GamepadAction.neutral()

        self._predict_count += 1

        # Generate noisy neutral action
        buttons = np.random.normal(0.0, self._noise, 20).astype(np.float32)

        # Clamp appropriately
        buttons[:8] = np.clip(buttons[:8], 0.0, 1.0)    # Digital + triggers
        buttons[8:12] = np.clip(buttons[8:12], -1.0, 1.0)  # Sticks
        buttons[12:] = np.clip(buttons[12:], 0.0, 1.0)  # D-pad + buttons

        return GamepadAction(buttons=buttons, source="mock_nitrogen")

    def is_ready(self) -> bool:
        return self._ready

    @property
    def predict_count(self) -> int:
        return self._predict_count


class NitroGenWrapper(NitroGenAgent):
    """
    Stub for real NitroGen model inference.

    Requires CUDA GPU + NitroGen model weights.
    """

    def __init__(self, model_path: str = "models/nitrogen-493m/", device: str = "cuda:0"):
        self._model_path = model_path
        self._device = device
        self._model = None
        self._ready = False
        logger.info(f"NitroGenWrapper stub initialized (path={model_path}, device={device})")

    def load(self) -> bool:
        logger.error("NitroGenWrapper.load() — stub, requires CUDA + model weights")
        return False

    def predict(self, frame: CapturedFrame) -> GamepadAction:
        return GamepadAction.neutral()

    def is_ready(self) -> bool:
        return self._ready
