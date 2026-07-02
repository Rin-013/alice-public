"""
GPU Bridge
==========

Abstract IPC between GPU0 (NitroGen / capture) and GPU1 (Alice brain).

LocalBridge: in-process (for Mac dev, single GPU)
SocketBridge: TCP socket (for Windows prod, dual GPU)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Optional

from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GPUBridge(ABC):
    """Abstract base for GPU-to-GPU communication."""

    @abstractmethod
    def send(self, channel: str, data: Dict[str, Any]) -> bool:
        """Send data on a named channel. Returns True on success."""
        ...

    @abstractmethod
    def receive(self, channel: str) -> Optional[Dict[str, Any]]:
        """Receive data from a named channel. Returns None if empty."""
        ...

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self):
        """Close the connection."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...


class LocalBridge(GPUBridge):
    """
    In-process bridge for single-GPU dev (Mac).

    Uses deques as channels — send() appends, receive() poplefts.
    """

    def __init__(self, max_buffer: int = 64):
        self._channels: Dict[str, deque] = {}
        self._max_buffer = max_buffer
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        logger.info("LocalBridge connected (in-process)")
        return True

    def disconnect(self):
        self._connected = False
        self._channels.clear()

    def is_connected(self) -> bool:
        return self._connected

    def send(self, channel: str, data: Dict[str, Any]) -> bool:
        if not self._connected:
            return False
        if channel not in self._channels:
            self._channels[channel] = deque(maxlen=self._max_buffer)
        self._channels[channel].append(data)
        return True

    def receive(self, channel: str) -> Optional[Dict[str, Any]]:
        if not self._connected:
            return None
        ch = self._channels.get(channel)
        if ch and len(ch) > 0:
            return ch.popleft()
        return None


class SocketBridge(GPUBridge):
    """
    TCP socket bridge stub for dual-GPU prod (Windows).

    Fully stubbed — will use asyncio or ZMQ in production.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9500):
        self._host = host
        self._port = port
        self._connected = False
        logger.info(f"SocketBridge stub initialized ({host}:{port})")

    def connect(self) -> bool:
        logger.info(f"SocketBridge.connect() — stub, would connect to {self._host}:{self._port}")
        return False

    def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def send(self, channel: str, data: Dict[str, Any]) -> bool:
        logger.debug(f"SocketBridge.send({channel}) — stub")
        return False

    def receive(self, channel: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"SocketBridge.receive({channel}) — stub")
        return None
