"""
Config Loader
=============

Hierarchical YAML config: default.yaml → hardware/*.yaml → games/*.yaml
Later layers override earlier ones.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

CONFIG_DIR = Path(__file__).parent.parent / "config"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class GamingConfig:
    """
    Hierarchical gaming configuration.

    Load order:
        1. config/default.yaml           (base)
        2. config/hardware/<hw>.yaml      (hardware overrides)
        3. config/games/<game>.yaml       (game-specific overrides)
    """

    def __init__(
        self,
        hardware: Optional[str] = None,
        game: Optional[str] = None,
        overrides: Optional[Dict] = None,
    ):
        self._data: Dict[str, Any] = {}
        self._load(hardware, game, overrides)

    def _load(
        self,
        hardware: Optional[str],
        game: Optional[str],
        overrides: Optional[Dict],
    ):
        # 1. Base config
        self._data = self._load_yaml(CONFIG_DIR / "default.yaml")

        # 2. Hardware overlay
        if hardware:
            hw_path = CONFIG_DIR / "hardware" / f"{hardware}.yaml"
            if hw_path.exists():
                hw_data = self._load_yaml(hw_path)
                self._data = _deep_merge(self._data, hw_data)

        # 3. Game overlay
        if game:
            game_path = CONFIG_DIR / "games" / f"{game}.yaml"
            if game_path.exists():
                game_data = self._load_yaml(game_path)
                self._data = _deep_merge(self._data, game_data)

        # 4. Runtime overrides
        if overrides:
            self._data = _deep_merge(self._data, overrides)

    @staticmethod
    def _load_yaml(path: Path) -> Dict:
        if not path.exists():
            return {}
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for config loading: pip install pyyaml")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    # --- Access helpers ---

    def get(self, dotpath: str, default: Any = None) -> Any:
        """
        Get a config value by dot-separated path.

        Example:
            cfg.get("mixer.base_weight", 0.3)
        """
        keys = dotpath.split(".")
        node = self._data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def section(self, name: str) -> Dict:
        """Get a top-level config section as a dict."""
        val = self._data.get(name, {})
        return val if isinstance(val, dict) else {}

    @property
    def data(self) -> Dict:
        return self._data

    def __repr__(self) -> str:
        return f"<GamingConfig keys={list(self._data.keys())}>"
