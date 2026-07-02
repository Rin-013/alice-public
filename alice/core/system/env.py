"""Environment bootstrap. Loads .env from repo root into os.environ.

Call once, very early — before any module reads os.environ for secrets
(Twitch tokens, API keys, etc.). Idempotent and ~1ms.
"""
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _REPO_ROOT / ".env"


def load_env() -> bool:
    """Load .env into os.environ. Returns True if a file was found and loaded."""
    return load_dotenv(_ENV_PATH)
