from __future__ import annotations

import os
from pathlib import Path


def _unquote_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def load_env_file(name: str = "project") -> None:
    """Load key-value pairs from a dotenv file into ``os.environ``."""
    path = Path(f".env.{name}")
    if not path.exists():
        path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), _unquote_env_value(value))


def require_env(key: str) -> str:
    """Return an environment variable or raise if it is unset."""
    value = os.environ.get(key)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def env_str(key: str, default: str) -> str:
    """Return a string environment variable with a default."""
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    """Return an integer environment variable with a default."""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(key: str, default: float) -> float:
    """Return a float environment variable with a default."""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return float(raw)


def env_bool(key: str, default: bool) -> bool:
    """Return a boolean environment variable with a default."""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
