"""Utility helpers for basic filesystem and tabular IO operations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) when it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a ``DataFrame`` with helpful error reporting."""

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Persist a DataFrame to disk and ensure the parent directory exists."""

    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def optional_path(value: Optional[str]) -> Optional[Path]:
    """Convert a string to ``Path`` if provided."""

    return Path(value).expanduser().resolve() if value else None
