#!/usr/bin/env python3
"""Registry of indexed folders, stored at ~/.llmdex/registry.json."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from llm_index.indexer import storage_dir

REGISTRY_FILE = Path.home() / ".llmdex" / "registry.json"


def _load() -> dict[str, dict]:
    """Load registry. Format: {"/abs/path": {"extensions": [".md", ".ts"], ...}}"""
    if not REGISTRY_FILE.exists():
        return {}
    try:
        return json.loads(REGISTRY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict[str, dict]):
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(data, indent=2))


def register(directory: str, extensions: list[str]):
    """Add or update a folder in the registry."""
    data = _load()
    data[directory] = {
        "extensions": extensions,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(data)


def unregister(directory: str) -> bool:
    """Remove a folder from the registry and delete its index data. Returns True if found."""
    data = _load()
    if directory not in data:
        return False

    # Delete index storage
    store = storage_dir(Path(directory))
    if store.exists():
        shutil.rmtree(store)

    del data[directory]
    _save(data)
    return True


def list_registered() -> dict[str, dict]:
    """Return all registered folders."""
    return _load()


def get_entry(directory: str) -> dict | None:
    """Get registry entry for a folder."""
    return _load().get(directory)
