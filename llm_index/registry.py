#!/usr/bin/env python3
"""Registry of indexed folders, stored at ~/.llmdex/registry.json."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from llm_index.indexer import storage_dir, EMBED_MODEL_NAME

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
    existing_tags = data.get(directory, {}).get("tags", [])
    data[directory] = {
        "extensions": extensions,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "tags": existing_tags,
        "embed_model": EMBED_MODEL_NAME,
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


def set_tags(directory: str, tags: list[str]):
    """Set tags for a registered folder. Replaces existing tags."""
    data = _load()
    if directory not in data:
        return False
    data[directory]["tags"] = sorted(set(tags))
    _save(data)
    return True


def find_by_tags(tags: list[str]) -> dict[str, dict]:
    """Return registered folders that have ALL of the given tags."""
    data = _load()
    tag_set = set(tags)
    return {
        d: meta
        for d, meta in data.items()
        if tag_set.issubset(set(meta.get("tags", [])))
    }


def list_all_tags() -> dict[str, list[str]]:
    """Return {tag: [directory, ...]} mapping of all tags in use."""
    data = _load()
    result: dict[str, list[str]] = {}
    for directory, meta in data.items():
        for tag in meta.get("tags", []):
            result.setdefault(tag, []).append(directory)
    return result
