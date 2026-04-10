#!/usr/bin/env python3
"""AST-aware code chunking using tree-sitter.

Splits code files at function/class/method boundaries instead of arbitrary
line counts. Each chunk is a semantic unit (one function, one class, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import get_parser

# Top-level node types to extract as individual chunks, per language.
# Anything not listed here gets grouped into a "preamble" chunk.
CHUNK_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "export_statement",  # often wraps a function/class
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
    },
    "rust": {
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
        "mod_item",
        "macro_definition",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "method_declaration",
        "record_declaration",
    },
}

# Node types that contain nested methods we should extract individually.
# The class body itself becomes a "shell" chunk (signature + field declarations).
CONTAINER_TYPES: dict[str, set[str]] = {
    "python": {"class_definition", "decorated_definition"},
    "typescript": {"class_declaration"},
    "javascript": {"class_declaration"},
    "rust": {"impl_item", "trait_item"},
    "go": set(),
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
}

# Node types that represent methods inside a container.
METHOD_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "decorated_definition"},
    "typescript": {"method_definition", "public_field_definition"},
    "javascript": {"method_definition"},
    "rust": {"function_item"},
    "go": set(),
    "java": {"method_declaration", "constructor_declaration"},
}

# Max lines for a single chunk before we fall back to splitting
MAX_CHUNK_LINES = 150


@dataclass
class Chunk:
    text: str
    start_line: int  # 1-based
    end_line: int  # 1-based
    symbol: str  # e.g. "class Foo", "def bar", or "" for preamble


def _node_name(node, language: str) -> str:
    """Extract a human-readable name from an AST node."""
    # For decorated definitions, dig into the actual definition
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _node_name(child, language)

    # For export statements, dig into the declaration
    if node.type == "export_statement":
        for child in node.children:
            if child.type in CHUNK_TYPES.get(language, set()):
                return _node_name(child, language)
            # export default function foo / export const foo
            if child.type in (
                "function_declaration",
                "class_declaration",
                "interface_declaration",
                "type_alias_declaration",
                "lexical_declaration",
            ):
                return _node_name(child, language)
        # Bare export (re-export) - use first line
        first_line = node.text.decode("utf-8", errors="replace").split("\n")[0]
        return first_line[:80]

    # Find the identifier child
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier"):
            prefix = node.type.replace("_definition", "").replace("_declaration", "").replace("_item", "")
            return f"{prefix} {child.text.decode('utf-8', errors='replace')}"

    # Fallback: first line trimmed
    first_line = node.text.decode("utf-8", errors="replace").split("\n")[0]
    return first_line[:80]


def _get_class_body_node(node, language: str):
    """Find the body/block child of a class/impl node."""
    body_types = {
        "python": "block",
        "typescript": "class_body",
        "javascript": "class_body",
        "rust": "declaration_list",
        "java": "class_body",
    }
    target = body_types.get(language)
    if not target:
        return None
    for child in node.children:
        if child.type == target:
            return child
    return None


def _split_container(node, source_bytes: bytes, language: str) -> list[Chunk]:
    """Split a class/impl into: shell chunk + individual method chunks."""
    method_types = METHOD_TYPES.get(language, set())
    body = _get_class_body_node(node, language)
    if not body or not method_types:
        # Can't split - return as single chunk
        text = node.text.decode("utf-8", errors="replace")
        return [Chunk(text, node.start_point.row + 1, node.end_point.row + 1, _node_name(node, language))]

    chunks = []
    container_name = _node_name(node, language)

    # Shell = everything except method bodies (signature, fields, etc.)
    # We build it by collecting non-method lines from the node
    methods = [child for child in body.children if child.type in method_types]

    if not methods:
        text = node.text.decode("utf-8", errors="replace")
        return [Chunk(text, node.start_point.row + 1, node.end_point.row + 1, container_name)]

    # Build shell: lines from container start to first method, plus inter-method gaps
    lines = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace").split("\n")
    node_start = node.start_point.row
    shell_lines = []

    # Lines before first method (class signature, fields)
    first_method_row = methods[0].start_point.row
    for i in range(node_start, first_method_row):
        shell_lines.append(lines[i - node_start])

    # Lines between methods and after last method (field declarations, etc.)
    for idx, method in enumerate(methods):
        method_end = method.end_point.row
        next_start = methods[idx + 1].start_point.row if idx + 1 < len(methods) else node.end_point.row + 1
        for i in range(method_end + 1, min(next_start, node.end_point.row + 1)):
            line = lines[i - node_start]
            if line.strip():  # skip blank lines in shell
                shell_lines.append(line)

    # Add closing brace if present
    last_line = lines[-1]
    if last_line.strip() in ("}", "end"):
        shell_lines.append(last_line)

    if shell_lines:
        chunks.append(Chunk(
            "\n".join(shell_lines),
            node.start_point.row + 1,
            node.end_point.row + 1,
            container_name,
        ))

    # Individual methods with class context prefix
    for method in methods:
        method_text = method.text.decode("utf-8", errors="replace")
        method_name = _node_name(method, language)
        symbol = f"{container_name}.{method_name}"
        chunks.append(Chunk(
            method_text,
            method.start_point.row + 1,
            method.end_point.row + 1,
            symbol,
        ))

    return chunks


def chunk_file(file_path: str, language: str) -> list[Chunk]:
    """Parse a file with tree-sitter and return semantic chunks."""
    try:
        with open(file_path, "rb") as f:
            source = f.read()
    except OSError:
        return []

    parser = get_parser(language)
    tree = parser.parse(source)
    root = tree.root_node

    chunk_types = CHUNK_TYPES.get(language, set())
    container_types = CONTAINER_TYPES.get(language, set())

    chunks: list[Chunk] = []
    preamble_lines: list[str] = []
    preamble_start: int | None = None
    last_chunk_end_row = -1

    for child in root.children:
        child_start = child.start_point.row
        child_end = child.end_point.row

        if child.type in chunk_types:
            # Flush any accumulated preamble
            if preamble_lines:
                text = "\n".join(preamble_lines)
                if text.strip():
                    chunks.append(Chunk(text, preamble_start + 1, child_start, "preamble"))
                preamble_lines = []
                preamble_start = None

            node_lines = child_end - child_start + 1

            # Split large containers into shell + methods
            if child.type in container_types and node_lines > MAX_CHUNK_LINES:
                chunks.extend(_split_container(child, source, language))
            else:
                text = child.text.decode("utf-8", errors="replace")
                chunks.append(Chunk(text, child_start + 1, child_end + 1, _node_name(child, language)))

            last_chunk_end_row = child_end
        else:
            # Accumulate into preamble (imports, constants, comments)
            if preamble_start is None:
                preamble_start = child_start
            text = child.text.decode("utf-8", errors="replace")
            preamble_lines.append(text)

    # Flush remaining preamble
    if preamble_lines:
        text = "\n".join(preamble_lines)
        if text.strip():
            start = preamble_start + 1 if preamble_start is not None else 1
            chunks.append(Chunk(text, start, root.end_point.row + 1, "preamble"))

    return chunks
