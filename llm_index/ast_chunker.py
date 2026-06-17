#!/usr/bin/env python3
"""AST-aware code chunking using tree-sitter.

Splits code files at function/class/method boundaries instead of arbitrary
line counts. Each chunk is a semantic unit (one function, one class, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from functools import lru_cache

from tree_sitter import Parser
from tree_sitter_language_pack import get_language


@lru_cache(maxsize=None)
def _parser(language: str) -> Parser:
    # Build via the modern tree_sitter binding. get_parser() can return a stale
    # Parser whose parse() expects str instead of bytes.
    return Parser(get_language(language))

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
    "csharp": {
        "class_declaration",
        "interface_declaration",
        "struct_declaration",
        "enum_declaration",
        "record_declaration",
        "delegate_declaration",
        "method_declaration",
        "constructor_declaration",
        "property_declaration",
    },
    "dart": {
        "class_definition",
        "mixin_declaration",
        "extension_declaration",
        "enum_declaration",
        "function_signature",  # top-level function; body is a sibling node
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
    "csharp": {"class_declaration", "struct_declaration", "interface_declaration", "record_declaration"},
    "dart": {"class_definition", "mixin_declaration", "extension_declaration"},
}

# Node types that represent methods inside a container.
METHOD_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "decorated_definition"},
    "typescript": {"method_definition", "public_field_definition"},
    "javascript": {"method_definition"},
    "rust": {"function_item"},
    "go": set(),
    "java": {"method_declaration", "constructor_declaration"},
    "csharp": {"method_declaration", "constructor_declaration", "property_declaration"},
    "dart": {"method_signature"},  # each followed by a sibling function_body
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

    # Dart wraps the name one level down (method_signature -> function_signature ...).
    if language == "dart" and node.type in ("method_signature", "declaration"):
        for child in node.children:
            if child.type.endswith("_signature"):
                return _node_name(child, language)

    # C# and Dart nodes expose the name via the "name" field.
    if language in ("csharp", "dart"):
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            prefix = (
                node.type.replace("_definition", "")
                .replace("_declaration", "")
                .replace("_signature", "")
            )
            return f"{prefix} {name_node.text.decode('utf-8', errors='replace')}".strip()

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
        "csharp": "declaration_list",
        "dart": "class_body",
    }
    target = body_types.get(language)
    if not target:
        return None
    for child in node.children:
        if child.type == target:
            return child
    return None


def _method_units(body, language: str) -> list[tuple]:
    """Return (signature_node, end_node) pairs for methods in a container body.

    For most languages end_node is the method node itself. In Dart the method
    signature and its body are separate sibling nodes, so end_node is the
    trailing function_body."""
    method_types = METHOD_TYPES.get(language, set())
    kids = list(body.children)
    units: list[tuple] = []
    i = 0
    while i < len(kids):
        c = kids[i]
        if c.type in method_types:
            end = c
            if language == "dart" and i + 1 < len(kids) and kids[i + 1].type == "function_body":
                end = kids[i + 1]
                i += 1
            units.append((c, end))
        i += 1
    return units


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
    methods = _method_units(body, language)

    if not methods:
        text = node.text.decode("utf-8", errors="replace")
        return [Chunk(text, node.start_point.row + 1, node.end_point.row + 1, container_name)]

    # Build shell: lines from container start to first method, plus inter-method gaps
    lines = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace").split("\n")
    node_start = node.start_point.row
    shell_lines = []

    # Lines before first method (class signature, fields)
    first_method_row = methods[0][0].start_point.row
    for i in range(node_start, first_method_row):
        shell_lines.append(lines[i - node_start])

    # Lines between methods and after last method (field declarations, etc.)
    for idx, (sig, end) in enumerate(methods):
        method_end = end.end_point.row
        next_start = methods[idx + 1][0].start_point.row if idx + 1 < len(methods) else node.end_point.row + 1
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
    for sig, end in methods:
        method_text = source_bytes[sig.start_byte : end.end_byte].decode("utf-8", errors="replace")
        method_name = _node_name(sig, language)
        symbol = f"{container_name}.{method_name}"
        chunks.append(Chunk(
            method_text,
            sig.start_point.row + 1,
            end.end_point.row + 1,
            symbol,
        ))

    return chunks


def _expand_csharp_namespace(node) -> list:
    """C# block-scoped namespaces nest declarations in a declaration_list.
    Flatten them so the inner types are chunked as if top-level."""
    if node.type == "namespace_declaration":
        body = next((c for c in node.children if c.type == "declaration_list"), None)
        if body is not None:
            out = []
            for c in body.children:
                out.extend(_expand_csharp_namespace(c))
            return out
    return [node]


def _top_level_nodes(root, language: str) -> list:
    """Top-level nodes to scan, with language-specific flattening."""
    if language == "csharp":
        out = []
        for c in root.children:
            out.extend(_expand_csharp_namespace(c))
        return out
    return list(root.children)


def chunk_file(file_path: str, language: str) -> list[Chunk]:
    """Parse a file with tree-sitter and return semantic chunks."""
    try:
        with open(file_path, "rb") as f:
            source = f.read()
    except OSError:
        return []

    tree = _parser(language).parse(source)
    root = tree.root_node

    chunk_types = CHUNK_TYPES.get(language, set())
    container_types = CONTAINER_TYPES.get(language, set())

    nodes = _top_level_nodes(root, language)
    chunks: list[Chunk] = []
    preamble_lines: list[str] = []
    preamble_start: int | None = None
    i = 0
    while i < len(nodes):
        child = nodes[i]
        # Dart splits a function into signature + body sibling nodes; pair them.
        end_node = child
        if language == "dart" and child.type in chunk_types and i + 1 < len(nodes) and nodes[i + 1].type == "function_body":
            end_node = nodes[i + 1]
            i += 1
        child_start = child.start_point.row
        child_end = end_node.end_point.row

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
                text = source[child.start_byte : end_node.end_byte].decode("utf-8", errors="replace")
                chunks.append(Chunk(text, child_start + 1, child_end + 1, _node_name(child, language)))

        else:
            # Accumulate into preamble (imports, constants, comments)
            if preamble_start is None:
                preamble_start = child_start
            text = source[child.start_byte : end_node.end_byte].decode("utf-8", errors="replace")
            preamble_lines.append(text)
        i += 1

    # Flush remaining preamble
    if preamble_lines:
        text = "\n".join(preamble_lines)
        if text.strip():
            start = preamble_start + 1 if preamble_start is not None else 1
            chunks.append(Chunk(text, start, root.end_point.row + 1, "preamble"))

    return chunks
