from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Filesystem Server")


def _notes_base() -> Path:
    return Path(os.getenv("NOTES_PATH", "study_materials/sample_notes")).resolve()


def _resolve_safe(filename: str) -> Path:
    base = _notes_base()
    candidate = (base / filename).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f"'{filename}' is outside the notes directory")
    return candidate


@mcp.tool()
def list_study_files() -> list[str]:
    """List available study note filenames, sorted alphabetically."""
    base = _notes_base()
    if not base.exists():
        return []
    return sorted(p.name for p in base.glob("*.md"))


@mcp.tool()
def read_study_file(filename: str) -> str:
    """Read the contents of a study note by filename."""
    path = _resolve_safe(filename)
    if not path.exists():
        return f"File not found: {filename}"
    return path.read_text(encoding="utf-8")


@mcp.tool()
def search_notes(query: str) -> list[str]:
    """Case-insensitive substring search across study notes.

    Returns up to 20 matches formatted as 'filename: line'.
    """
    base = _notes_base()
    results: list[str] = []
    if not base.exists():
        return results
    needle = query.lower()
    for path in sorted(base.glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if needle in line.lower():
                results.append(f"{path.name}: {line.strip()}")
                if len(results) >= 20:
                    return results
    return results


@mcp.resource("notes://index")
def notes_index() -> str:
    """Markdown index of available study materials with file sizes."""
    base = _notes_base()
    if not base.exists():
        return "# Study Materials\n\n(no notes directory found)"
    lines = ["# Study Materials", ""]
    for path in sorted(base.glob("*.md")):
        lines.append(f"- {path.name} ({path.stat().st_size} bytes)")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
