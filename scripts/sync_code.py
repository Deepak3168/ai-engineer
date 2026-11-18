#!/usr/bin/env python3
"""Keep code blocks inside lesson Markdown identical to the real source files.

Why: every code listing in a lesson is a real file that is linted, tested and
executed in CI. The Markdown never holds a hand-copied variant that can drift.

Authoring syntax (inside any .md file):

    <!-- code:begin path="code/day01/demo.py" -->
    <!-- code:end -->

Running this script fills (or refreshes) everything between the markers with a
caption plus a fenced code block read from `path` (relative to the repo root).

Optional attribute:  lines="10-40"  -> embed only that 1-based, inclusive range.

Usage:
    python scripts/sync_code.py                          # rewrite all lesson files in place
    python scripts/sync_code.py --check                  # exit 1 if any block is stale/missing
    python scripts/sync_code.py lessons/day-05-*.md      # only these files (safe for parallel authors)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEARCH_DIRS = ("lessons", "projects", "resources")

BLOCK_RE = re.compile(
    r'(?P<begin><!-- code:begin (?P<attrs>[^>]*?)-->)\n'
    r'(?P<body>.*?)'
    r'(?P<end><!-- code:end -->)',
    re.DOTALL,
)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

LANG_BY_SUFFIX = {
    ".py": "python",
    ".sql": "sql",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sh": "bash",
    ".md": "markdown",
    ".txt": "text",
    ".ini": "ini",
    ".env": "bash",
}


def render(attrs: dict[str, str]) -> str:
    rel = attrs["path"]
    src = ROOT / rel
    if not src.is_file():
        raise FileNotFoundError(f"code:begin path not found: {rel}")
    text = src.read_text(encoding="utf-8").rstrip("\n")
    lines = text.split("\n")
    suffix_note = ""
    if "lines" in attrs:
        start_s, _, end_s = attrs["lines"].partition("-")
        start, end = int(start_s), int(end_s or start_s)
        if not (1 <= start <= end <= len(lines)):
            raise ValueError(f"{rel}: lines={attrs['lines']} outside 1..{len(lines)}")
        lines = lines[start - 1 : end]
        suffix_note = f" (lines {start}-{end})"
    lang = LANG_BY_SUFFIX.get(src.suffix, "")
    fence = "````" if any(ln.startswith("```") for ln in lines) else "```"
    body = "\n".join(lines)
    return f"**`{rel}`**{suffix_note}\n\n{fence}{lang}\n{body}\n{fence}\n"


def sync_text(text: str, md_path: Path) -> str:
    def repl(m: re.Match[str]) -> str:
        attrs = dict(ATTR_RE.findall(m.group("attrs")))
        if "path" not in attrs:
            raise ValueError(f'{md_path}: code:begin without path="..."')
        return f'{m.group("begin")}\n{render(attrs)}{m.group("end")}'

    return BLOCK_RE.sub(repl, text)


def markdown_files() -> list[Path]:
    files: list[Path] = [ROOT / "README.md"]
    for d in SEARCH_DIRS:
        files += sorted((ROOT / d).rglob("*.md"))
    return [f for f in files if f.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="do not write; fail if stale")
    ap.add_argument("files", nargs="*", help="restrict to these Markdown files (default: all)")
    args = ap.parse_args()

    stale: list[Path] = []
    errors: list[str] = []
    blocks = 0
    targets = [Path(f).resolve() for f in args.files] if args.files else markdown_files()
    for md in targets:
        original = md.read_text(encoding="utf-8")
        blocks += len(BLOCK_RE.findall(original))
        try:
            updated = sync_text(original, md)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{md.relative_to(ROOT)}: {exc}")
            continue
        if updated != original:
            stale.append(md)
            if not args.check:
                md.write_text(updated, encoding="utf-8")

    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    verb = "stale" if args.check else "updated"
    for md in stale:
        print(f"{verb}: {md.relative_to(ROOT)}")
    print(f"{blocks} code blocks scanned, {len(stale)} {verb}, {len(errors)} errors")
    if errors or (args.check and stale):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
