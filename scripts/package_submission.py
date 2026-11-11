"""Build a clean submission zip: source, docs, sample files, no secrets or local caches.

    python scripts/package_submission.py            # -> dist/AlPrime_RAG_Assistant_Submission.zip

Excludes `.env`, virtualenvs, caches, runtime data, Office lock files (`~$*`), and `dist/`.
Before writing, every packaged text file is checked against the secret values in `.env`;
if any value appears, nothing is written. Secret values are never printed.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "AlPrime_RAG_Assistant_Submission.zip"
EXCLUDED_DIRS = {".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                 "data", "uploads", ".fastembed_cache", "local_cache", "dist", ".git", ".idea", ".vscode"}
EXCLUDED_FILES = {".env", ".env.local", "Thumbs.db", ".DS_Store"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".html", ".css", ".js", ".example", ""}


def _included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in rel.parts[:-1]):
        return False
    name = rel.name
    return not (name in EXCLUDED_FILES or name.startswith("~$") or name.endswith((".pyc", ".log")))


PUBLIC_VALUES = {"alprime-dev-key"}  # documented local-demo default (app.config.DEFAULT_API_KEY)


def _secret_values() -> list[str]:
    env = ROOT / ".env"
    if not env.exists():
        return []
    values = []
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().upper(), value.strip().strip('"').strip("'")
        if "MODEL" in key or value in PUBLIC_VALUES:  # model IDs and the public dev default are not secrets
            continue
        if any(t in key for t in ("KEY", "SECRET", "TOKEN", "PASSWORD")) and len(value) >= 12:
            values.append(value)
    return values


def main() -> int:
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and _included(p))
    secrets = _secret_values()
    leaks = []
    for path in files:
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            leaks += [str(path.relative_to(ROOT)) for s in secrets if s in text]
    if leaks:
        print("ABORTED: a secret value from .env appears in:", *sorted(set(leaks)), sep="\n  ")
        return 1

    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, Path("AlPrime_RAG_Assistant") / path.relative_to(ROOT))
    print(f"Packaged {len(files)} files ({OUT.stat().st_size / 1024:.0f} KB) -> {OUT}")
    print(f"Secret check: {len(secrets)} key value(s) from .env scanned, none found in packaged files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
