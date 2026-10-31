#!/usr/bin/env python3
"""Fail if a secret looks committed. Scans tracked files (git) or the working tree, skipping git-ignored .env files.

Usage: python scripts/check_secrets.py [--include-env]   (exit code 1 when something is found)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = [
    ("Groq key", re.compile(r"gsk_[A-Za-z0-9]{20,}")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{30,}|\bAQ\.[A-Za-z0-9_-]{30,}")),
    ("Tavily key", re.compile(r"tvly-[A-Za-z0-9]{10,}")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Generic secret assignment", re.compile(r"(?i)\b(api[_-]?key|secret|token|password|client_secret)\b\s*[:=]\s*['\"]?([A-Za-z0-9_\-/+=]{20,})")),
]
SKIP_DIRS = {".git", ".agent_artifacts", "__pycache__", ".venv", "venv", "node_modules"}
PLACEHOLDER = re.compile(r"(?i)(example|your[_-]?key|changeme|placeholder|xxxx|<.*>|\.\.\.|os\.environ|getenv)")


def tracked_files(include_env: bool):
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        files = [ROOT / f for f in out if f]
        if files:
            return files, True
    except Exception:
        pass
    files = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if f == ".env" and not include_env:
                continue                      # a local, git-ignored file: not "committed"
            files.append(Path(dp) / f)
    return files, False


def main() -> int:
    include_env = "--include-env" in sys.argv
    files, via_git = tracked_files(include_env)
    problems = []
    if via_git and any(f.name == ".env" for f in files):
        problems.append((".env", 0, "a .env file is tracked by git"))
    for f in files:
        if f.suffix in {".png", ".jpg", ".zip", ".pkl", ".joblib", ".pyc", ".gz"} or not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if PLACEHOLDER.search(line) and "gsk_" not in line:
                continue
            for name, rx in PATTERNS:
                if rx.search(line):
                    problems.append((str(f.relative_to(ROOT)), n, name))
    for p in problems:
        print(f"SECRET? {p[0]}:{p[1]}: {p[2]}")
    print("secret scan:", "FAILED" if problems else "clean")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
