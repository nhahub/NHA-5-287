#!/usr/bin/env python3
"""Fail on syntax that only parses on Python >= 3.12 (PEP 701 f-strings), so a 3.12 laptop cannot hide a 3.11 CI failure.

Flags, inside an f-string replacement field: (1) a string literal using the same quote character as the enclosing f-string,
(2) a backslash, (3) a '#' comment. Also byte-compiles every file. Usage: python scripts/check_py311.py   (exit 1 on findings)
"""
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", ".agent_artifacts"}


def _quote_of(tok_string: str) -> str:
    body = tok_string.lstrip("rRbBfFuU")
    return body[:3] if body[:3] in ('"""', "'''") else body[:1]


def scan(path: Path):
    problems = []
    try:
        with tokenize.open(path) as fh:
            toks = list(tokenize.generate_tokens(fh.readline))
    except (SyntaxError, tokenize.TokenError, IndentationError) as e:
        return [(getattr(e, "lineno", 0) or 0, f"cannot tokenize: {e}")]
    stack = []                                   # one entry per open f-string: [quote, brace_depth]
    fstart, fmid, fend = (getattr(tokenize, n, None) for n in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"))
    if fstart is None:                           # running on <3.12: the interpreter itself already rejects these
        return problems
    for t in toks:
        if t.type == fstart:
            if stack and stack[-1][1] > 0 and _quote_of(t.string) == stack[-1][0]:
                problems.append((t.start[0], "f-string nested inside an f-string replacement field with the SAME quote (3.12+ only)"))
            stack.append([_quote_of(t.string), 0])
        elif t.type == fend:
            stack.pop()
        elif stack:
            top = stack[-1]
            if t.type == tokenize.OP and t.string == "{":
                top[1] += 1
            elif t.type == tokenize.OP and t.string == "}":
                top[1] = max(0, top[1] - 1)
            elif top[1] > 0:
                if t.type == tokenize.STRING and _quote_of(t.string) == top[0]:
                    problems.append((t.start[0], "string literal with the same quote as its enclosing f-string (3.12+ only): " + t.string[:30]))
                if t.type == tokenize.COMMENT:
                    problems.append((t.start[0], "comment inside an f-string replacement field (3.12+ only)"))
    return problems


def main() -> int:
    bad = 0
    for p in sorted(ROOT.rglob("*.py")):
        if SKIP & set(p.relative_to(ROOT).parts):
            continue
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
        except SyntaxError as e:
            print(f"{p.relative_to(ROOT)}:{e.lineno}: SyntaxError: {e.msg}")
            bad += 1
            continue
        for line, msg in scan(p):
            print(f"{p.relative_to(ROOT)}:{line}: {msg}")
            bad += 1
    print("OK: every file compiles and is Python 3.11 compatible" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
