"""Claim validation: numbers and wording the LLM writes must be backed by the evidence register.

* numeric check: every figure in an LLM sentence must match a figure in the evidence (within rounding), or be a simple
  difference/ratio of two figures from the SAME cited evidence items;
* wording check: ROI / savings / certainty language is rejected unless the user supplied cost data.
Used as a two-stage gate: the first failure gives the LLM one corrective retry with the exact problems listed,
the second only annotates the claim as "unverified" (the run is never killed by this check).
"""
import re
from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence, Tuple

NUM = re.compile(r"(?<![\w.])(\$|€|£)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|x\b|×|-fold|\s?fold\b|pp\b| percentage points?)?", re.I)
BANNED_ROI = re.compile(r"\b(roi|return on investment|payback|net savings|cost savings of)\b", re.I)
BANNED_CERTAIN = re.compile(r"\b(guarantee[sd]?|definitely will|will certainly|proves? that|proven to)\b", re.I)


def _numbers(text: str) -> List[Dict[str, Any]]:
    out = []
    for m in NUM.finditer(text or ""):
        whole, frac, unit = m.group(2).replace(",", ""), m.group(3) or "", (m.group(4) or "").strip().lower()
        val = float(whole + frac)
        dec = len(frac) - 1 if frac else 0
        if not unit and not m.group(1) and not frac and (val <= 12 or 1900 <= val <= 2100):
            continue                                           # item counts, years: not figures to verify
        out.append({"raw": m.group(0).strip(), "val": val, "decimals": dec, "unit": unit})
    return out


def _flatten(obj: Any, acc: List[float], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, acc, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in list(obj)[:400]:
            _flatten(v, acc, depth + 1)


def _evidence_numbers(ev: Dict[str, Any]) -> List[float]:
    acc: List[float] = [n["val"] for n in _numbers(ev.get("text", ""))]
    for m in NUM.finditer(ev.get("text", "")):                 # small integers inside evidence text also count as evidence
        try:
            acc.append(float(m.group(2).replace(",", "") + (m.group(3) or "")))
        except ValueError:
            pass
    _flatten(ev.get("data"), acc)
    return acc


def _variants(x: float) -> List[float]:
    v = [x, x * 100.0, x / 100.0]
    return v


def _match(c: Dict[str, Any], pool: Iterable[float]) -> bool:
    tol = max(0.5 * 10 ** (-c["decimals"]), 0.0)
    for e in pool:
        for ev in _variants(e):
            if abs(c["val"] - ev) <= max(tol, 0.02 * abs(ev)) + 1e-9:
                return True
    return False


def check_text(text: str, cited: Sequence[Dict[str, Any]], all_ev: Sequence[Dict[str, Any]], allow_roi: bool) -> List[str]:
    """Return a list of problems found in one piece of LLM text."""
    problems: List[str] = []
    cited_nums: List[float] = []
    for e in cited:
        cited_nums.extend(_evidence_numbers(e))
    all_nums: List[float] = []
    for e in all_ev:
        all_nums.extend(_evidence_numbers(e))
    diffs: List[float] = []
    ratios: List[float] = []
    uniq = sorted(set(round(x, 6) for x in cited_nums))[:40]
    for a, b in combinations(uniq, 2):
        diffs.append(abs(a - b))                         # "a 20-point gap" is legitimate arithmetic on cited figures
        if b:
            ratios.append(a / b)
        if a:
            ratios.append(b / a)
    for c in _numbers(text):
        multiplier = c["unit"] in ("x", "×", "-fold", "fold")      # ratios only justify "Nx"/"N-fold" claims, never AUCs or rates
        if _match(c, all_nums) or _match(c, diffs) or (multiplier and _match(c, ratios)):
            continue
        problems.append(f"figure '{c['raw']}' does not appear in the evidence")
    if not allow_roi and BANNED_ROI.search(text or ""):
        problems.append("ROI/payback language is not allowed because no cost data was provided")
    if BANNED_CERTAIN.search(text or ""):
        problems.append("certainty language (guarantee/prove) is not allowed for observational evidence")
    return problems


class Gate:
    """Two-stage gate. check() raises ValueError on the first failing call (so ask() retries once with the details),
    afterwards it only reports problems so the caller can annotate claims as unverified."""

    def __init__(self):
        self.calls = 0

    def check(self, items: Sequence[Tuple[str, List[str], str]], evidence: Sequence[Dict[str, Any]],
              allow_roi: bool) -> Dict[str, List[str]]:
        """items = [(key, evidence_ids, text)] -> {key: [problems]} (only keys with problems)."""
        by_id = {e["id"]: e for e in evidence}
        bad: Dict[str, List[str]] = {}
        for key, ids, text in items:
            p = check_text(text, [by_id[i] for i in ids if i in by_id], evidence, allow_roi)
            if p:
                bad[key] = p
        self.calls += 1
        if bad and self.calls == 1:
            detail = "; ".join(f"{k}: {', '.join(v[:2])}" for k, v in list(bad.items())[:4])
            raise ValueError("unsupported claims -> " + detail + ". Use only figures present in the evidence and avoid ROI/certainty language.")
        return bad
