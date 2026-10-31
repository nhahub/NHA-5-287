"""LLM client (OpenAI-compatible chat-completions API: Google Gemini or Groq) using only the stdlib.

Design rules
  * The key comes from GEMINI_API_KEY (or GROQ_API_KEY for Groq) and is never logged or put in errors.
  * ask_json never raises: it returns (data, error). The agents' ask() helper turns any error or unusable
    output into LLMRequired: there is NO rule-based fallback, so a bad key or rate limit stops the run loudly.
  * The LLM only ever sees aggregated metadata/evidence (no raw rows; example values only if
    NEXUSML_SEND_SAMPLES=1). Numbers in outputs must come from the evidence we pass in.
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from . import config

class LLMRequired(RuntimeError):
    """Raised when an LLM decision is needed and cannot be obtained. There is NO rule-based fallback:
    the run stops and tells the user exactly why."""


CALLS: list = []   # in-process audit log (purpose, ok, seconds, tokens)


def available() -> bool:
    return bool(os.environ.get(config.LLM_KEY_ENV))


def _extract_json(text: str) -> Any:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # some reasoning models inline their thinking
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        return json.loads(t[i:j + 1])
    raise ValueError("no JSON object in model output")


def _post(body: Dict[str, Any], provider: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = config.LLM_BASE_URL if provider is None else provider["base_url"]
    key = os.environ[config.LLM_KEY_ENV] if provider is None else os.environ[provider["api_key_env"]]
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key,
                 "User-Agent": "nexusml/2.0"},   # default urllib UA is blocked by Groq's edge
        method="POST")
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _err_detail(e: urllib.error.HTTPError) -> str:
    try:
        msg = json.loads(e.read().decode("utf-8", "replace")).get("error", {}).get("message", "")
        return msg.replace("\n", " ")[:200]
    except Exception:
        return ""


_NON_CHAT = ("whisper", "guard", "tts", "orpheus", "transcri", "embed", "safeguard", "playai")


def list_models() -> list:
    """Chat-capable model ids your key can use right now (GET /models). Raises LLMRequired on failure."""
    if not available():
        raise LLMRequired(f"{config.LLM_KEY_ENV} is not set.")
    req = urllib.request.Request(config.LLM_BASE_URL.rstrip("/") + "/models", method="GET",
                                 headers={"Authorization": "Bearer " + os.environ[config.LLM_KEY_ENV], "User-Agent": "nexusml/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            ids = [str(m["id"]).removeprefix("models/") for m in json.loads(r.read().decode("utf-8")).get("data", []) if m.get("active", True)]
    except Exception as e:
        raise LLMRequired(f"Could not list {config.LLM_PROVIDER_NAME} models ({type(e).__name__}).")
    return sorted(i for i in ids if not any(x in i.lower() for x in _NON_CHAT))


_SEM: Dict[str, Any] = {"n": None, "sem": None}


def _gate():
    """Process-wide limit on simultaneous LLM calls (parallel branches must not trip rate limits)."""
    n = max(1, int(config.LLM_CONCURRENCY))
    if _SEM["sem"] is None or _SEM["n"] != n:
        _SEM["n"], _SEM["sem"] = n, threading.BoundedSemaphore(n)
    return _SEM["sem"]


def fallback_providers() -> list:
    """Extra OpenAI-compatible providers tried (in order) when the primary fails.
    NEXUSML_FALLBACK_PROVIDERS='[{"name":"x","base_url":"https://.../v1","api_key_env":"X_KEY","model":"m"}]'"""
    raw = os.environ.get("NEXUSML_FALLBACK_PROVIDERS", "").strip()
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        return []
    ok = []
    for p in items if isinstance(items, list) else []:
        if isinstance(p, dict) and all(k in p for k in ("base_url", "api_key_env", "model")) and os.environ.get(p["api_key_env"]):
            ok.append({"name": p.get("name", p["base_url"]), "base_url": p["base_url"], "api_key_env": p["api_key_env"], "model": p["model"]})
    return ok


def supports_images() -> bool:
    """Images are never sent to the LLM unless a vision-capable model is configured explicitly
    (NEXUSML_LLM_VISION=1). The default text model gets only aggregated image statistics."""
    return os.environ.get("NEXUSML_LLM_VISION", "0") == "1"


def _cache_path(key: str):
    from pathlib import Path
    d = Path(config.artifacts_dir()) / "llm_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _ask_one(provider: Optional[Dict[str, Any]], system: str, prompt: str, purpose: str, max_tokens: int,
             temperature: float) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    model = config.LLM_MODEL if provider is None else provider["model"]
    rec = {"purpose": purpose, "model": model, "ok": False, "seconds": 0.0, "tokens": 0}
    if provider is None and not available():
        return None, f"{config.LLM_KEY_ENV} is not set", rec
    if len(prompt) > config.LLM_PROMPT_CHARS:
        prompt = prompt[: config.LLM_PROMPT_CHARS] + "\n...[truncated]"
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system + "\nReply with a single valid JSON object only."},
                         {"role": "user", "content": prompt}]}
    if model.startswith("openai/gpt-oss"):          # reasoning model: keep thinking short, leave room for the answer
        body.update(reasoning_effort="low", include_reasoning=False)
        body["max_tokens"] = max_tokens + config.LLM_REASONING_HEADROOM
    t0, err = time.time(), None
    for attempt in range(config.LLM_MAX_RETRIES + 3):
        try:
            resp = _post(body) if provider is None else _post(body, provider)
            rec["tokens"] = int((resp.get("usage") or {}).get("total_tokens", 0))
            data = _extract_json(resp["choices"][0]["message"]["content"])
            if not isinstance(data, dict):
                raise ValueError("model returned JSON that is not an object")
            rec.update(ok=True, seconds=round(time.time() - t0, 2))
            CALLS.append(rec)
            return data, None, rec
        except urllib.error.HTTPError as e:
            detail = _err_detail(e)
            err = f"HTTP {e.code}" + (f" ({detail})" if detail else "")
            if e.code in (429, 500, 502, 503):
                try:
                    wait = float(e.headers.get("retry-after", 2 ** attempt * 2)) if e.headers else 2 ** attempt * 2
                except (TypeError, ValueError):
                    wait = 2 ** attempt * 2
                time.sleep(min(wait, 60))
                continue
            if e.code == 400:                                  # unsupported optional parameter for this model: drop one, retry
                dropped = next((k for k in ("include_reasoning", "reasoning_effort", "response_format") if k in body), None)
                if dropped:
                    body.pop(dropped)
                    continue
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err = f"network error ({type(e).__name__})"
            time.sleep(2 ** attempt)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            err = f"unusable model output ({e})"
            break
    rec["seconds"] = round(time.time() - t0, 2)
    CALLS.append(rec)
    return None, err or "unknown error", rec


def ask_json(system: str, prompt: str, purpose: str = "", max_tokens: int = 1600,
             temperature: float = 0.1) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """Return (parsed_json | None, error | None, call_record).

    Primary provider is Groq (unchanged). If it fails and NEXUSML_FALLBACK_PROVIDERS lists others, they are tried in order.
    Concurrency is capped (NEXUSML_LLM_CONCURRENCY); identical calls can be cached on disk (NEXUSML_LLM_CACHE=1)."""
    ckey = hashlib.sha256(json.dumps([config.LLM_MODEL, system, prompt, max_tokens, temperature]).encode()).hexdigest()[:24]
    if config.LLM_CACHE:
        cp = _cache_path(ckey)
        if cp.is_file():
            try:
                hit = json.loads(cp.read_text(encoding="utf-8"))
                return hit, None, {"purpose": purpose, "model": "cache", "ok": True, "seconds": 0.0, "tokens": 0, "cached": True}
            except ValueError:
                pass
    with _gate():
        data, err, rec = _ask_one(None, system, prompt, purpose, max_tokens, temperature)
        if err:
            for p in fallback_providers():
                d2, e2, r2 = _ask_one(p, system, prompt, purpose, max_tokens, temperature)
                if not e2:
                    r2["fallback_from"] = config.LLM_MODEL
                    data, err, rec = d2, None, r2
                    break
    if config.LLM_CACHE and not err and data is not None:
        try:
            _cache_path(ckey).write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass
    return data, err, rec


DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"      # Groq; with Gemini the main model is multimodal and is used


def vision_model() -> str:
    return os.environ.get("NEXUSML_VISION_MODEL") or (config.LLM_MODEL if config.LLM_PROVIDER == "gemini" else DEFAULT_VISION_MODEL)


def ask_vision(system: str, prompt: str, image_bytes: bytes, mime: str = "image/jpeg", max_tokens: int = 700,
               model: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """One image + a question -> JSON. Uses the primary provider (Groq's OpenAI-compatible API) with a vision-capable model
    (default meta-llama/llama-4-scout-17b-16e-instruct, override with NEXUSML_VISION_MODEL). The IMAGE LEAVES THE MACHINE:
    callers must have the user's consent."""
    import base64
    model = model or vision_model()
    rec = {"purpose": "vision", "model": model, "ok": False, "seconds": 0.0, "tokens": 0}
    if not available():
        return None, f"{config.LLM_KEY_ENV} is not set", rec
    url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    body = {"model": model, "temperature": 0.1, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system + "\nReply with a single valid JSON object only."},
                         {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}]}
    t0, err = time.time(), None
    with _gate():
        for attempt in range(config.LLM_MAX_RETRIES + 2):
            try:
                resp = _post(body)
                rec["tokens"] = int((resp.get("usage") or {}).get("total_tokens", 0))
                data = _extract_json(resp["choices"][0]["message"]["content"])
                if not isinstance(data, dict):
                    raise ValueError("the model did not return a JSON object")
                rec.update(ok=True, seconds=round(time.time() - t0, 2))
                CALLS.append(rec)
                return data, None, rec
            except urllib.error.HTTPError as e:
                detail = _err_detail(e)
                if e.code in (429, 500, 502, 503):
                    err = f"HTTP {e.code}"
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                if e.code == 400 and "response_format" in body and "image" not in (detail or "").lower() and "model" not in (detail or "").lower():
                    body.pop("response_format")
                    continue
                if e.code in (400, 404) and "model" in (detail or "").lower():
                    err = (f"The vision model '{model}' is not available to your {config.LLM_PROVIDER_NAME} account ({detail[:120]}). Set NEXUSML_VISION_MODEL to a "
                           "vision-capable model listed at console.groq.com/docs/vision.")
                else:
                    err = f"HTTP {e.code}" + (f" ({detail[:160]})" if detail else "")
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                err = f"network error ({type(e).__name__})"
                time.sleep(2 ** attempt)
            except (ValueError, KeyError) as e:
                err = f"unusable model output ({e})"
                break
    rec["seconds"] = round(time.time() - t0, 2)
    CALLS.append(rec)
    return None, err or "unknown error", rec


def preflight() -> None:
    """Fail fast (before any heavy work) if the key is missing or Groq rejects the request."""
    if not available():
        raise LLMRequired(f"{config.LLM_KEY_ENV} is not set. Put it in your environment or in a .env file ({config.LLM_KEY_ENV}=...).")
    data, err, _ = ask_json("Reply with the JSON object {\"ok\": true}.", "{\"ping\": true}", purpose="preflight", max_tokens=20)
    if err:
        code = err.split(" ")[1].rstrip(":") if err.startswith("HTTP") else ""
        if code == "404":
            try:
                live = ", ".join(list_models()[:12]) or "(none returned)"
            except LLMRequired:
                live = "(could not list models)"
            raise LLMRequired(f"{config.LLM_PROVIDER_NAME} preflight failed ({err}): model '{config.LLM_MODEL}' is not available to your account. "
                              f"Models you can use right now: {live}. Set NEXUSML_LLM_MODEL (in .env) to one of them.")
        hint = {"401": "the key was rejected: create a new key in " + ("the Google AI Studio console" if config.LLM_PROVIDER == "gemini" else "the Groq console"),
                "403": "access denied for this key/organization",
                "429": "rate limited: wait a minute or use a model with a higher limit"}.get(code, "check connectivity and NEXUSML_LLM_MODEL")
        raise LLMRequired(f"{config.LLM_PROVIDER_NAME} preflight failed ({err}): {hint}.")
