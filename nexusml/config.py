"""Central thresholds and settings. Tune here instead of hunting through agent code.

Secrets (GROQ_API_KEY, POWERBI_*) are read from the environment by the modules that need
them. They are never placed in graph state, prompts, logs or reports.
"""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env reader (KEY=VALUE lines) so the key can live in a git-ignored file instead of the shell.
    Real environment variables always win. Looks in the current directory and the project root."""
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        f = base / ".env"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.split(" #")[0].strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
            break


_load_dotenv()

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MIN_ROWS = 30

# Data-engineer thresholds
ID_UNIQUE_RATIO = 0.95      # >=95% unique categorical/integer column => ID-like
HIGH_MISSING_PCT = 60.0     # drop columns missing >= 60%
LEAK_CORR = 0.98            # |corr| with target at/above this => leakage suspect
LEAK_PURITY = 0.999         # category fully determines class => leakage suspect
LEAK_SINGLE_FEATURE = 0.97  # a lone feature predicting the target this well => leakage suspect

# Critic thresholds
MIN_IMPROVEMENT = 0.02      # best CV score must beat baseline CV by this much
OVERFIT_GAP = 0.15          # selection score - held-out score
HIGH_CV_STD = 0.10
IMBALANCE_SHARE = 0.20
PERFECT_SCORE = 0.995
SKEW_THRESHOLD = 1.0
SMALL_TRAIN = 200
MIN_SILHOUETTE = 0.25       # clustering: below this the segments are weak

MAX_PERM_ROWS = 1500

# Hyper-parameter tuning
TUNE_N_ITER = int(os.environ.get("NEXUSML_TUNE_ITER", "12"))
TUNE_TOP_K = 2              # tune the best K default models
TUNE_INNER_CV = 3
TUNE_OUTER_CV = 3           # nested CV => unbiased estimate of the *tuned* model
TUNE_MAX_TRIALS_REPORTED = 10

# Statistical analysis
ALPHA = 0.05
MAX_TESTS = 14              # max tests the Statistician executes per run
CLT_N = 200                 # groups at least this big => parametric by CLT (no Shapiro needed)
SHAPIRO_MAX_N = 5000
MIN_GROUP_N = 5

# Visualisation
CHART_DPI = 110
MAX_HIST_COLS = 12
MAX_CORR_COLS = 18

# LLM: any OpenAI-compatible chat endpoint. API KEYS ARE NEVER STORED IN CODE OR IN THE APP: only the NAME of the environment
# variable that holds each key is listed here. Set the variable in your shell / hosting platform / a git-ignored .env file.
#   id -> (base_url, key_env_var, default_model, display_name)
PROVIDERS = {"groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b", "Groq"),
             "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-3.1-flash-lite", "Google AI Studio (Gemini)"),
             "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini", "OpenAI"),
             "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-4o-mini", "OpenRouter"),
             "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "Together AI")}


def _extra_providers() -> None:
    """Add providers without touching code: NEXUSML_EXTRA_PROVIDERS='[{"id":"x","name":"X","base_url":"https://.../v1","api_key_env":"X_API_KEY","model":"m"}]'.
    Only the variable NAME goes in the JSON, never the key itself."""
    import json
    raw = os.environ.get("NEXUSML_EXTRA_PROVIDERS", "").strip()
    if not raw:
        return
    try:
        for e in json.loads(raw):
            if isinstance(e, dict) and all(k in e for k in ("id", "base_url", "api_key_env", "model")) and str(e["api_key_env"]).replace("_", "").isalnum():
                PROVIDERS[str(e["id"]).lower()] = (str(e["base_url"]), str(e["api_key_env"]), str(e["model"]), str(e.get("name") or e["id"]))
    except (ValueError, TypeError):
        pass


_extra_providers()
_gemini_env = "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") or not os.environ.get("GOOGLE_API_KEY") else "GOOGLE_API_KEY"


def key_env_for(provider: str) -> str:
    """Name of the environment variable that holds this provider's key (Gemini also accepts GOOGLE_API_KEY)."""
    return _gemini_env if provider == "gemini" else PROVIDERS[provider][1]


def provider_configured(provider: str) -> bool:
    return provider in PROVIDERS and bool(os.environ.get(key_env_for(provider)))


def _default_provider() -> str:
    forced = os.environ.get("NEXUSML_LLM_PROVIDER", "").lower()
    if forced in PROVIDERS:
        return forced
    if os.environ.get(_gemini_env):                         # historical rule: Gemini wins when both Gemini and Groq keys exist
        return "gemini"
    return next((pid for pid in PROVIDERS if provider_configured(pid)), "groq")


def set_provider(provider: str, model: str = "") -> str:
    """Switch the active LLM provider at run time (Settings page, CLI). Returns the provider id. Never touches keys."""
    global LLM_PROVIDER, LLM_KEY_ENV, LLM_PROVIDER_NAME, LLM_MODEL, LLM_BASE_URL
    provider = (provider or "").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")
    LLM_PROVIDER, LLM_KEY_ENV, LLM_PROVIDER_NAME = provider, key_env_for(provider), PROVIDERS[provider][3]
    LLM_MODEL = model or os.environ.get("NEXUSML_LLM_MODEL") or PROVIDERS[provider][2]
    LLM_BASE_URL = os.environ.get("NEXUSML_LLM_BASE_URL") or (os.environ.get("GROQ_BASE_URL") if provider == "groq" else None) or PROVIDERS[provider][0]
    return provider


def provider_status() -> list:
    """For the Settings page: one row per provider. Reports whether a key is PRESENT in the environment (never its value)."""
    return [{"id": pid, "name": v[3], "key_env": key_env_for(pid), "configured": provider_configured(pid), "default_model": v[2],
             "active": pid == LLM_PROVIDER} for pid, v in PROVIDERS.items()]


LLM_PROVIDER = LLM_KEY_ENV = LLM_PROVIDER_NAME = LLM_MODEL = LLM_BASE_URL = ""
set_provider(_default_provider())
LLM_TIMEOUT = 120
LLM_REASONING_HEADROOM = 2500   # extra output tokens for reasoning models (thinking tokens count against max_tokens)
LLM_MAX_RETRIES = 3
LLM_PROMPT_CHARS = 22000    # hard cap on prompt size (keeps within free-tier token limits)
LLM_SEND_SAMPLES = os.environ.get("NEXUSML_SEND_SAMPLES", "0") == "1"  # raw example values to the LLM


def artifacts_dir() -> str:
    return os.environ.get("NEXUSML_ARTIFACTS", ".agent_artifacts")


# v4 settings
# reproducibility / audit
PARALLEL = os.environ.get("NEXUSML_PARALLEL", "1") == "1"        # run independent graph branches concurrently
LLM_CONCURRENCY = int(os.environ.get("NEXUSML_LLM_CONCURRENCY", "2"))   # simultaneous LLM calls (rate-limit friendly)
N_JOBS = int(os.environ.get("NEXUSML_N_JOBS", "1"))              # processes used inside hyper-parameter search
LLM_CACHE = os.environ.get("NEXUSML_LLM_CACHE", "0") == "1"       # cache identical LLM calls on disk
LLM_PRICE_PER_MTOK = float(os.environ.get("NEXUSML_PRICE_PER_MTOK", "0"))   # USD per million tokens (0 = unknown)
SELF_CONSISTENCY_N = int(os.environ.get("NEXUSML_SELF_CONSISTENCY", "1"))   # >1: sample several plans, keep the majority

# uncertainty / honesty
BOOTSTRAP_N = 500
CI_LEVEL = 0.95
SEED_STABILITY_SEEDS = 3
MIN_TEST_POSITIVES_WARN = 60      # fewer test positives than this => warn that the score is noisy

# business cost / thresholds
DEFAULT_COST_FN = 5.0             # relative cost of missing a positive (used ONLY when the user gives no costs; labelled as default)
DEFAULT_COST_FP = 1.0

# analysis depth
INTERACTION_MIN_SUPPORT = 30
INTERACTION_MIN_LIFT = 1.4
INTERACTION_MAX_ITEMS = 6
DRIVER_GROUP_CORR = 0.6           # |r| above this between top drivers => same driver group
SENSITIVE_PATTERNS = ("gender", "sex", "age", "race", "ethnic", "religion", "marital", "nationality", "disab", "pregnan",
                      "orientation", "citizen", "birth", "veteran")

# agent collaboration
MAX_AGENT_REQUESTS = 4            # cap on agent-to-agent requests per run

# class imbalance / decision threshold (see imbalance.py, threshold.py)
DEFAULT_IMBALANCE = os.environ.get("NEXUSML_IMBALANCE", "").lower()      # "", "smote", "adasyn", "random": force a resampler from the first attempt
TUNE_THRESHOLD = os.environ.get("NEXUSML_TUNE_THRESHOLD", "1") == "1"   # binary classification: choose the decision threshold out-of-fold
THRESHOLD_OBJECTIVE = os.environ.get("NEXUSML_THRESHOLD_OBJECTIVE", "f1")

# targets
TARGET_MAX_TRIALS = 24
TARGET_MAX_SECONDS = 300

# text / image
TEXT_MIN_AVG_WORDS = 4            # string column whose mean word count is at least this (and mostly unique) => free text
TEXT_MAX_FEATURES = 300
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
IMAGE_SIZE = 16                   # handcrafted image features use an NxN thumbnail

# app / production
APP_PASSWORD_ENV = "NEXUSML_APP_PASSWORD"
