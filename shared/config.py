"""Shared settings. Everything can be overridden with environment variables / .env"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # the repo folder
load_dotenv(ROOT / ".env")  # read API keys and overrides from .env (real environment variables win)


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


def _b(name, default="0"):
    return os.getenv(name, default).lower() in ("1", "true", "yes")


# ---- agent loop ----
SCORE_THRESHOLD = _f("SCORE_THRESHOLD", 0.75)   # accept app when score >= this
MAX_ROUNDS = _i("MAX_ROUNDS", 4)                # round budget
MIN_IMPROVEMENT = _f("MIN_IMPROVEMENT", 0.02)   # early stop if a round doesn't beat best by this
BUILD_TIMEOUT_S = _f("BUILD_TIMEOUT_S", 180)    # per-build timeout

# ---- models ----
# Checked 22 Nov 2026: gemini-2.5-flash is closed to new keys and Groq removed Llama 4 Scout.
# The free tier allows only 20 requests/day PER MODEL, so generation tries these in order and moves on
# when one is out of quota or overloaded. The first one is the main model.
GEMINI_MODELS = os.getenv("GEMINI_MODELS", "gemini-3.5-flash,gemini-3.7-flash,gemini-3.8-flash,"
                                           "gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",")
GEMINI_MODEL = GEMINI_MODELS[0]
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")  # only vision model left on Groq
# Groq's free tier allows 1000 output tokens/min, too few for a whole page -> off for generation by default
GROQ_FALLBACK = _b("GROQ_FALLBACK", "0")
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", GROQ_VISION_MODEL)

# ---- scorer ----
CLIP_MODEL = os.getenv("CLIP_MODEL", "google/siglip-base-patch16-224")  # frozen feature extractor
CLIP_LOW = _f("CLIP_LOW", 0.55)    # cosine mapped to 0.0  (calibrate on hand-made pages!)
CLIP_HIGH = _f("CLIP_HIGH", 0.90)  # cosine mapped to 1.0
CLIP_WEIGHT = _f("CLIP_WEIGHT", 0.6)  # final = w*clip + (1-w)*judge
USE_REAL_SCORER = _b("USE_REAL_SCORER", "0")  # 0 = placeholder scorer, 1 = scoring/score.py

# ---- sandbox / rendering ----
SANDBOX_URL = os.getenv("SANDBOX_URL", "")  # empty = render with local Playwright
VIEWPORT_W = _i("VIEWPORT_W", 1280)  # desktop webpage, same as the Sketch2Code training data
VIEWPORT_H = _i("VIEWPORT_H", 800)

# ---- prompts / canary ----
STABLE_PROMPT_VERSION = os.getenv("STABLE_PROMPT_VERSION", "v1")

# ---- experiment tracking ----
# SQLite database, because the MLflow model registry needs a database (plain files are not enough)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

# ---- app ----
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
APPS_DIR = os.getenv("APPS_DIR", "data/apps")

# ---- testing without API keys / browser ----
MOCK_MODE = _b("MOCK_MODE", "0")
