"""
Configuration loader.
Reads .env for secrets and config/default.yaml for runtime settings.
All other modules import `settings` from here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = three levels above this file  (src/absa/utils/config.py → root)
ROOT = Path(__file__).resolve().parents[3]

load_dotenv(ROOT / ".env")


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            "Copy .env.example → .env and fill in your credentials."
        )
    return val


class Settings:
    def __init__(self) -> None:
        self._yaml: dict[str, Any] = {}
        yaml_path = ROOT / "config" / "default.yaml"
        if yaml_path.exists():
            with yaml_path.open() as f:
                self._yaml = yaml.safe_load(f) or {}

    # ---- Secrets (lazily validated on first access) ----

    @property
    def reddit_client_id(self) -> str:
        return _require("CLIENT_ID")

    @property
    def reddit_client_secret(self) -> str:
        return _require("CLIENT_SECRET")

    @property
    def reddit_user_agent(self) -> str:
        return _require("USER_AGENT")

    @property
    def gemini_api_key(self) -> str:
        return _require("GEMINI_API_KEY")

    # ---- Fetch settings ----

    @property
    def fetch_post_limit(self) -> int:
        return self._yaml.get("fetch", {}).get("post_limit", 50)

    @property
    def fetch_comment_limit(self) -> int:
        return self._yaml.get("fetch", {}).get("comment_limit", 200)

    @property
    def fetch_time_filter(self) -> str:
        return self._yaml.get("fetch", {}).get("time_filter", "year")

    @property
    def fetch_min_score(self) -> int:
        return self._yaml.get("fetch", {}).get("min_score", 2)

    @property
    def fetch_min_comments(self) -> int:
        return self._yaml.get("fetch", {}).get("min_comments", 5)

    @property
    def default_subreddits(self) -> list[str]:
        return self._yaml.get("fetch", {}).get(
            "default_subreddits", ["reviews", "technology", "gadgets"]
        )

    # ---- Preprocess settings ----

    @property
    def min_sentence_tokens(self) -> int:
        return self._yaml.get("preprocess", {}).get("min_sentence_tokens", 4)

    @property
    def max_sentence_tokens(self) -> int:
        return self._yaml.get("preprocess", {}).get("max_sentence_tokens", 80)

    @property
    def spacy_model(self) -> str:
        return self._yaml.get("preprocess", {}).get("spacy_model", "en_core_web_sm")

    # ---- Topic model settings ----

    @property
    def embedding_model(self) -> str:
        return self._yaml.get("topic_model", {}).get(
            "embedding_model", "all-MiniLM-L6-v2"
        )

    # ---- ABSA settings ----

    @property
    def gemini_model(self) -> str:
        return self._yaml.get("absa", {}).get("gemini_model", "gemini-2.0-flash")

    # ---- Paths ----

    @property
    def root(self) -> Path:
        return ROOT

    @property
    def raw_dir(self) -> Path:
        return ROOT / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return ROOT / "data" / "processed"

    @property
    def results_dir(self) -> Path:
        return ROOT / "data" / "results"

    @property
    def outputs_dir(self) -> Path:
        return ROOT / "outputs"


settings = Settings()
