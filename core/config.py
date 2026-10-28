"""Configuration management for StudyRAG.

Handles persistent application settings, validation with Pydantic v2,
and environment/path resolution outside the repository directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator


class ConfigError(Exception):
    """Raised when configuration loading, validation, or persistence fails."""


class Settings(BaseModel):
    """Application settings with defaults and constraints."""

    # Data & storage paths
    data_dir: str = Field(
        default="~/StudyRAG/data",
        description="Path to WhatsApp exported data directory.",
    )
    output_dir: str = Field(
        default="",
        description="Optional output directory for generated study packages.",
    )

    # Models & LLM configuration
    # Note: 'gemini-2.5-flash-lite' is a provisional default.
    # Must be verified against available Gemini API models prior to LLM integration.
    gemini_model: str = Field(
        default="gemini-2.5-flash-lite",
        description="Gemini model identifier for grading and generation.",
    )
    embed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model identifier for chunk embeddings.",
    )
    rerank_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Reranker model identifier for candidate relevance ranking.",
    )
    rerank_top_k: int = Field(
        default=20,
        ge=1,
        description="Maximum candidate chunks to rerank per query.",
    )
    rerank_device: str = Field(
        default="",
        description="Device for reranker inference ('cpu', 'cuda', or empty string for auto-detect).",
    )
    disable_reranker: bool = Field(
        default=False,
        description="Whether to disable local cross-encoder reranking and use hybrid fallback.",
    )
    context_ranking_weight: float = Field(
        default=0.08,
        ge=0.0,
        le=0.5,
        description="Weight for secondary WhatsApp context relevance scoring (0.0 to 0.5).",
    )
    disable_context_ranking: bool = Field(
        default=False,
        description="Whether to disable WhatsApp context-aware ranking adjustments.",
    )

    # Retrieval & grading parameters
    strictness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Retrieval strictness threshold (0.0 to 1.0).",
    )
    use_gemini_grading: bool = Field(
        default=True,
        description="Whether to use Gemini for relevance verification and grading.",
    )
    daily_token_cap: int = Field(
        default=100000,
        ge=0,
        description="Maximum daily token budget for LLM calls (0 = unlimited).",
    )
    max_candidates_graded: int = Field(
        default=20,
        ge=1,
        description="Maximum candidate chunks sent to LLM per topic query.",
    )
    explain_matches: bool = Field(
        default=False,
        description="Whether LLM should generate match explanations for retrieved files.",
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        description="Maximum API retry attempts on transient failures.",
    )

    # Syllabus & topic limits
    min_files_per_topic: int = Field(
        default=3,
        ge=1,
        description="Minimum target files to collect per syllabus topic.",
    )
    max_files_per_topic: int = Field(
        default=10,
        ge=1,
        description="Maximum target files to collect per syllabus topic.",
    )

    # OCR configuration
    tesseract_cmd: str = Field(
        default="",
        description="Path to tesseract executable if not in system PATH.",
    )
    ocr_lang: str = Field(
        default="eng",
        description="Tesseract language code for OCR processing.",
    )

    # Document chunking configuration
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Target maximum size of document chunks in characters (must be > 0).",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Character overlap between consecutive chunks (must be >= 0 and < chunk_size).",
    )

    @model_validator(mode="after")
    def validate_settings(self) -> Settings:
        """Validate interdependent settings including topic limits and chunking bounds."""
        if self.min_files_per_topic > self.max_files_per_topic:
            raise ValueError(
                f"min_files_per_topic ({self.min_files_per_topic}) cannot exceed "
                f"max_files_per_topic ({self.max_files_per_topic})"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @property
    def resolved_data_dir(self) -> Path:
        """Return data_dir with user home directory expanded."""
        return Path(self.data_dir).expanduser().resolve()

    @property
    def resolved_output_dir(self) -> Path | None:
        """Return output_dir with user home expanded, or None if empty."""
        if not self.output_dir.strip():
            return None
        return Path(self.output_dir).expanduser().resolve()

    @property
    def resolved_chroma_dir(self) -> Path:
        """Return chroma directory within the application home."""
        return get_chroma_dir()


DEFAULT_APP_DIR = Path.home() / ".studyrag"


def get_app_dir(custom_dir: Path | str | None = None) -> Path:
    """Return the application home directory.

    Priority:
    1. custom_dir parameter if provided
    2. STUDYRAG_HOME environment variable if set
    3. Default ~/.studyrag
    """
    if custom_dir:
        return Path(custom_dir).resolve()
    env_dir = os.environ.get("STUDYRAG_HOME")
    if env_dir:
        return Path(env_dir).resolve()
    return DEFAULT_APP_DIR.resolve()


def get_settings_path(app_dir: Path | str | None = None) -> Path:
    """Return the path to settings.json."""
    return get_app_dir(app_dir) / "settings.json"


def get_db_path(app_dir: Path | str | None = None) -> Path:
    """Return the path to studyrag.db."""
    return get_app_dir(app_dir) / "studyrag.db"


def get_chroma_dir(app_dir: Path | str | None = None) -> Path:
    """Return the path to the chroma vector store directory."""
    return get_app_dir(app_dir) / "chroma"


def get_log_path(app_dir: Path | str | None = None) -> Path:
    """Return the path to app.log."""
    return get_app_dir(app_dir) / "app.log"


def ensure_app_dirs(app_dir: Path | str | None = None) -> Path:
    """Safely create the application home directory and its subdirectories."""
    base_dir = get_app_dir(app_dir)
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir = get_chroma_dir(base_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ConfigError(f"Failed to create application directories at {base_dir}: {err}") from err
    return base_dir


def load(path: Path | str | None = None) -> Settings:
    """Load settings from JSON file or return defaults if file does not exist.

    If the settings file exists but contains invalid JSON or violates validation rules,
    raises ConfigError and does NOT overwrite or repair the corrupted file.
    """
    settings_path = Path(path).resolve() if path else get_settings_path()

    if not settings_path.exists():
        return Settings()

    try:
        raw_text = settings_path.read_text(encoding="utf-8")
    except OSError as err:
        raise ConfigError(f"Cannot read settings file at '{settings_path}': {err}") from err

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as err:
        raise ConfigError(
            f"Malformed JSON in settings file '{settings_path}' (line {err.lineno}, col {err.colno}): {err.msg}"
        ) from err

    if not isinstance(data, dict):
        raise ConfigError(
            f"Settings file '{settings_path}' must contain a JSON object, got {type(data).__name__}"
        )

    try:
        return Settings.model_validate(data)
    except ValidationError as err:
        raise ConfigError(
            f"Validation error in settings file '{settings_path}':\n{err}"
        ) from err


def save(settings: Settings, path: Path | str | None = None) -> Path:
    """Persist validated settings as formatted JSON.

    Creates parent directories safely if they do not exist.
    """
    if not isinstance(settings, Settings):
        raise TypeError(f"Expected Settings instance, got {type(settings).__name__}")

    target_path = Path(path).resolve() if path else get_settings_path()
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = settings.model_dump_json(indent=2)
        # Write atomically via temp file in same directory
        temp_path = target_path.with_suffix(".tmp")
        temp_path.write_text(json_data + "\n", encoding="utf-8")
        temp_path.replace(target_path)
    except OSError as err:
        raise ConfigError(f"Failed to save settings to '{target_path}': {err}") from err

    return target_path


def reset_defaults(path: Path | str | None = None) -> Settings:
    """Explicitly reset settings to default values and persist to disk."""
    default_settings = Settings()
    save(default_settings, path=path)
    return default_settings
