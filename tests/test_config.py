"""Tests for StudyRAG configuration management."""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from core.config import (
    ConfigError,
    Settings,
    ensure_app_dirs,
    get_app_dir,
    get_chroma_dir,
    get_db_path,
    get_log_path,
    get_settings_path,
    load,
    reset_defaults,
    save,
)


def test_default_settings_valid():
    """Verify that default settings instantiate with expected values and pass validation."""
    s = Settings()
    assert s.data_dir == "~/StudyRAG/data"
    assert s.output_dir == ""
    assert s.gemini_model == "gemini-2.5-flash-lite"
    assert s.embed_model == "BAAI/bge-small-en-v1.5"
    assert s.rerank_model == "BAAI/bge-reranker-base"
    assert s.strictness == 0.5
    assert s.use_gemini_grading is True
    assert s.daily_token_cap == 100000
    assert s.max_candidates_graded == 20
    assert s.explain_matches is False
    assert s.max_retries == 1
    assert s.min_files_per_topic == 3
    assert s.max_files_per_topic == 10
    assert s.tesseract_cmd == ""
    assert s.ocr_lang == "eng"
    assert s.chunk_size == 1000
    assert s.chunk_overlap == 200
    assert s.resolved_data_dir.is_absolute()
    assert s.resolved_output_dir is None


def test_strictness_validation():
    """Strictness must be constrained to 0.0 - 1.0."""
    # Valid bounds
    Settings(strictness=0.0)
    Settings(strictness=1.0)
    Settings(strictness=0.75)

    # Invalid below 0.0
    with pytest.raises(ValidationError):
        Settings(strictness=-0.1)

    # Invalid above 1.0
    with pytest.raises(ValidationError):
        Settings(strictness=1.05)


def test_topic_limits_validation():
    """min_files_per_topic cannot exceed max_files_per_topic."""
    Settings(min_files_per_topic=5, max_files_per_topic=5)  # Equal is valid
    Settings(min_files_per_topic=3, max_files_per_topic=10)

    with pytest.raises(ValidationError):
        Settings(min_files_per_topic=10, max_files_per_topic=5)


def test_chunking_config_validation():
    """chunk_size must be > 0 and chunk_overlap must be >= 0 and strictly < chunk_size."""
    # Valid configurations
    Settings(chunk_size=500, chunk_overlap=100)
    Settings(chunk_size=100, chunk_overlap=0)

    # Invalid non-positive chunk_size
    with pytest.raises(ValidationError):
        Settings(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValidationError):
        Settings(chunk_size=-100, chunk_overlap=0)

    # Invalid negative overlap
    with pytest.raises(ValidationError):
        Settings(chunk_size=500, chunk_overlap=-10)

    # Invalid overlap >= chunk_size
    with pytest.raises(ValidationError):
        Settings(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValidationError):
        Settings(chunk_size=500, chunk_overlap=600)


def test_save_and_load_roundtrip(tmp_path: Path):
    """Saving and loading settings must preserve all values faithfully."""
    settings_file = tmp_path / "settings.json"
    custom = Settings(
        data_dir="C:/Custom/Data",
        output_dir="C:/Custom/Output",
        gemini_model="gemini-1.5-pro",
        strictness=0.85,
        daily_token_cap=50000,
        max_candidates_graded=35,
        explain_matches=True,
        min_files_per_topic=4,
        max_files_per_topic=15,
        ocr_lang="fra",
    )
    written_path = save(custom, path=settings_file)
    assert written_path == settings_file
    assert settings_file.exists()

    loaded = load(path=settings_file)
    assert loaded.data_dir == "C:/Custom/Data"
    assert loaded.output_dir == "C:/Custom/Output"
    assert loaded.gemini_model == "gemini-1.5-pro"
    assert loaded.strictness == 0.85
    assert loaded.daily_token_cap == 50000
    assert loaded.max_candidates_graded == 35
    assert loaded.explain_matches is True
    assert loaded.min_files_per_topic == 4
    assert loaded.max_files_per_topic == 15
    assert loaded.ocr_lang == "fra"


def test_load_nonexistent_returns_defaults(tmp_path: Path):
    """Loading from a non-existent path returns default Settings."""
    missing = tmp_path / "nonexistent.json"
    loaded = load(path=missing)
    assert loaded == Settings()


def test_malformed_json_produces_error_and_preserves_file(tmp_path: Path):
    """A malformed JSON file produces ConfigError and is NOT silently overwritten."""
    corrupted_file = tmp_path / "corrupted_settings.json"
    malformed_content = "{ invalid json content: 123, "
    corrupted_file.write_text(malformed_content, encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load(path=corrupted_file)

    assert "Malformed JSON" in str(exc_info.value)
    # Ensure file was not overwritten with defaults
    assert corrupted_file.read_text(encoding="utf-8") == malformed_content


def test_validation_error_in_file_produces_error_and_preserves_file(tmp_path: Path):
    """A settings file violating validation rules produces ConfigError and is NOT replaced."""
    invalid_file = tmp_path / "invalid_settings.json"
    invalid_data = {
        "strictness": 2.5,  # Violates le=1.0
    }
    invalid_file.write_text(json.dumps(invalid_data), encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load(path=invalid_file)

    assert "Validation error" in str(exc_info.value)
    # Ensure original content is preserved
    assert json.loads(invalid_file.read_text(encoding="utf-8")) == invalid_data


def test_reset_defaults(tmp_path: Path):
    """reset_defaults writes default settings and returns them."""
    settings_file = tmp_path / "settings.json"
    custom = Settings(strictness=0.9)
    save(custom, path=settings_file)

    assert load(path=settings_file).strictness == 0.9

    res = reset_defaults(path=settings_file)
    assert res.strictness == 0.5
    assert load(path=settings_file).strictness == 0.5


def test_app_dirs_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Path helpers construct correct file paths within target directory."""
    test_home = tmp_path / "custom_home"
    monkeypatch.setenv("STUDYRAG_HOME", str(test_home))

    assert get_app_dir() == test_home
    assert get_settings_path() == test_home / "settings.json"
    assert get_db_path() == test_home / "studyrag.db"
    assert get_chroma_dir() == test_home / "chroma"
    assert get_log_path() == test_home / "app.log"

    ensure_app_dirs()
    assert test_home.exists()
    assert (test_home / "chroma").exists()
