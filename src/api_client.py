# src/api_client.py
import json
import os
import shutil
from pathlib import Path

import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

GAIA_REPO = "gaia-benchmark/GAIA"
GAIA_SPLIT_DIR = "2023/validation"

BASE_URL = "https://agents-course-unit4-scoring.hf.space"
DATA_DIR = Path("data")
FILES_DIR = DATA_DIR / "files"
QUESTIONS_CACHE = DATA_DIR / "questions.json"


def get_questions(refresh: bool = False) -> list[dict]:
    """Return the benchmark questions, cached locally after the first download."""
    if QUESTIONS_CACHE.exists() and not refresh:
        return json.loads(QUESTIONS_CACHE.read_text(encoding="utf-8"))

    resp = requests.get(f"{BASE_URL}/questions", timeout=30)
    resp.raise_for_status()
    questions = resp.json()

    DATA_DIR.mkdir(exist_ok=True)
    QUESTIONS_CACHE.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    return questions


def _download_from_gaia(file_name: str) -> Path:
    """Download one attachment from the (gated) GAIA dataset."""
    local = hf_hub_download(
        repo_id=GAIA_REPO,
        repo_type="dataset",
        filename=f"{GAIA_SPLIT_DIR}/{file_name}",
        token=os.getenv("HF_TOKEN"),
    )
    return Path(local)


def _download_from_scoring_api(task_id: str, target: Path) -> Path:
    resp = requests.get(f"{BASE_URL}/files/{task_id}", timeout=60)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    return target


def download_file(task_id: str, file_name: str) -> Path:
    """Download the file attached to a task (cached). Tries the GAIA dataset first, then the scoring API."""
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    target = FILES_DIR / file_name
    if target.exists():
        return target

    errors = []
    try:
        shutil.copy(_download_from_gaia(file_name), target)
        return target
    except Exception as e:
        errors.append(f"GAIA dataset: {type(e).__name__}: {e}")
    try:
        return _download_from_scoring_api(task_id, target)
    except Exception as e:
        errors.append(f"scoring API: {type(e).__name__}: {e}")

    raise RuntimeError(" | ".join(errors))