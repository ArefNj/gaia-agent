# src/store.py
import json

from src.api_client import DATA_DIR

ANSWERS_FILE = DATA_DIR / "answers.json"
BAD_ANSWERS = {"", "UNKNOWN", "NOT FOUND"}


def load_answers() -> dict[str, str]:
    if ANSWERS_FILE.exists():
        return json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
    return {}


def save_answer(task_id: str, answer: str, status: str) -> None:
    """Merge one answer into answers.json. Errors are never stored, and UNKNOWN never replaces a stored answer."""
    if status != "ok":
        return
    answers = load_answers()
    if answer.strip().upper() in BAD_ANSWERS and task_id in answers:
        return
    answers[task_id] = answer
    DATA_DIR.mkdir(exist_ok=True)
    ANSWERS_FILE.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")