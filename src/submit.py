# src/submit.py
import argparse
import json
import os

import requests
from dotenv import load_dotenv

from src.api_client import BASE_URL, get_questions
from src.store import load_answers

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Submit stored answers to the course scoring API.")
    parser.add_argument("--code-url", required=True, help="Public URL of your agent code")
    parser.add_argument("--dry-run", action="store_true", help="Print the payload instead of sending it")
    args = parser.parse_args()

    username = (os.getenv("HF_USERNAME") or "").strip()
    if not username:
        raise SystemExit("Set HF_USERNAME in .env")

    questions = get_questions()
    stored = load_answers()

    answers, missing = [], []
    for idx, q in enumerate(questions, 1):
        answer = stored.get(q["task_id"])
        if answer is None:
            missing.append(idx)
            answer = "UNKNOWN"
        answers.append({"task_id": q["task_id"], "submitted_answer": answer})

    print(f"{len(questions) - len(missing)}/{len(questions)} questions have stored answers. Missing (sent as UNKNOWN): {missing}")

    payload = {"username": username, "agent_code": args.code_url, "answers": answers}
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    resp = requests.post(f"{BASE_URL}/submit", json=payload, timeout=60)
    print(resp.status_code, resp.text)


if __name__ == "__main__":
    main()