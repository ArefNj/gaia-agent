# src/runner.py
import argparse
import json
import time

from src.agent import Agent
from src.api_client import DATA_DIR, download_file, get_questions
from src.tools.web import WEB_TOOLS

from pathlib import Path
from src.tools.files import FILE_TOOLS

def build_prompt(question: dict, file_path, file_missing: bool = False) -> str:
    if file_path is not None:
        absolute = Path(file_path).resolve()
        return f'{question["question"]}\n\nAn attached file is available at this absolute local path: {absolute}'
    if file_missing:
        return (
            f'{question["question"]}\n\nNOTE: the attached file could not be downloaded and is NOT available. '
            "Do not search the file system for it. If the answer depends on the file, reply with FINAL ANSWER: UNKNOWN."
        )
    return question["question"]

def select_questions(questions: list[dict], only: str, limit: int, skip_files: bool):
    """Return (1-based index, question) pairs after applying the CLI filters."""
    indexed = list(enumerate(questions, 1))
    if only:
        wanted = {int(x) for x in only.split(",")}
        indexed = [(i, q) for i, q in indexed if i in wanted]
    if skip_files:
        indexed = [(i, q) for i, q in indexed if not q.get("file_name")]
    if limit:
        indexed = indexed[:limit]
    return indexed


def main():
    parser = argparse.ArgumentParser(description="Run the agent on the GAIA course questions.")
    parser.add_argument("--only", default="", help="Comma-separated 1-based question numbers, e.g. 1,3,5")
    parser.add_argument("--limit", type=int, default=0, help="Run at most N questions")
    parser.add_argument("--skip-files", action="store_true", help="Skip questions that have an attached file")
    parser.add_argument("--refresh", action="store_true", help="Re-download the question list")
    parser.add_argument("--quiet", action="store_true", help="Hide per-step tool logs")
    parser.add_argument("--preview", type=int, default=200, help="Characters of each tool result to print")
    args = parser.parse_args()

    questions = get_questions(refresh=args.refresh)
    agent = Agent(tools=WEB_TOOLS + FILE_TOOLS)
    results = []

    for idx, q in select_questions(questions, args.only, args.limit, args.skip_files):
        print(f"\n=== Question {idx}/{len(questions)} [{q['task_id'][:8]}] level={q.get('Level')} ===")
        print(q["question"][:300])

        file_path = None
        if q.get("file_name"):
            try:
                file_path = download_file(q["task_id"], q["file_name"])
                print(f"(attached file: {file_path})")
            except Exception as e:
                print(f"(failed to download attached file: {e})")

        file_missing = bool(q.get("file_name")) and file_path is None

        start = time.time()
        try:
            answer, status = agent.run(build_prompt(q, file_path, file_missing), verbose=not args.quiet, preview=args.preview), "ok"
        except Exception as e:
            answer, status = f"ERROR: {type(e).__name__}: {e}", "error"
        elapsed = time.time() - start

        print(f"ANSWER: {answer}  ({elapsed:.1f}s)")
        results.append({
            "index": idx,
            "task_id": q["task_id"],
            "file_name": q.get("file_name", ""),
            "answer": answer,
            "status": status,
            "seconds": round(elapsed, 1),
        })

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== SUMMARY =====")
    for r in results:
        file_flag = "F" if r["file_name"] else "-"
        print(f'{r["index"]:>2} {file_flag} {r["status"]:<5} {r["seconds"]:>6}s  {r["answer"][:80]!r}')


if __name__ == "__main__":
    main()