# src/healthcheck.py
import os

import litellm
from dotenv import load_dotenv

from src.llm_router import GROQ_MODEL, OPENROUTER_MODEL

load_dotenv()
TOOLS = [{"type": "function", "function": {
    "name": "ping", "description": "test",
    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
}}]


def check(label: str, model: str, key: str) -> None:
    try:
        litellm.completion(
            model=model, api_key=key, max_tokens=100, tools=TOOLS,
            messages=[{"role": "user", "content": "Say hi"}],
        )
        print(f"{label}: OK")
    except Exception as e:
        print(f"{label}: FAIL {type(e).__name__}: {str(e)[:160]}")


for i in range(1, 6):
    key = os.getenv(f"GROQ_API_KEY_{i}")
    if key:
        check(f"groq key {i}", GROQ_MODEL, key)

key = os.getenv("OPENROUTER_API_KEY_1")
if key:
    check("openrouter key 1", OPENROUTER_MODEL, key)