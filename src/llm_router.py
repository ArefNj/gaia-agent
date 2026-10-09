# LLM Router for handling multiple API providers
import os
from dotenv import load_dotenv
from litellm import Router

load_dotenv()

import logging
import litellm
import time

litellm.suppress_debug_info = True
for _name in ("LiteLLM Router", "LiteLLM"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-120b")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free")
GROQ_MODEL_2 = os.getenv("GROQ_MODEL_2", "groq/openai/gpt-oss-20b")


def _groq_deployments(group: str, model: str) -> list[dict]:
    deployments = []
    for i in range(1, 6):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            deployments.append({
                "model_name": group,
                "litellm_params": {"model": model, "api_key": key},
            })
    return deployments

def _openrouter_deployments() -> list[dict]:
    deployments = []
    for i in range(1, 6):
        key = os.getenv(f"OPENROUTER_API_KEY_{i}")
        if key:
            deployments.append({
                "model_name": "fallback",
                "litellm_params": {
                    "model": OPENROUTER_MODEL,
                    "api_key": key,
                },
            })
    return deployments

def build_router() -> Router:
    model_list = (
        _groq_deployments("primary", GROQ_MODEL)
        + _groq_deployments("secondary", GROQ_MODEL_2)
        + _openrouter_deployments()
    )
    
    if not model_list:
        raise RuntimeError("no valid API keys found for any model. Please set GROQ_API_KEY_1..5 or OPENROUTER_API_KEY_1..5 in your environment.")

    return Router(
        model_list=model_list,
        fallbacks=[{"primary": ["secondary", "fallback"]}],
        num_retries=4,          # how many times to retry a model if it fails
        timeout=60,
        routing_strategy="simple-shuffle",
        set_verbose=False,
    )

router = build_router()

def ask(messages: list[dict], **kwargs) -> str:
    """
    Ask a question to the LLM router and return the response.
    """
    response = router.completion(model="primary", messages=messages, **kwargs)
    return response.choices[0].message.content

def _is_rate_limit(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(s in text for s in ("ratelimit", "rate limit", "no deployments available", "429", "overloaded"))


def complete(**kwargs):
    """router.completion that waits and retries when every deployment is rate limited."""
    waits = [20, 45, 60, 90]
    for attempt, wait in enumerate(waits + [None], 1):
        try:
            return router.completion(**kwargs)
        except Exception as e:
            if wait is None or not _is_rate_limit(e):
                raise
            print(f"    [rate limit] all deployments busy, waiting {wait}s (retry {attempt}/{len(waits)})")
            time.sleep(wait)