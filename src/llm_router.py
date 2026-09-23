# LLM Router for handling multiple API providers
import os
from dotenv import load_dotenv
from litellm import Router

load_dotenv()

GROQ_MODEL = "groq/llama-3.3-70b-versatile"
OPENROUTER_MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"  

def _groq_deployments() -> list[dict]:
    deployments = []
    for i in range(1, 6):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            deployments.append({
                "model_name": "primary",
                "litellm_params": {
                    "model": GROQ_MODEL,
                    "api_key": key,
                },
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
    model_list = _groq_deployments() + _openrouter_deployments()
    if not model_list:
        raise RuntimeError("no valid API keys found for any model. Please set GROQ_API_KEY_1..5 or OPENROUTER_API_KEY_1..5 in your environment.")

    return Router(
        model_list=model_list,
        fallbacks=[{"primary": ["fallback"]}],
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