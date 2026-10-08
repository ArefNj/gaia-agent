# src/agent.py
import json
from src.llm_router import router

SYSTEM_PROMPT = """You are a general-purpose AI agent answering benchmark questions.
Rules:
1. Reply with ONLY the exact final answer — no explanation, no "FINAL ANSWER:" prefix, no extra words or punctuation unless explicitly part of the answer.
2. If a number is requested, give just the number.
3. If a list is requested, follow the exact requested format (e.g. comma-separated, alphabetized).
4. Never guess when a tool can give you a verified answer — use the available tools for external facts, files, or computation.
5. If a tool result does not contain what you need, do NOT repeat the same query. Change strategy: use different keywords, a different page, or a different tool.
6. For Wikipedia content use read_wikipedia. If the question refers to a specific year or date (e.g. "2022 version", "as of July 2023"), pass the `date` argument.
7. When using read_webpage or read_wikipedia, write `question` as a specific description of the information you need. Content is filtered by a small extractor, so if the result looks incomplete, ask again with different wording instead of giving up.
"""
    
def _is_raw_tool_call(text: str) -> bool:
    """Some models print tool calls as plain text instead of using the tool-calling API."""
    return "<tool_call>" in text or "<function=" in text

class Agent:
    def __init__(self, tools: list | None = None):
        """
        tools: a list of dictionaries {"schema": <openai function schema>, "func": <callable>}
        """
        self.tools = tools or []
        self._tool_map = {t["schema"]["function"]["name"]: t["func"] for t in self.tools}
        self._schemas = [t["schema"] for t in self.tools]
            
    def run(self, question: str, max_steps: int = 8, verbose: bool = True) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for step in range(max_steps):
            kwargs = {"model": "primary", "messages": messages}
            if self._schemas:
                kwargs["tools"] = self._schemas
                kwargs["tool_choice"] = "auto"

            response = router.completion(**kwargs)
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                content = (message.content or "").strip()
                if _is_raw_tool_call(content):
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "Use the tool-calling API properly, or reply with the final answer only.",
                    })
                    continue
                return content

            messages.append(message.model_dump(exclude_none=True))

            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if verbose:
                    print(f"[step {step}] tool={name} args={args}")

                func = self._tool_map.get(name)
                if func is None:
                    result = f"Error: unknown tool '{name}'"
                else:
                    try:
                        result = func(**args)
                    except Exception as e:
                        result = f"Error running tool '{name}': {e}"

                if verbose:
                    print(f"    -> {str(result)[:200]!r}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                })

        # Out of steps: force the model to answer with what it has gathered so far
        messages.append({
            "role": "user",
            "content": "You are out of tool steps. Do NOT call any tools. Reply with only your best final answer in the required exact format.",
        })
        kwargs = {"model": "primary", "messages": messages}
        if self._schemas:
            kwargs["tools"] = self._schemas
            kwargs["tool_choice"] = "none"
        response = router.completion(**kwargs)
        if verbose:
            print(f"[step {step}] model={response.model}")
        answer = (response.choices[0].message.content or "").strip()
        return "UNKNOWN" if _is_raw_tool_call(answer) else answer