# src/agent.py
import json
from src.llm_router import router

SYSTEM_PROMPT = """You are a general-purpose AI agent answering benchmark questions.
Rules:
1. Reply with ONLY the exact final answer — no explanation, no "FINAL ANSWER:" prefix, no extra words or punctuation unless explicitly part of the answer.
2. If a number is requested, give just the number.
3. If a list is requested, follow the exact requested format (e.g. comma-separated, alphabetized).
4. Never guess when a tool can give you a verified answer — use the available tools for external facts, files, or computation.
"""

class Agent:
    def __init__(self, tools: list | None = None):
        """
        tools: a list of dictionaries {"schema": <openai function schema>, "func": <callable>}
        """
        self.tools = tools or []
        self._tool_map = {t["schema"]["function"]["name"]: t["func"] for t in self.tools}
        self._schemas = [t["schema"] for t in self.tools]

    def run(self, question: str, max_steps: int = 8) -> str:
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
                return (message.content or "").strip()

            messages.append(message.model_dump(exclude_none=True))

            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                func = self._tool_map.get(name)
                if func is None:
                    result = f"Error: unknown tool '{name}'"
                else:
                    try:
                        result = func(**args)
                    except Exception as e:
                        result = f"Error running tool '{name}': {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                })

        return "Error: maximum steps exceeded without reaching a final answer."