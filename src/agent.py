# src/agent.py
import json
from src.llm_router import router
from src.llm_router import complete
import re
import inspect

SYSTEM_PROMPT = """You are a general-purpose AI agent answering benchmark questions with exact-match grading.
Rules:
1. You may reason briefly, but you MUST end your reply with a final line in exactly this form:
   FINAL ANSWER: <answer>
2. The answer must be a number, OR as few words as possible, OR a comma-separated list of numbers and/or strings.
3. For a number: digits only, no thousands separators, no units or % sign unless the question asks for them.
4. For a string: no articles and no abbreviations (write city names in full) unless the question asks otherwise.
5. For a comma-separated list: apply the rules above to each element, follow any requested ordering, and separate elements with ", ".
6. No trailing period, no quotes and no markdown inside the final answer.
7. Use tools for external facts, files and computation instead of answering from memory. Evidence must come from primary or reference sources: crossword sites, puzzle-answer sites and pages that merely repeat the question are not evidence.
8. If a tool result does not contain what you need, do NOT repeat the same query. Change strategy: different keywords, a different page, or a different tool. Never invent tool parameters or URL query strings.
9. For Wikipedia use read_wikipedia. If the question refers to a specific year or date (e.g. "2022 version", "as of July 2023"), pass the `date` argument.
10. read_webpage and read_wikipedia return the sections that best match the wording of `question`. Describe the information you need with words likely to appear in the target section or its title; if the result looks incomplete, ask again with different wording.
11. If a publisher site blocks you (captcha, 403), look for the arXiv version: https://arxiv.org/abs/<id> and https://arxiv.org/pdf/<id>.
12. Attached files: a local path is given in the question. Inspect it first with the matching tool and never answer from memory: read_file for spreadsheets, PDF and text; transcribe_audio for audio; run_python(file_path=...) for .py files (never retype their code); run_python with pandas for calculations over file data. Always print() what you need to see.
13. For lists transcribed from audio, keep the descriptive words that belong to each item as the speaker said it (e.g. "freshly ground black pepper" stays as is); drop only quantities, units and cooking instructions.
14. If you cannot find the answer after trying different approaches, reply FINAL ANSWER: UNKNOWN. Never invent an answer.
"""


def extract_final_answer(text: str) -> str:
    """Pull the answer from a 'FINAL ANSWER: ...' line; fall back to the last non-empty line."""
    if not text or not text.strip():
        return ""
    matches = re.findall(r"final answer\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if matches:
        answer = matches[-1]
    else:
        answer = [line for line in text.strip().splitlines() if line.strip()][-1]
    answer = answer.strip().strip("*`\"'").strip()
    return re.sub(r"[.\s]+$", "", answer)
    
def _is_raw_tool_call(text: str) -> bool:
    """Some models print tool calls as plain text instead of using the tool-calling API."""
    return "<tool_call>" in text or "<function=" in text

def _clean_assistant_message(message) -> dict:
    """Keep only the fields every provider accepts (drops provider_specific_fields etc.)."""
    cleaned = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        cleaned["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
            }
            for call in message.tool_calls
        ]
    return cleaned

KEEP_FULL_TOOL_RESULTS = 2
OLD_TOOL_RESULT_CHARS = 1200


def _trim_history(messages: list[dict]) -> None:
    """Shrink all but the most recent tool results to save tokens (in place)."""
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    for i in tool_idx[:-KEEP_FULL_TOOL_RESULTS]:
        content = messages[i]["content"]
        if len(content) > OLD_TOOL_RESULT_CHARS:
            messages[i]["content"] = content[:OLD_TOOL_RESULT_CHARS] + "\n...[older tool output truncated]"


def _filter_args(func, args: dict) -> dict:
    """Drop arguments the tool does not accept (models sometimes invent parameters)."""
    params = inspect.signature(func).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return args
    return {k: v for k, v in args.items() if k in params}

class Agent:
    def __init__(self, tools: list | None = None):
        """
        tools: a list of dictionaries {"schema": <openai function schema>, "func": <callable>}
        """
        self.tools = tools or []
        self._tool_map = {t["schema"]["function"]["name"]: t["func"] for t in self.tools}
        self._schemas = [t["schema"] for t in self.tools]
            
    def run(self, question: str, max_steps: int = 12, verbose: bool = True, preview: int = 200) -> str:        
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
            if verbose:
                print(f"[step {step}] model={response.model}")
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
                return extract_final_answer(content)

            messages.append(_clean_assistant_message(message))
            
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
                    print(f"    -> {str(result)[:preview]!r}")

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
        return "UNKNOWN" if _is_raw_tool_call(answer) else extract_final_answer(answer)