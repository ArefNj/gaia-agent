# src/tools/base.py


def _tool(name: str, description: str, properties: dict, required: list, func) -> dict:
    return {
        "schema": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        },
        "func": func,
    }

def make_tool(name: str, description: str, properties: dict, required: list, func) -> dict:
    return {
        "schema": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        },
        "func": func,
    }