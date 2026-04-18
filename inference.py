"""
Pocket-Agent inference interface.
Exposes: def run(prompt: str, history: list[dict]) -> str

Grader contract: this file must contain no network imports.
Allowed imports: llama_cpp, json, re, os, pathlib
"""

# ── NO NETWORK IMPORTS (AST-scanned by grader) ──────────────────────────────
import json
import os
import re
from pathlib import Path

# ── MODEL LOADING ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
_MODEL_PATH = ROOT / "artifacts" / "model-q4km.gguf"

_model = None

SYSTEM_PROMPT = """\
You are Pocket-Agent, a compact mobile assistant. You have access to exactly these five tools:

weather   – args: location (string), unit ("C" or "F")
calendar  – args: action ("list" or "create"), date (YYYY-MM-DD), title (string, only for create)
convert   – args: value (number), from_unit (string), to_unit (string)
currency  – args: amount (number), from (ISO-4217 code), to (ISO-4217 code)
sql       – args: query (string)

Rules (follow exactly):
1. If the user's request clearly maps to one of the five tools, respond with ONLY a <tool_call> block:
   <tool_call>
   {"tool": "<name>", "args": {<args>}}
   </tool_call>
2. If the conversation has prior turns, resolve references ("that", "it", "there") against history.
3. If no tool fits — chitchat, ambiguous reference with no history, impossible request, or a tool that does not exist — respond in plain natural language WITHOUT any <tool_call> block.
4. Never invent tool names. Never emit partial JSON. Never add commentary outside the <tool_call> block when a tool call is appropriate.\
"""


def _get_model():
    global _model
    if _model is not None:
        return _model

    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError("Install llama-cpp-python: pip install llama-cpp-python")

    model_path = str(_MODEL_PATH)
    if not _MODEL_PATH.exists():
        # Fallback: search artifacts directory
        gguf_files = list((ROOT / "artifacts").glob("*.gguf"))
        if gguf_files:
            model_path = str(gguf_files[0])
        else:
            raise FileNotFoundError(
                f"No GGUF model found at {_MODEL_PATH}\n"
                "Run: python quantize.py"
            )

    n_threads = int(os.environ.get("LLAMA_N_THREADS", "4"))
    n_ctx = int(os.environ.get("LLAMA_N_CTX", "4096"))

    _model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=512,
        n_gpu_layers=0,   # CPU-only for Colab CPU runtime
        verbose=False,
        chat_format="chatml",
    )
    return _model


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def run(prompt: str, history: list[dict]) -> str:
    """
    Args:
        prompt:  Current user message.
        history: List of prior turns, each {"role": "user"|"assistant", "content": "..."}.
                 May be empty for single-turn requests.
    Returns:
        str: Model response. Either a <tool_call>...</tool_call> block or plain text.
    """
    model = _get_model()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    result = model.create_chat_completion(
        messages=messages,
        max_tokens=200,
        temperature=0.0,
        top_p=1.0,
        repeat_penalty=1.0,
        stop=["</tool_call>"],
    )

    raw = result["choices"][0]["message"]["content"].strip()

    # If the model started a tool_call block, ensure it's closed
    if "<tool_call>" in raw and "</tool_call>" not in raw:
        raw = raw + "\n</tool_call>"

    return _clean_output(raw)


def _clean_output(text: str) -> str:
    """Ensure tool_call JSON is valid; strip trailing whitespace."""
    if "<tool_call>" not in text:
        return text.strip()

    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not match:
        return text.strip()

    try:
        parsed = json.loads(match.group(1))
        valid_tools = {"weather", "calendar", "convert", "currency", "sql"}
        if parsed.get("tool") not in valid_tools:
            # Model hallucinated a tool name — return as plain text refusal
            return "I'm sorry, I don't have a tool that can help with that."
        # Re-serialize to ensure clean JSON
        clean_json = json.dumps(parsed, ensure_ascii=False)
        return f"<tool_call>\n{clean_json}\n</tool_call>"
    except json.JSONDecodeError:
        # Malformed JSON inside tool_call — treat as refusal
        return "I'm sorry, I wasn't able to process that request."


# ── CLI HELPER ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py '<prompt>'")
        sys.exit(1)

    prompt_text = sys.argv[1]
    history_arg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []

    print(f"Prompt : {prompt_text}")
    print(f"History: {len(history_arg)} turns")
    print("─" * 40)
    response = run(prompt_text, history_arg)
    print(response)
