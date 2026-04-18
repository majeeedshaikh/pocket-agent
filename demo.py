"""
Pocket-Agent Gradio chatbot demo.
Multi-turn conversation with visible tool-call output.

Usage:
    pip install gradio
    python demo.py
    # Or on Colab:
    python demo.py --share
"""

import argparse
import json
import re
import sys

try:
    import gradio as gr
except ImportError:
    sys.exit("pip install gradio")

from inference import run


def format_tool_call(text: str) -> str:
    """Pretty-print a tool_call block for display."""
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not match:
        return text

    try:
        parsed = json.loads(match.group(1))
        tool = parsed.get("tool", "?")
        args = parsed.get("args", {})
        args_str = json.dumps(args, indent=2, ensure_ascii=False)
        return (
            f"🔧 **Tool call:** `{tool}`\n\n"
            f"```json\n{args_str}\n```\n\n"
            f"<details><summary>Raw</summary>\n\n```\n{text}\n```\n</details>"
        )
    except json.JSONDecodeError:
        return text


def chat_fn(message: str, history: list[tuple[str, str]]) -> str:
    """Gradio chat function. history is list of (user, assistant) tuples."""
    formatted_history = []
    for user_msg, asst_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        formatted_history.append({"role": "assistant", "content": asst_msg})

    raw_response = run(message, formatted_history)
    return format_tool_call(raw_response)


EXAMPLES = [
    ["What's the weather in Tokyo in Celsius?"],
    ["Convert 100 USD to EUR"],
    ["Schedule 'Team standup' for 2025-06-15"],
    ["What's on my calendar for 2025-06-15?"],
    ["Convert 5 kilometers to miles"],
    ["Run this SQL: SELECT * FROM users WHERE active = 1"],
    ["Tell me a joke"],  # refusal
    ["Send an email to my boss"],  # refusal - no email tool
]

DESCRIPTION = """
## Pocket-Agent 🤖

On-device mobile assistant powered by **Qwen3-0.6B** fine-tuned with QLoRA.
Runs fully offline — no network calls at inference time.

**Available tools:** `weather` · `calendar` · `convert` · `currency` · `sql`

For requests outside these tools, the model responds naturally without calling any tool.
"""


def build_interface(share: bool = False):
    with gr.Blocks(title="Pocket-Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
            show_label=True,
            render_markdown=True,
        )
        msg = gr.Textbox(
            label="Your message",
            placeholder="e.g. What's the weather in London in Celsius?",
            lines=2,
        )

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        gr.Examples(
            examples=EXAMPLES,
            inputs=msg,
            label="Try these examples",
        )

        gr.Markdown("""
---
**Scoring rubric (for reference):**
- `+1.0` — Exact tool + all args correct
- `+0.5` — Correct tool, ≥1 arg wrong
- `0.0` — Wrong tool or malformed JSON
- `−0.5` — Tool call when refusal was correct
        """)

        def respond(message, history):
            if not message.strip():
                return history, ""
            response = chat_fn(message, history)
            history.append((message, response))
            return history, ""

        submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print("Loading Pocket-Agent model...")
    # Warm up model (load weights into memory)
    try:
        _ = run("hello", [])
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: model pre-load failed ({e}). Will load on first request.")

    demo = build_interface(share=args.share)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
