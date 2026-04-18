# Pocket-Agent — Hackathon Problem Statement

## Task

Fine-tune an open-weight base model (**up to 2B parameters**) to perform structured tool calls for an on-device mobile assistant. The fine-tuned model must:

- Match a frontier teacher LLM's tool-calling accuracy on a held-out adversarial test set
- Try to fit in **≤ 800 MB** after quantization
- Run at **≤ 200 ms per turn on CPU or 80-150 tok/sec**
- Operate fully offline (no network calls at inference)

**Time:** 2 hours
**Compute:** Free Google Colab T4 (16 GB VRAM)
**Base model:** Any open-weight model ≤ 2B parameters (your choice — document it in your README)

---

## Tool Schema

The model emits JSON wrapped in `<tool_call>...</tool_call>` tags, or plain natural language for refusals. Five tools, schema is final:

```json
{"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}
{"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}
{"tool": "convert",  "args": {"value": "number", "from_unit": "string", "to_unit": "string"}}
{"tool": "currency", "args": {"amount": "number", "from": "ISO3", "to": "ISO3"}}
{"tool": "sql",      "args": {"query": "string"}}
```

---

## Required Behaviors

1. **Single-turn tool calls** — emit a valid tool call for unambiguous requests
2. **Multi-turn context** — resolve references like "convert that to euros" against prior turns
3. **Refusals** — emit a plain-text response (no tool call) when no tool fits: chitchat, ambiguous references with no history, or requests for tools that don't exist
4. **Argument fidelity** — units, ISO codes, dates, and numerical values must match the user's intent exactly

---

## Submission

Submit a public GitHub repository containing:

- **Complete training codebase** — synthetic data generation, fine-tuning, quantization, evaluation. Reproducible end-to-end via `make all` or a documented sequence of commands.
- **Trained artifacts** — LoRA adapter and quantized model file (or a script that produces them from the trained adapter).
- **Working chatbot demo** — a runnable interface (Gradio, Streamlit, or CLI) that loads your quantized model and supports multi-turn conversation with visible tool-call output. Must run on Colab CPU runtime out of the box.
- **`inference.py`** — exposes `def run(prompt: str, history: list[dict]) -> str` for the grader.
- **`README.md`** — setup instructions, design decisions, model choices, what worked, what didn't.

---

## Grading

Your `inference.py` is run against a **20-example private test set**, split into four slices:

| Slice | Count | Content |
|---|---|---|
| **A. In-distribution** | 8 | Standard tool calls similar to public examples |
| **B. Paraphrased** | 5 | Same intents, different wording |
| **C. Adversarial** | 5 | Typos, code-switched prompts (Hindi/Urdu/Spanish/Arabic + English), unit ambiguity, hallucination-bait entities |
| **D. Refusals & multi-turn** | 2 | Impossible tools, ambiguous references, 2–3 turn conversations |

### Per-example scoring

| Score | Condition |
|---|---|
| **+1.0** | Exact tool match, all args correct (numerical args within ±1%) |
| **+0.5** | Correct tool, ≥1 arg wrong |
| **0.0** | Wrong tool, malformed JSON, or wrong refusal decision |
| **−0.5** | Emitted a tool call when refusal was correct |

### Hard gates (zero score if any fail)

| Gate | Check |
|---|---|
| Adapter loads on the declared base model (≤ 2B params) in `transformers` v5 | Automated |
| Quantized model ≤ 500 MB | Automated |
| Mean inference latency ≤ 200 ms/turn on Colab CPU runtime | Timed over 20 examples |
| Training data shares zero prompts with the public test set | SHA-256 hash comparison |
| `inference.py` contains no network imports (`requests`, `urllib`, `http`, `socket`) | AST scan |
| Chatbot demo launches and accepts input | Manual judge check |

### Bonus points (max +25)

| Bonus | Condition |
|---|---|
| **+10** | Beat GPT-4o-mini's zero-shot Slice C score (baseline published post-grading) |
| **+10** | Quantized model ≤ 250 MB and all gates still pass |
| **+5** | README error analysis shows specific debugging insight |

---

## Starter Pack

```
starter/
├── public_test.jsonl         40 examples (dev set, not in private grading set)
├── eval_harness_contract.py  exact interface the grader will call
├── tool_schemas.json         the 5 tool schemas
└── teacher_examples.jsonl    20 hand-crafted seed examples
```

---

## Submission Process

Submit a GitHub repository link to the hackathon platform before the deadline. Grading runs on a clean Colab T4: clone repo → load adapter onto base model → quantize → score on the private 20-example set → judges manually launch the chatbot demo. Per-submission grading time: ~5 minutes. Leaderboard updates every 15 minutes.
