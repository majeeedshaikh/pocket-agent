"""
Evaluation harness for Pocket-Agent.
Scores model output against public_test.jsonl (or any labeled JSONL).

Usage:
    python eval/evaluate.py --test-file data/public_test.jsonl
    python eval/evaluate.py --test-file data/public_test.jsonl --verbose
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Scoring constants (from problem statement) ────────────────────────────────

SCORE_EXACT = 1.0
SCORE_CORRECT_TOOL_BAD_ARGS = 0.5
SCORE_WRONG = 0.0
SCORE_FALSE_TOOL_CALL = -0.5
NUMERICAL_TOLERANCE = 0.01   # ±1%


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_tool_call(text: str) -> dict | None:
    """Extract and parse the <tool_call> JSON block. Returns None if not present."""
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"_malformed": True, "_raw": match.group(1)}


def args_match(predicted_args: dict, expected_args: dict) -> bool:
    """Check if predicted args match expected args (numerical ±1%, strings exact)."""
    if set(predicted_args.keys()) != set(expected_args.keys()):
        return False
    for key in expected_args:
        p_val = predicted_args[key]
        e_val = expected_args[key]
        if isinstance(e_val, (int, float)):
            try:
                p_num = float(p_val)
                e_num = float(e_val)
                if e_num == 0:
                    if p_num != 0:
                        return False
                elif abs(p_num - e_num) / abs(e_num) > NUMERICAL_TOLERANCE:
                    return False
            except (ValueError, TypeError):
                return False
        else:
            if str(p_val).strip() != str(e_val).strip():
                return False
    return True


def score_example(predicted_text: str, expected: dict) -> tuple[float, str]:
    """
    Returns (score, reason).
    expected keys: tool (str or None), args (dict or None)
    """
    pred_call = parse_tool_call(predicted_text)
    expected_tool = expected.get("tool")
    expected_args = expected.get("args")

    # Ground truth: should be a refusal (no tool call)
    if expected_tool is None:
        if pred_call is None:
            return SCORE_EXACT, "correct refusal"
        else:
            return SCORE_FALSE_TOOL_CALL, f"false tool call: {pred_call.get('tool', '?')}"

    # Ground truth: should be a tool call
    if pred_call is None:
        return SCORE_WRONG, "missed tool call (responded with plain text)"

    if pred_call.get("_malformed"):
        return SCORE_WRONG, "malformed JSON in <tool_call>"

    pred_tool = pred_call.get("tool", "")
    if pred_tool != expected_tool:
        return SCORE_WRONG, f"wrong tool: got '{pred_tool}', expected '{expected_tool}'"

    pred_args = pred_call.get("args", {})
    if args_match(pred_args, expected_args):
        return SCORE_EXACT, "exact match"
    else:
        return SCORE_CORRECT_TOOL_BAD_ARGS, f"correct tool, wrong args: got {pred_args}, expected {expected_args}"


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_test_file(path: Path) -> list[dict]:
    """Load test JSONL. Each line: {messages, expected_tool?, expected_args?, slice?}"""
    examples = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            examples.append(json.loads(line))
    return examples


def extract_last_user_prompt(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split messages into (last user prompt, history)."""
    history = []
    prompt = ""
    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            continue
        if i == len(messages) - 1 and msg["role"] == "user":
            prompt = msg["content"]
        else:
            if msg["role"] in ("user", "assistant"):
                history.append({"role": msg["role"], "content": msg["content"]})
    return prompt, history


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(test_path: Path, verbose: bool = False, latency: bool = True) -> dict:
    from inference import run

    examples = load_test_file(test_path)
    print(f"Loaded {len(examples)} test examples from {test_path}")

    results = []
    slice_scores: dict[str, list[float]] = {}
    latencies = []

    for idx, ex in enumerate(examples):
        messages = ex.get("messages", [])
        expected_tool = ex.get("expected_tool")
        expected_args = ex.get("expected_args", {}) or {}
        slice_name = ex.get("slice", "unknown")

        prompt, history = extract_last_user_prompt(messages)
        if not prompt:
            # Fallback: use last message
            for m in reversed(messages):
                if m["role"] == "user":
                    prompt = m["content"]
                    break

        t0 = time.perf_counter()
        predicted = run(prompt, history)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        score, reason = score_example(predicted, {"tool": expected_tool, "args": expected_args})

        results.append({
            "idx": idx,
            "slice": slice_name,
            "prompt": prompt,
            "expected_tool": expected_tool,
            "expected_args": expected_args,
            "predicted": predicted,
            "score": score,
            "reason": reason,
            "latency_ms": elapsed_ms,
        })

        slice_scores.setdefault(slice_name, []).append(score)

        if verbose:
            status = "✓" if score == SCORE_EXACT else ("⚠" if score > 0 else "✗")
            print(f"[{idx:02d}] {status} {score:+.1f} | {slice_name:20s} | {reason}")
            print(f"       Prompt    : {prompt[:80]}")
            print(f"       Predicted : {predicted[:100]}")
            print()

    # Summary
    total_score = sum(r["score"] for r in results)
    max_score = len(results) * SCORE_EXACT
    mean_latency = sum(latencies) / len(latencies) if latencies else 0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total score : {total_score:.1f} / {max_score:.1f} ({total_score/max_score*100:.1f}%)")
    print(f"Mean latency: {mean_latency:.1f} ms/turn")
    if mean_latency <= 200:
        print(f"  ✓ Passes ≤200 ms latency gate")
    else:
        print(f"  ✗ Exceeds 200 ms latency gate (recheck on target hardware)")

    print("\nPer-slice scores:")
    for slice_name, scores in sorted(slice_scores.items()):
        slice_total = sum(scores)
        slice_max = len(scores)
        print(f"  {slice_name:20s}: {slice_total:.1f}/{slice_max:.1f} ({slice_total/slice_max*100:.1f}%)")

    print("\nScore distribution:")
    for label, val in [("Exact (+1.0)", SCORE_EXACT), ("Partial (+0.5)", SCORE_CORRECT_TOOL_BAD_ARGS),
                        ("Wrong (0.0)", SCORE_WRONG), ("False call (-0.5)", SCORE_FALSE_TOOL_CALL)]:
        cnt = sum(1 for r in results if r["score"] == val)
        print(f"  {label:20s}: {cnt}")

    # Failure analysis
    failures = [r for r in results if r["score"] < SCORE_EXACT]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures[:10]:
            print(f"  [{r['idx']:02d}] {r['score']:+.1f} {r['reason']}")
            print(f"       prompt: {r['prompt'][:70]}")

    summary = {
        "total_score": total_score,
        "max_score": max_score,
        "pct": total_score / max_score * 100,
        "mean_latency_ms": mean_latency,
        "per_slice": {k: sum(v) / len(v) for k, v in slice_scores.items()},
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="data/public_test.jsonl")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--out", help="Write results JSON to this file")
    args = parser.parse_args()

    test_path = ROOT / args.test_file
    if not test_path.exists():
        sys.exit(f"Test file not found: {test_path}")

    summary = evaluate(test_path, verbose=args.verbose)

    if args.out:
        out_path = ROOT / args.out
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
