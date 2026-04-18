"""
Pocket-Agent sanity check.

Two modes:
  python sanity_check.py --local      # No GPU needed. Runs locally right now.
  python sanity_check.py --colab      # On Colab T4. Smoke-tests full pipeline
                                      # in ~5 min before committing to full run.

Exit code 0 = all checks passed. Non-zero = something will break during training.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

errors: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def fail(msg: str) -> None:
    print(f"{FAIL} {msg}")
    errors.append(msg)


def warn(msg: str) -> None:
    print(f"{WARN} {msg}")
    warnings.append(msg)


def section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── Local checks (no GPU, no model download) ──────────────────────────────────

def check_files_exist():
    section("1. Required files")
    required = [
        "inference.py",
        "demo.py",
        "quantize.py",
        "Makefile",
        "requirements.txt",
        "requirements-colab.txt",
        "colab_train.ipynb",
        "data/tool_schemas.json",
        "data/generate_data.py",
        "train/finetune.py",
        "eval/evaluate.py",
    ]
    for f in required:
        p = ROOT / f
        if p.exists():
            ok(f"{f} ({p.stat().st_size // 1024} KB)")
        else:
            fail(f"MISSING: {f}")


def check_no_network_imports():
    section("2. inference.py — no network imports (AST scan)")
    src = (ROOT / "inference.py").read_text()
    tree = ast.parse(src)
    banned = {"requests", "urllib", "http", "socket", "httpx", "aiohttp", "httplib2", "boto3"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in banned:
                    found.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in banned:
                    found.add(top)
    if found:
        fail(f"Network imports found in inference.py: {found}")
    else:
        ok("No network imports found")

    # Also check all top-level imports are printed for manual review
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    ok(f"All imports: {sorted(set(imports))}")


def check_system_prompt_consistency():
    section("3. SYSTEM_PROMPT consistency (generate_data.py vs inference.py)")
    gen_src = (ROOT / "data" / "generate_data.py").read_text()
    inf_src = (ROOT / "inference.py").read_text()

    def extract_prompt(src: str) -> str:
        m = re.search(r'SYSTEM_PROMPT\s*=\s*"""\\?\n(.*?)"""', src, re.DOTALL)
        return m.group(1).strip() if m else ""

    gen_prompt = extract_prompt(gen_src)
    inf_prompt = extract_prompt(inf_src)

    if not gen_prompt:
        fail("Could not extract SYSTEM_PROMPT from generate_data.py")
    elif not inf_prompt:
        fail("Could not extract SYSTEM_PROMPT from inference.py")
    elif gen_prompt == inf_prompt:
        ok("SYSTEM_PROMPT is identical in both files")
    else:
        # Find first diff
        g_lines = gen_prompt.splitlines()
        i_lines = inf_prompt.splitlines()
        for idx, (gl, il) in enumerate(zip(g_lines, i_lines)):
            if gl != il:
                fail(f"SYSTEM_PROMPT mismatch at line {idx+1}:")
                print(f"    generate_data.py: {gl!r}")
                print(f"    inference.py    : {il!r}")
                break
        else:
            if len(g_lines) != len(i_lines):
                fail(f"SYSTEM_PROMPT length differs: {len(g_lines)} vs {len(i_lines)} lines")


def check_tool_schemas():
    section("4. tool_schemas.json validity")
    path = ROOT / "data" / "tool_schemas.json"
    try:
        schemas = json.loads(path.read_text())
        tools = schemas.get("tools", [])
        names = [t["name"] for t in tools]
        expected = {"weather", "calendar", "convert", "currency", "sql"}
        if set(names) == expected:
            ok(f"All 5 tools present: {sorted(names)}")
        else:
            fail(f"Tool names mismatch. Got: {names}, expected: {sorted(expected)}")
        for tool in tools:
            if "args" in tool:
                ok(f"  {tool['name']}: {list(tool['args'].keys())}")
    except json.JSONDecodeError as e:
        fail(f"tool_schemas.json is invalid JSON: {e}")


def check_data_generation():
    section("5. Data generation (rule-based, no API)")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        out_path = tf.name

    result = subprocess.run(
        [sys.executable, str(ROOT / "data" / "generate_data.py"),
         "--no-api", "--out", out_path, "--seed", "42"],
        capture_output=True, text=True, cwd=str(ROOT),
    )

    if result.returncode != 0:
        fail(f"generate_data.py failed:\n{result.stderr[-500:]}")
        return

    lines = [l for l in Path(out_path).read_text().splitlines() if l.strip()]
    if not lines:
        fail("generate_data.py produced 0 examples")
        return

    ok(f"Generated {len(lines)} examples")

    # Parse and validate structure
    malformed = 0
    tool_counts: dict[str, int] = {}
    refusals = 0
    valid_tools = {"weather", "calendar", "convert", "currency", "sql"}
    city_mismatches = 0
    unit_errors = 0

    for line in lines:
        ex = json.loads(line)
        messages = ex.get("messages", [])

        # Check structure
        roles = [m["role"] for m in messages]
        if roles[0] != "system":
            malformed += 1
            continue
        if roles[-1] != "assistant":
            malformed += 1
            continue

        # Find assistant content
        asst = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")

        if "<tool_call>" in asst:
            m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", asst, re.DOTALL)
            if not m:
                malformed += 1
                continue
            try:
                parsed = json.loads(m.group(1))
                tool = parsed.get("tool", "")
                if tool not in valid_tools:
                    malformed += 1
                    continue
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

                # Check city/unit consistency for weather
                if tool == "weather":
                    location = parsed["args"].get("location", "")
                    unit = parsed["args"].get("unit", "")
                    if location.lower() not in user.lower():
                        city_mismatches += 1
                    prompt_lower = user.lower()
                    if "celsius" in prompt_lower and unit != "C":
                        unit_errors += 1
                    if "fahrenheit" in prompt_lower and unit != "F":
                        unit_errors += 1
            except json.JSONDecodeError:
                malformed += 1
        else:
            refusals += 1

    if malformed:
        fail(f"{malformed} malformed examples")
    else:
        ok("0 malformed examples")

    if city_mismatches:
        fail(f"{city_mismatches} weather examples: city in prompt ≠ city in args (BUG!)")
    else:
        ok("Weather city consistency: PASS")

    if unit_errors:
        fail(f"{unit_errors} weather examples: explicit unit in prompt ≠ unit in args (BUG!)")
    else:
        ok("Weather unit consistency: PASS")

    print(f"\n  Distribution:")
    for tool, cnt in sorted(tool_counts.items()):
        print(f"    {tool:12s}: {cnt}")
    print(f"    {'refusals':12s}: {refusals}")

    Path(out_path).unlink(missing_ok=True)


def check_eval_scoring_logic():
    section("6. Evaluation scoring logic")
    sys.path.insert(0, str(ROOT))

    # Temporarily patch run() so we can test eval without a model
    import eval.evaluate as ev

    test_cases = [
        # (predicted_text, expected_tool, expected_args, expected_score)
        (
            '<tool_call>\n{"tool": "weather", "args": {"location": "Paris", "unit": "C"}}\n</tool_call>',
            "weather", {"location": "Paris", "unit": "C"}, 1.0,
        ),
        (
            '<tool_call>\n{"tool": "weather", "args": {"location": "Paris", "unit": "F"}}\n</tool_call>',
            "weather", {"location": "Paris", "unit": "C"}, 0.5,
        ),
        (
            '<tool_call>\n{"tool": "convert", "args": {"value": 5, "from_unit": "km", "to_unit": "miles"}}\n</tool_call>',
            "convert", {"value": 5.0, "from_unit": "km", "to_unit": "miles"}, 1.0,
        ),
        (
            "I don't have a weather tool for that.",
            "weather", {"location": "Paris", "unit": "C"}, 0.0,
        ),
        (
            "I can't help with that.",
            None, None, 1.0,  # correct refusal
        ),
        (
            '<tool_call>\n{"tool": "weather", "args": {"location": "Tokyo", "unit": "C"}}\n</tool_call>',
            None, None, -0.5,  # false tool call
        ),
        # Numerical ±1% tolerance check
        (
            '<tool_call>\n{"tool": "currency", "args": {"amount": 100.5, "from": "USD", "to": "EUR"}}\n</tool_call>',
            "currency", {"amount": 100.0, "from": "USD", "to": "EUR"}, 1.0,  # 0.5% diff → pass
        ),
        (
            '<tool_call>\n{"tool": "currency", "args": {"amount": 102.0, "from": "USD", "to": "EUR"}}\n</tool_call>',
            "currency", {"amount": 100.0, "from": "USD", "to": "EUR"}, 0.5,  # 2% diff → fail
        ),
    ]

    all_pass = True
    for predicted, exp_tool, exp_args, expected_score in test_cases:
        score, reason = ev.score_example(predicted, {"tool": exp_tool, "args": exp_args})
        if abs(score - expected_score) < 0.01:
            ok(f"score={score:+.1f} ({reason[:50]})")
        else:
            fail(f"Expected score {expected_score:+.1f}, got {score:+.1f} — {reason}")
            all_pass = False

    if all_pass:
        ok("All scoring cases correct")


def check_inference_importable():
    section("7. inference.py — importable without a model")
    try:
        # We just check it parses and defines run()
        src = (ROOT / "inference.py").read_text()
        tree = ast.parse(src)
        func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if "run" in func_names:
            ok("run() function defined")
        else:
            fail("run() function NOT found in inference.py")

        # Check signature
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                args = [a.arg for a in node.args.args]
                if "prompt" in args and "history" in args:
                    ok(f"run() signature correct: {args}")
                else:
                    fail(f"run() signature wrong: {args} (need prompt, history)")
    except SyntaxError as e:
        fail(f"inference.py has syntax error: {e}")


def check_colab_notebook():
    section("8. colab_train.ipynb — valid JSON")
    nb_path = ROOT / "colab_train.ipynb"
    try:
        nb = json.loads(nb_path.read_text())
        cells = nb.get("cells", [])
        code_cells = [c for c in cells if c["cell_type"] == "code"]
        ok(f"Valid notebook: {len(cells)} cells ({len(code_cells)} code cells)")
    except json.JSONDecodeError as e:
        fail(f"colab_train.ipynb invalid JSON: {e}")


def check_makefile():
    section("9. Makefile targets")
    makefile = (ROOT / "Makefile").read_text()
    required_targets = ["all", "data", "train", "quantize", "eval", "demo"]
    for target in required_targets:
        if re.search(rf"^{target}:", makefile, re.MULTILINE):
            ok(f"  target '{target}' present")
        else:
            fail(f"  target '{target}' MISSING from Makefile")


# ── Colab smoke test (GPU required, ~5 minutes) ───────────────────────────────

def colab_smoke_test():
    section("COLAB SMOKE TEST — full pipeline, 10 steps")
    print("  This validates the entire pipeline before the full 45-min run.\n")

    # 1. Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            ok(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        else:
            warn("No GPU detected — training will be very slow")
    except ImportError:
        fail("torch not installed — run: pip install torch")
        return

    # 2. Check Unsloth
    try:
        from unsloth import FastLanguageModel
        ok("unsloth imported")
    except ImportError:
        fail("unsloth not installed — run: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        return

    # 3. Check TRL + datasets
    try:
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        ok("trl + datasets imported")
    except ImportError as e:
        fail(f"Missing training dep: {e}")
        return

    # 4. Generate tiny dataset
    print("\n  Generating 20-example smoke dataset...")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tf:
        smoke_data_path = tf.name
        # 20 hardcoded examples — enough to test the trainer
        examples = [
            {"messages": [
                {"role": "system", "content": "You are a tool-calling assistant. Use <tool_call> JSON for tool requests."},
                {"role": "user", "content": f"Weather in London in Celsius?"},
                {"role": "assistant", "content": '<tool_call>\n{"tool": "weather", "args": {"location": "London", "unit": "C"}}\n</tool_call>'},
            ]},
            {"messages": [
                {"role": "system", "content": "You are a tool-calling assistant. Use <tool_call> JSON for tool requests."},
                {"role": "user", "content": "Convert 100 USD to EUR."},
                {"role": "assistant", "content": '<tool_call>\n{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "EUR"}}\n</tool_call>'},
            ]},
            {"messages": [
                {"role": "system", "content": "You are a tool-calling assistant. Use <tool_call> JSON for tool requests."},
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "I'm a task-focused assistant — jokes aren't my specialty!"},
            ]},
        ] * 7  # repeat to get 21 examples
        for ex in examples[:20]:
            tf.write(json.dumps(ex) + "\n")
    ok(f"Smoke dataset: {smoke_data_path}")

    # 5. Load model (4-bit)
    print("\n  Loading Qwen3-0.6B-Instruct (4-bit)...")
    t0 = time.time()
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-0.6B-Instruct",
            max_seq_length=512,
            load_in_4bit=True,
            dtype=None,
        )
        ok(f"Model loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        fail(f"Model load failed: {e}")
        return

    # 6. Apply LoRA
    try:
        model = FastLanguageModel.get_peft_model(
            model, r=8, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", use_gradient_checkpointing="unsloth", random_state=42,
        )
        ok(f"LoRA applied: {model.num_parameters(only_trainable=True):,} trainable params")
    except Exception as e:
        fail(f"LoRA setup failed: {e}")
        return

    # 7. Run 10 training steps
    print("\n  Training 10 steps (smoke test)...")
    try:
        from datasets import Dataset as HFDataset

        records = [json.loads(l) for l in Path(smoke_data_path).read_text().splitlines() if l.strip()]
        dataset = HFDataset.from_list(records)

        def fmt(ex):
            return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

        dataset = dataset.map(fmt, remove_columns=dataset.column_names)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir="/tmp/smoke_ckpt",
                max_steps=10,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=5,
                dataset_text_field="text",
                max_seq_length=512,
                report_to="none",
                seed=42,
            ),
        )
        t0 = time.time()
        stats = trainer.train()
        elapsed = time.time() - t0
        ok(f"10 steps completed in {elapsed:.1f}s, final loss: {stats.metrics.get('train_loss', '?'):.4f}")
        per_step = elapsed / 10
        eta_500 = per_step * 500
        ok(f"Projected full run (500 steps): {eta_500/60:.0f} min")
    except Exception as e:
        fail(f"Training failed: {e}")
        import traceback; traceback.print_exc()
        return

    # 8. Save adapter
    smoke_adapter_path = "/tmp/smoke_adapter"
    try:
        model.save_pretrained(smoke_adapter_path)
        tokenizer.save_pretrained(smoke_adapter_path)
        ok(f"Adapter saved to {smoke_adapter_path}")
    except Exception as e:
        fail(f"Adapter save failed: {e}")
        return

    # 9. Export GGUF (q8_0 is fastest, f16 skips quantization entirely)
    print("\n  Exporting to GGUF (q8_0 — fastest export for smoke test)...")
    smoke_gguf_dir = "/tmp/smoke_gguf"
    Path(smoke_gguf_dir).mkdir(exist_ok=True)
    try:
        t0 = time.time()
        model.save_pretrained_gguf(smoke_gguf_dir, tokenizer, quantization_method="q8_0")
        elapsed = time.time() - t0
        gguf_files = list(Path(smoke_gguf_dir).glob("*.gguf"))
        if gguf_files:
            size_mb = gguf_files[0].stat().st_size / 1e6
            ok(f"GGUF exported: {gguf_files[0].name} ({size_mb:.0f} MB) in {elapsed:.0f}s")
            ok(f"Projected q4_k_m export time: ~{elapsed*0.6:.0f}s (q4_k_m is smaller/faster)")
        else:
            fail("No GGUF file produced")
            return
    except Exception as e:
        fail(f"GGUF export failed: {e}")
        import traceback; traceback.print_exc()
        return

    # 10. Load GGUF with llama-cpp-python and run inference
    print("\n  Loading GGUF with llama-cpp-python...")
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=str(gguf_files[0]),
            n_ctx=512, n_threads=2, n_gpu_layers=0, verbose=False, chat_format="chatml",
        )
        ok("llama-cpp-python loaded GGUF")

        # Test inference
        from inference import SYSTEM_PROMPT
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Weather in Tokyo in Celsius?"},
            ],
            max_tokens=100, temperature=0.0,
        )
        output = result["choices"][0]["message"]["content"]
        ok(f"Inference output: {output[:120]}")

        if "<tool_call>" in output:
            ok("Model produced a <tool_call> block ✓")
        else:
            warn("Model didn't produce <tool_call> — expected after only 10 steps (normal for smoke test)")

    except ImportError:
        fail("llama-cpp-python not installed — run: pip install llama-cpp-python")
        return
    except Exception as e:
        fail(f"Inference failed: {e}")
        return

    # 11. Test inference.py run() interface with a monkey-patched model
    print("\n  Testing inference.run() interface...")
    try:
        os.environ["LLAMA_N_THREADS"] = "2"
        # Patch model path temporarily
        import inference as inf_module
        inf_module._MODEL_PATH = Path(str(gguf_files[0]))
        inf_module._model = None  # force reload

        resp = inf_module.run("Weather in London?", [])
        ok(f"inference.run() returned: {resp[:100]}")
        resp2 = inf_module.run("Convert that to JPY.", [
            {"role": "user", "content": "What's 100 USD in EUR?"},
            {"role": "assistant", "content": '<tool_call>\n{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "EUR"}}\n</tool_call>'},
        ])
        ok(f"Multi-turn run() returned: {resp2[:100]}")
    except Exception as e:
        fail(f"inference.run() failed: {e}")
        return

    # 12. Test evaluate.py with 3 mock examples
    print("\n  Testing evaluate.py scoring...")
    mock_test_path = Path("/tmp/smoke_test.jsonl")
    mock_examples = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Weather in Tokyo?"},
            ],
            "expected_tool": "weather",
            "expected_args": {"location": "Tokyo", "unit": "C"},
            "slice": "smoke",
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Tell me a joke."},
            ],
            "expected_tool": None,
            "expected_args": None,
            "slice": "smoke",
        },
    ]
    mock_test_path.write_text("\n".join(json.dumps(e) for e in mock_examples))
    try:
        import eval.evaluate as ev
        summary = ev.evaluate(mock_test_path, verbose=True, latency=True)
        ok(f"evaluate.py ran: score={summary['total_score']:.1f}/{summary['max_score']:.1f}, latency={summary['mean_latency_ms']:.0f}ms")
        if summary["mean_latency_ms"] <= 200:
            ok(f"Latency PASSES ≤200ms gate: {summary['mean_latency_ms']:.0f}ms")
        else:
            warn(f"Latency {summary['mean_latency_ms']:.0f}ms > 200ms — may fail hard gate. "
                 "This is expected for smoke model; real model may differ.")
    except Exception as e:
        fail(f"evaluate.py failed: {e}")

    # 13. Check demo imports (don't launch)
    print("\n  Checking demo.py imports...")
    try:
        import gradio as gr
        ok(f"gradio {gr.__version__} available")
    except ImportError:
        fail("gradio not installed — run: pip install gradio")

    # Cleanup
    Path(smoke_data_path).unlink(missing_ok=True)
    mock_test_path.unlink(missing_ok=True)


# ── Report ────────────────────────────────────────────────────────────────────

def report():
    print(f"\n{'='*55}")
    print("  SANITY CHECK COMPLETE")
    print(f"{'='*55}")
    if errors:
        print(f"\n  ✗ {len(errors)} FAILURE(S):")
        for e in errors:
            print(f"    • {e}")
    if warnings:
        print(f"\n  ⚠  {len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"    • {w}")
    if not errors:
        print("\n  ✓ All checks passed — safe to run full training!\n")
        return 0
    else:
        print("\n  ✗ Fix the failures above before starting training.\n")
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run local checks only (no GPU)")
    parser.add_argument("--colab", action="store_true", help="Run full Colab smoke test (GPU required)")
    args = parser.parse_args()

    if not args.local and not args.colab:
        parser.print_help()
        print("\nRun with --local for pre-flight checks (no GPU needed).")
        print("Run with --colab on Colab T4 to smoke-test the full pipeline.")
        sys.exit(0)

    print("=" * 55)
    print("  POCKET-AGENT SANITY CHECK")
    print("=" * 55)

    # Local checks always run
    check_files_exist()
    check_no_network_imports()
    check_system_prompt_consistency()
    check_tool_schemas()
    check_data_generation()
    check_eval_scoring_logic()
    check_inference_importable()
    check_colab_notebook()
    check_makefile()

    if args.colab:
        colab_smoke_test()

    sys.exit(report())


if __name__ == "__main__":
    main()
