"""
Export the fine-tuned LoRA adapter to GGUF Q4_K_M.
Merges adapter weights into the base model, then quantizes via Unsloth.

Run on the same Colab T4 session after train/finetune.py completes.

Usage:
    python quantize.py [--adapter artifacts/adapter] [--out artifacts/model-q4km.gguf] [--quant q4_k_m]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="artifacts/adapter")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--quant", default="q4_k_m",
                   choices=["q4_k_m", "q3_k_m", "q2_k", "q8_0", "f16"],
                   help="Quantization method. q4_k_m → ~397 MB. q3_k_m → ~347 MB (bonus tier).")
    return p.parse_args()


def get_base_model(adapter_path: Path) -> str:
    meta_path = adapter_path / "training_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta.get("base_model", "unsloth/Qwen3-0.6B")
    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        return cfg.get("base_model_name_or_path", "unsloth/Qwen3-0.6B")
    return "unsloth/Qwen3-0.6B"


def main():
    args = parse_args()
    adapter_path = ROOT / args.adapter
    out_path = ROOT / args.out
    out_path.mkdir(parents=True, exist_ok=True)

    if not adapter_path.exists():
        sys.exit(f"Adapter not found: {adapter_path}\nRun: python train/finetune.py")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        sys.exit("Install dependencies: pip install 'unsloth[colab-new]'")

    base_model = get_base_model(adapter_path)
    print(f"=== Pocket-Agent GGUF Export ===")
    print(f"Adapter    : {adapter_path}")
    print(f"Base model : {base_model}")
    print(f"Quant      : {args.quant}")
    print(f"Output dir : {out_path}")

    # Load base + adapter merged
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )

    # Export to GGUF — merges LoRA into base weights then quantizes
    print(f"\nMerging LoRA and exporting GGUF ({args.quant})...")
    model.save_pretrained_gguf(
        str(out_path),
        tokenizer,
        quantization_method=args.quant,
    )

    # Find and report the exported file
    gguf_files = list(out_path.glob("*.gguf"))
    if gguf_files:
        gguf_file = gguf_files[0]
        size_mb = gguf_file.stat().st_size / (1024 * 1024)
        print(f"\n✓ GGUF exported: {gguf_file.name}")
        print(f"  Size: {size_mb:.1f} MB")

        if size_mb <= 250:
            print("  ✓ Qualifies for ≤250 MB bonus!")
        elif size_mb <= 500:
            print("  ✓ Passes ≤500 MB hard gate")
        else:
            print(f"  ✗ WARNING: {size_mb:.1f} MB exceeds 500 MB hard gate!")
            print("  Re-run with --quant q3_k_m or q2_k to reduce size")

        # Write gate check result
        gate_result = {
            "gguf_file": str(gguf_file),
            "size_mb": round(size_mb, 1),
            "passes_500mb_gate": size_mb <= 500,
            "qualifies_250mb_bonus": size_mb <= 250,
            "quantization": args.quant,
        }
        (out_path / "gate_check.json").write_text(json.dumps(gate_result, indent=2))
    else:
        print("  WARNING: No .gguf file found in output directory")


if __name__ == "__main__":
    main()
