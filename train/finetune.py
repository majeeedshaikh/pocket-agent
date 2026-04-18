"""
Unsloth QLoRA fine-tuning for Pocket-Agent on Qwen3-0.6B-Instruct.
Run this on a Colab T4 (16 GB VRAM). ~35-45 minutes for 500 steps.

Usage (Colab):
    !pip install -q "unsloth[colab-new]" trl datasets
    !python train/finetune.py --data data/training_data.jsonl --output artifacts/adapter
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/training_data.jsonl")
    p.add_argument("--output", default="artifacts/adapter")
    p.add_argument("--model", default="unsloth/Qwen3-0.6B")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(data_path: Path):
    from datasets import Dataset

    records = []
    for line in data_path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} training examples from {data_path}")
    return Dataset.from_list(records)


def main():
    args = parse_args()
    data_path = ROOT / args.data
    output_path = ROOT / args.output
    output_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        sys.exit(f"Training data not found: {data_path}\nRun: python data/generate_data.py")

    try:
        from unsloth import FastLanguageModel
        import torch
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        sys.exit("Install dependencies: pip install 'unsloth[colab-new]' trl datasets")

    print(f"=== Pocket-Agent Fine-tuning ===")
    print(f"Base model : {args.model}")
    print(f"Data       : {data_path} ({sum(1 for _ in data_path.open())} examples)")
    print(f"Output     : {output_path}")
    print(f"Max steps  : {args.max_steps}")
    print(f"LoRA r/α   : {args.lora_r}/{args.lora_alpha}")

    # Load base model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
    )

    print(f"\nTrainable parameters: {model.num_parameters(only_trainable=True):,}")
    print(f"Total parameters    : {model.num_parameters():,}")

    # Load and format dataset
    dataset = load_dataset(data_path)

    def format_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    print(f"Dataset formatted. Example:\n{dataset[0]['text'][:300]}...\n")

    # Training config
    training_args = SFTConfig(
        output_dir=str(output_path / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=min(50, args.max_steps // 10),
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        seed=args.seed,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer_stats = trainer.train()

    print(f"\n=== Training Complete ===")
    print(f"Runtime   : {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"Final loss: {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")

    # Save LoRA adapter
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"\nLoRA adapter saved to: {output_path}")

    # Save training metadata
    meta = {
        "base_model": args.model,
        "max_steps": args.max_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.lr,
        "train_runtime_s": trainer_stats.metrics["train_runtime"],
        "final_loss": trainer_stats.metrics.get("train_loss"),
    }
    (output_path / "training_meta.json").write_text(json.dumps(meta, indent=2))
    print("Training metadata saved.")


if __name__ == "__main__":
    main()
