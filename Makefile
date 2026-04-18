.PHONY: all data train quantize eval demo install install-colab clean

# Full pipeline: data → train → quantize → eval
all: data train quantize eval

# ── Dependencies ──────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-colab:
	pip install -r requirements-colab.txt

# ── Pipeline stages ───────────────────────────────────────────────────────────

data:
	python data/generate_data.py

data-no-api:
	python data/generate_data.py --no-api

train:
	python train/finetune.py

train-quick:
	python train/finetune.py --max-steps 200

quantize:
	python quantize.py

quantize-small:
	python quantize.py --quant q3_k_m

eval:
	python eval/evaluate.py --test-file data/public_test.jsonl --verbose

eval-quiet:
	python eval/evaluate.py --test-file data/public_test.jsonl

# ── Demo ──────────────────────────────────────────────────────────────────────

demo:
	python demo.py

demo-share:
	python demo.py --share

# ── Validation ────────────────────────────────────────────────────────────────

sanity-local:
	python sanity_check.py --local

sanity-colab:
	python sanity_check.py --colab

check-gates:
	@echo "=== Hard Gate Checks ==="
	@python -c "from pathlib import Path; f=list(Path('artifacts').glob('*.gguf')); \
		mb=f[0].stat().st_size/1024/1024 if f else 999; \
		print(f'  Model size: {mb:.1f} MB'); \
		print('  ✓ ≤500 MB' if mb<=500 else '  ✗ EXCEEDS 500 MB'); \
		print('  ✓ ≤250 MB bonus' if mb<=250 else f'  - Not ≤250 MB ({mb:.1f} MB)')"
	@python -c "import ast,sys; \
		src=open('inference.py').read(); \
		tree=ast.parse(src); \
		bad=['requests','urllib','http','socket','httpx','aiohttp']; \
		imports=[n.names[0].name.split('.')[0] if isinstance(n,ast.Import) else n.module.split('.')[0] if n.module else '' for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]; \
		found=[i for i in imports if i in bad]; \
		print('  ✓ No network imports' if not found else f'  ✗ Network imports found: {found}')"
	@echo "=== Done ==="

check-inference-imports:
	@python -c "import ast; \
		tree=ast.parse(open('inference.py').read()); \
		bad=['requests','urllib','http','socket','httpx','aiohttp','httplib']; \
		imports=[n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import)] + \
			[n.module.split('.')[0] if n.module else '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]; \
		found=[i for i in imports if i in bad]; \
		print('PASS: no network imports' if not found else f'FAIL: {found}')"

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf artifacts/checkpoints artifacts/adapter __pycache__ */__pycache__

clean-all: clean
	rm -rf artifacts/ data/training_data.jsonl
