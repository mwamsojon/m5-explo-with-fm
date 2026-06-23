"""
Demo: PeftModel (LoRA) + Chronos-2 bug and fix.

ROOT CAUSE
----------
PEFT's PeftModelForFeatureExtraction.forward() follows the standard HuggingFace
convention and injects 'input_ids' into the keyword arguments before calling the
base model.  Chronos-2's forward() signature uses a completely different interface
(context, context_mask, group_ids, ...) — it does NOT accept 'input_ids'.

Result: wrapping a Chronos-2 model with get_peft_model() and running inference
through the Chronos pipeline raises:
    TypeError: Chronos2Model.forward() got an unexpected keyword argument 'input_ids'

The adapters are stored in memory and appear in named_parameters(), so training
succeeds if you call model() with the right kwargs directly — but as soon as the
Chronos pipeline calls pl.model(context=..., ...) through its normal inference
path the PeftModel wrapper injects 'input_ids' and crashes.

FIX
---
Call peft_model.merge_and_unload() after training.  This folds the LoRA delta
weights (lora_B @ lora_A) into the base Linear weights and returns a plain
Chronos2Model with no PEFT wrapper.  The pipeline then calls Chronos2Model.forward()
directly, the adapter contribution is permanently baked in, and inference works.

Run:
    python experiments/demo_peft_bug.py
"""

import traceback
import torch
from peft import LoraConfig, get_peft_model, TaskType

print("=" * 65)
print("Loading autogluon/chronos-2-small …")
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "autogluon/chronos-2-small",
    device_map="cpu",
    dtype=torch.float32,
)

# ── Shared helpers ────────────────────────────────────────────────────────────
rng = torch.Generator().manual_seed(42)
context = torch.rand(1, 64, generator=rng)

def run_pipeline(pl, ctx):
    with torch.no_grad():
        out = pl.model(
            context=ctx,
            group_ids=torch.zeros(ctx.shape[0], dtype=torch.long),
            num_output_patches=1,
        )
    return out.quantile_preds.detach().clone()

# ── 1. Base model reference ───────────────────────────────────────────────────
base_preds = run_pipeline(pipeline, context)
print(f"\n[BASE]  median forecast (first 6 steps): "
      f"{base_preds[0, 6, :6].numpy().round(4)}")

# ── 2. Apply LoRA and simulate post-training adapter weights ──────────────────
print("\n" + "=" * 65)
print("Applying LoRA (r=4, targeting q/v projections) …")

base_model = pipeline.model          # save reference before wrapping

lora_cfg = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q", "v"],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)
peft_model = get_peft_model(base_model, lora_cfg)
peft_model.print_trainable_parameters()

# Simulate training: set lora_B to non-zero so adapters have a measurable effect
n_adapted = 0
for name, param in peft_model.named_parameters():
    if "lora_B" in name:
        param.data.fill_(0.5)
        n_adapted += 1
print(f"  → {n_adapted} lora_B tensors set to 0.5 (simulate trained adapter)")

# ── 3. BUG: run inference through the PeftModel-wrapped pipeline ──────────────
print("\n" + "=" * 65)
print("BUG: inference with PeftModel wrapper …")

pipeline.model = peft_model
bug_triggered = False
try:
    peft_preds = run_pipeline(pipeline, context)
    peft_eq_base = torch.allclose(base_preds, peft_preds, atol=1e-5)
    print(f"  Inference succeeded (outputs match base: {peft_eq_base})")
    if peft_eq_base:
        print("  *** BUG (silent): adapter weights present but outputs unchanged ***")
except TypeError as e:
    bug_triggered = True
    print(f"  *** BUG (crash): {e} ***")
    print()
    print("  Why this happens:")
    print("    PeftModelForFeatureExtraction.forward() follows the standard HF")
    print("    convention and injects 'input_ids' before calling the base model.")
    print("    Chronos2Model.forward() expects (context, context_mask, group_ids,")
    print("    ...) and has no 'input_ids' parameter — the call crashes.")
    print()
    print("  The adapters ARE stored in memory (named_parameters() lists them),")
    print("  but the Chronos pipeline can never reach them through its normal")
    print("  inference path because the PEFT wrapper crashes first.")

# ── 4. FIX: merge_and_unload() ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("FIX: calling merge_and_unload() …")
print("  → folds lora_B @ lora_A delta into base Linear weights")
print("  → returns plain Chronos2Model (no PeftModel wrapper)")

merged_model   = peft_model.merge_and_unload()
pipeline.model = merged_model

merged_preds = run_pipeline(pipeline, context)
print(f"\n[MERGED] median forecast (first 6 steps): "
      f"{merged_preds[0, 6, :6].numpy().round(4)}")

merged_eq_base = torch.allclose(base_preds, merged_preds, atol=1e-5)
max_delta      = (merged_preds - base_preds).abs().max().item()
print(f"\n  Merged == base model: {merged_eq_base}   (max |Δ| = {max_delta:.6f})")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
if bug_triggered and not merged_eq_base:
    print("  BUG  : PeftModel wrapper crashes Chronos-2 inference (TypeError)")
    print("  FIX  : merge_and_unload() removes wrapper, bakes adapter into weights")
    print("  CHECK: merged model produces different outputs from base — CONFIRMED")
    print()
    print("  Practical rule: always call merge_and_unload() after LoRA fine-tuning")
    print("  before handing the model back to BaseChronosPipeline.")
elif bug_triggered and merged_eq_base:
    print("  BUG confirmed (crash), but merged model == base — check lora_B setup.")
elif not bug_triggered:
    print("  No crash observed.  Check PEFT version or task_type setting.")
