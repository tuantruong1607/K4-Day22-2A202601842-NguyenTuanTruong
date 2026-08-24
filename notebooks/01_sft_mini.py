# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB1 — SFT-mini: Build the Lab 21 SFT checkpoint inline
#
# **Stack:** Unsloth + LoRA r=16 + bitsandbytes 4-bit base + 1k VN Alpaca, 1 epoch.
# Maps to deck §1 (why SFT alone insufficient — motivates the upcoming DPO step) +
# deck §3 (DPO will need this SFT checkpoint as initial policy).
#
# > **Mục tiêu:** tạo 1 SFT adapter "đủ tốt" để DPO có gì align lên. Đây là
# > Lab 21 thu gọn — nếu bạn đã hoàn thành Lab 21 sibling repo
# > ([VinUni-AI20k/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA)),
# > bạn có thể SKIP notebook này và copy adapter cũ vào `adapters/sft-mini/`.
# >
# > Nếu chưa, notebook này build từ đầu trong ~10 phút trên T4 (15 phút trên Colab CPU runtime — *đừng làm vậy*).

# %% [markdown]
# ## 0. Setup

# %%
import os
from pathlib import Path

# Tier detection. Defaults to T4 if env not set.
COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
assert COMPUTE_TIER in ("T4", "BIGGPU"), f"Invalid COMPUTE_TIER: {COMPUTE_TIER}"

# Tier-specific hyperparameters
if COMPUTE_TIER == "T4":
    BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
    MAX_LEN = 512
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 8
else:  # BIGGPU
    BASE_MODEL = "unsloth/Qwen2.5-7B-bnb-4bit"
    MAX_LEN = 1024
    PER_DEVICE_BATCH = 2
    GRAD_ACCUM = 4

SFT_DATASET = os.environ.get(
    "SFT_DATASET",
    "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated",
)
SFT_SLICE = int(os.environ.get("SFT_SLICE", "1000"))
SEED = int(os.environ.get("SEED", "42"))
NUM_EPOCHS = 1

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ADAPTER_OUT = REPO_ROOT / "adapters" / "sft-mini"
ADAPTER_OUT.mkdir(parents=True, exist_ok=True)

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"BASE_MODEL:      {BASE_MODEL}")
print(f"SFT_DATASET:     {SFT_DATASET}  (slice: {SFT_SLICE})")
print(f"SEED:            {SEED}")
print(f"max_seq_length:  {MAX_LEN}")
print(f"effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"output:          {ADAPTER_OUT}")

# %%
import torch

assert torch.cuda.is_available(), "DPO needs a CUDA GPU. See HARDWARE-GUIDE.md."
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}  ({gpu.total_memory / 1e9:.1f} GB)")

# %% [markdown]
# ## 1. Load base model with Unsloth
#
# Unsloth bundles patched 4-bit kernels — that's how Qwen2.5-3B (or 7B) stays
# in T4 / A100 budget. The `FastLanguageModel.from_pretrained` call returns a
# 4-bit quantized base; `get_peft_model` attaches the LoRA adapter on top.

# %%
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,                # auto: bf16 on Ampere+, fp16 on Turing
    load_in_4bit=True,
)

# Critical for batch training — Qwen tokenizers ship without pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer.pad_token = eos_token")

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,           # Unsloth supports dropout=0 for free speed
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # 30% VRAM savings
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# ## 2. Load + audit + format VN Alpaca slice
#
# The selected dataset has bilingual column names. Normalize the Vietnamese
# instruction/input/output fields before applying Qwen's chat template.
# %%
from datasets import load_dataset

raw_ds = load_dataset(SFT_DATASET, split="train")
raw_count = len(raw_ds)
required_columns = {"instruction_vi", "input_vi", "output_vi"}
missing_columns = required_columns.difference(raw_ds.column_names)
assert not missing_columns, (
    f"SFT dataset is missing expected Vietnamese columns: {sorted(missing_columns)}. "
    f"Available columns: {raw_ds.column_names}"
)

raw_ds = raw_ds.filter(
    lambda row: bool((row.get("instruction_vi") or "").strip())
    and bool((row.get("output_vi") or "").strip())
)
filtered_count = len(raw_ds)
raw_ds = raw_ds.shuffle(seed=SEED)
raw_ds = raw_ds.select(range(min(SFT_SLICE, len(raw_ds))))


def normalize_vietnamese_alpaca(row):
    return {
        "instruction": row["instruction_vi"].strip(),
        "input": (row.get("input_vi") or "").strip(),
        "output": row["output_vi"].strip(),
    }


ds = raw_ds.map(
    normalize_vietnamese_alpaca,
    remove_columns=raw_ds.column_names,
)

print(f"Loaded {raw_count} rows from {SFT_DATASET}")
print(f"Rows after instruction/output filter: {filtered_count}")
print(f"Selected examples: {len(ds)}  (seed={SEED})")
print(f"Columns after normalization: {ds.column_names}")

empty_input_pct = 100 * sum(not row["input"] for row in ds) / max(len(ds), 1)
instruction_lengths = [len(row["instruction"]) for row in ds]
output_lengths = [len(row["output"]) for row in ds]
duplicate_pairs = len(ds) - len({(row["instruction"], row["output"]) for row in ds})

import numpy as np


def describe_lengths(values):
    return {
        "min": int(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(np.max(values)),
    }


print(f"Empty input ratio: {empty_input_pct:.1f}%")
print(f"Instruction character lengths: {describe_lengths(instruction_lengths)}")
print(f"Output character lengths:      {describe_lengths(output_lengths)}")
print(f"Duplicate instruction/output pairs: {duplicate_pairs}")

print("\nQuality-audit samples (inspect at least 20 manually in the notebook output):")
for i in range(min(3, len(ds))):
    print(f"\n────── Audit sample {i + 1} ──────")
    print(f"INSTRUCTION: {ds[i]['instruction']}")
    print(f"INPUT:       {ds[i]['input'] or '[empty]'}")
    print(f"OUTPUT:      {ds[i]['output']}")
# %%
# Alpaca → ChatML format (Qwen2.5's native template)
def format_alpaca_to_chat(row):
    messages = []
    if row.get("instruction"):
        prompt = row["instruction"]
        if row.get("input"):
            prompt += "\n\n" + row["input"]
        messages.append({"role": "user", "content": prompt})
    if row.get("output"):
        messages.append({"role": "assistant", "content": row["output"]})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


ds_formatted = ds.map(format_alpaca_to_chat, remove_columns=ds.column_names)
print(f"\nSample formatted text (first 500 chars):\n{ds_formatted[0]['text'][:500]}")

# %% [markdown]
# ## 3. Train SFT-mini

# %%
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir=str(ADAPTER_OUT.parent / "sft-mini-checkpoints"),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",        # Save only at the end via trainer.model.save_pretrained
    optim="adamw_8bit",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    seed=42,
    max_length=MAX_LEN,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=ds_formatted,
)

# %%
train_result = trainer.train()
print(f"\nFinal train loss: {train_result.training_loss:.4f}")

# %% [markdown]
# ### 3a. Plot loss curve (deliverable: `02_sft_loss.png`)

# %%
import matplotlib.pyplot as plt

losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
steps = [log["step"] for log in trainer.state.log_history if "loss" in log]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, losses, marker="o", markersize=3, linewidth=1.2)
ax.set_xlabel("Training step")
ax.set_ylabel("Loss")
ax.set_title(f"SFT-mini loss · {COMPUTE_TIER} · {BASE_MODEL.split('/')[-1]} · {SFT_SLICE} samples")
ax.grid(True, alpha=0.3)
fig.tight_layout()
screenshot_dir = REPO_ROOT / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(screenshot_dir / "02-sft-loss.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Save adapter + sanity-check generation

# %%
trainer.model.save_pretrained(str(ADAPTER_OUT))
tokenizer.save_pretrained(str(ADAPTER_OUT))
print(f"Saved SFT adapter to {ADAPTER_OUT}")

# %%
# Sanity: generate 1 sample to confirm the adapter loaded correctly.
FastLanguageModel.for_inference(model)
prompt = "Giải thích ngắn gọn (3-4 câu) thuật toán quicksort hoạt động thế nào."
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to("cuda")
with torch.no_grad():
    out = model.generate(input_ids=inputs, max_new_tokens=200, do_sample=False)
generated = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"PROMPT: {prompt}\n")
print(f"SFT-mini response:\n{generated}")

# %% [markdown]
# ## 5. Vibe-coding callout
#
# Bạn vừa tái tạo Lab 21 trong ~10 phút. Một câu hỏi để brainstorm:
#
# > **Thật ra, bạn cần *bao nhiêu* SFT để DPO có ý nghĩa?**
# >
# > Thử thay `SFT_SLICE = 1000` → `100`. Re-run NB1 → NB3 với SFT yếu hơn.
# > Quan sát: reward gap có còn tăng được không? Output coherent không?
# >
# > Đây là 1 design decision *think-hard zone* (xem VIBE-CODING.md): không có
# > đáp án sẵn trong deck. Hypothesize trước, train sau, viết kết quả vào
# > `submission/REFLECTION.md` § 6.
#
# **Next:** NB2 — load + format preference data.
