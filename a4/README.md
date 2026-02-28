# Assignment 4: Pretrained Transformer Models and Knowledge Access

| Coding Project | Model | Date |
|----------------|-------|------|
| Pretrained Transformer models and knowledge access | Transformer, minGPT | 2-28-2026 |

This repository contains the implementation of a character-level Transformer for the birth-place prediction task: pretraining on Wikipedia (span corruption), finetuning on QA pairs, and evaluation with Vanilla GPT and RoPE variants.


---

## Task Overview

**Goal:** Given a question *"Where was [Person Name] born?"*, predict the birth place (e.g., Lebanon, Columbus, Glasgow).

The task is knowledge-intensive: the model needs to "know" facts about people that are not in the finetuning corpus. Pretraining on Wikipedia enables the model to store and later access such knowledge.

---

## Inputs and Outputs

### Data Files

| File | Description | Format |
|------|-------------|--------|
| `wiki.txt` | Wikipedia-style text for pretraining | Plain text (one document per line) |
| `birth_places_train.tsv` | Training data for the birth-place task | `Question\tBirthPlace` (e.g., `Where was Khatchig Mouradian born?\tLebanon`) |
| `birth_dev.tsv` | Development set (with labels) | Same as train |
| `birth_test_inputs.tsv` | Test set (no labels) | One question per line |

### Input Format (at inference)

- Input: `Where was [Name] born?⁇` (the `⁇` character marks where the model should generate the answer)
- Output: The predicted birth place string (e.g., `Lebanon`)

### Output Files

| Output | Description |
|--------|-------------|
| `*.params` | Saved model weights (e.g., `vanilla.pretrain.params`, `vanilla.finetune.params`) |
| `*.predictions` | One predicted birth place per line, aligned with the input file |

---

## Models

### 1. Vanilla GPT

- Standard decoder-only Transformer (minGPT-style)
- **Learned positional embeddings**
- Architecture: 4 layers, 8 heads, 256 embedding dim, block size 128
- ~3.3M parameters

### 2. RoPE GPT (Rotary Position Embeddings)

- Same architecture as Vanilla, but uses **Rotary Position Embeddings (RoPE)** instead of learned positional embeddings
- RoPE encodes position via sinusoidal functions; no separate position embedding parameters
- Often better for long sequences and transfer

---

## Training Pipeline

### 1. Pretraining (on Wikipedia)

- **Objective:** Span corruption (masked language modeling)
- Randomly mask spans in documents; model learns to reconstruct the masked content
- Builds general language and world knowledge

### 2. Finetuning (on birth places)

- **With pretraining:** Load pretrained weights, finetune on `birth_places_train.tsv`
- **Without pretraining:** Train from scratch on the birth-place task only (fails to generalize)

### 3. Evaluation

- Predict birth places for dev/test inputs
- Accuracy = % of correct predictions (exact string match)

---

## Usage

### Running in Google Colab (GPU T4)

Training takes ~2–3 hours. Use the [Colab setup notebook](CS_224N_2025_A4_Colab_Setup.ipynb) to upload files, mount Drive, and run the scripts. Ensure the runtime uses a **GPU** (e.g., T4) as the hardware accelerator.

From the `a4/` directory in Colab, run:

| Model | Script | Description |
|-------|--------|-------------|
| **Vanilla (with pretraining)** | `! bash scripts/run_vanilla.sh` | Pretrain → finetune → evaluate on dev & test |
| **Vanilla (without pretraining)** | `! bash scripts/run_vanilla_no_pretraining.sh` | Finetune from scratch → evaluate |
| **RoPE (with pretraining)** | `! bash scripts/run_rope.sh` | Pretrain → finetune → evaluate (RoPE model) |
| **London baseline** | `! python src/london_baseline.py` | Predict "London" for all; writes `london_baseline_accuracy.txt` |

Example (in a Colab cell):

```python
%cd /content/drive/MyDrive/ColabNotebooks/a4  # or your a4 path
! bash scripts/run_vanilla.sh
```

### Prerequisites

- **Local:** CPU mode, use `cs224n-cpu` env (same as a3)
- **GCP:** GPU mode, use `cs224n-gpu` env (same as a3)
- **Colab:** GPU mode, no need to install env; use the Colab setup notebook
---

## Project Structure

```
a4/
├── src/
│   ├── run.py          # Main entry: pretrain, finetune, evaluate
│   ├── models.py       # GPT and GPTConfig
│   ├── attention.py    # Causal self-attention + RoPE
│   ├── dataset.py     # CharCorruptionDataset, NameDataset
│   ├── trainer.py     # Training loop
│   └── utils.py       # Sampling, evaluation
├── mingpt-demo/       # Character-level GPT demo (play_char.ipynb)
├── scripts/           # Shell scripts for full pipelines
├── wiki.txt           # Pretraining corpus
├── birth_places_train.tsv
├── birth_dev.tsv
├── birth_test_inputs.tsv
└── collect_submission.sh
```

---

## Key Concepts

- **Character-level modeling:** Inputs are sequences of characters (not subwords). Vocabulary ~256 chars.
- **Span corruption:** Pretraining objective where random spans are masked and the model predicts them.
- **Knowledge access:** Pretraining on Wikipedia gives the model factual knowledge; finetuning teaches it to extract and generate that knowledge in the QA format.

---

## References

- [CS224N Assignment 4 Handout](handout.pdf)
- [minGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
