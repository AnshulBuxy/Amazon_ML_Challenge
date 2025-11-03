# Amazon_ML_Challenge
## MJAVE Price Regression (PyTorch)

This repository contains a PyTorch reimplementation of a MJAVE-style multimodal model for price regression (images + text). The codebase provides utilities to precompute embeddings (text and image), a "fast" training path that uses those embeddings, scripts to replace only text embeddings with your own DeBERTa weights, and multiple inference modes.

Summary
- Model: multimodal (text + image) MJAVE-style network implemented in `custom_multimodel/mjave_price_model.py`.
- Task: regression (predicting price). The model outputs log-price values during training/prediction.
- Best reported validation SMAPE achieved in this workspace: 45.2% (user-reported).

Quick structure
- `custom_multimodel/` - model implementation and helpers (e.g. `mjave_price_model.py`).
- `precompute_embeddings.py` - produce training embeddings (text + image) from raw data.
- `precompute_test_embeddings.py` - (rewritten) produce test embeddings using the same extractors.
- `replace_text_embeddings.py` - replace only the text embeddings in an existing embedding folder using a DeBERTa checkpoint (useful if you already have image embeddings).
- `train_mjave_price_fast.py` - fast training loop that loads precomputed embeddings from `./embeddings`.
- `inference_mjave_exact.py`, `test_mjave_inference.py`, `test_mjave_inference_live.py` - inference scripts (precomputed vs on-the-fly).
- `calculate_smape.py` - evaluation helper (note: handles log-space correctly).

Important invariants
- Text tokenization is run with MAX_LENGTH=256 which produces 254 content token vectors after removing special tokens (the code expects text feature tensors shaped like `[N, 254, 768]`).
- Image regional features: `[N, 49, 2048]`. Image global features: `[N, 2048]`.
- Model outputs and metrics expect log-space prices internally. `compute_smape()` expects log-space arrays (it runs expm1 internally). When evaluating against real prices saved to CSV, use `np.log1p(price)` before calling the training SMAPE routine.

Quick start (assumes Linux, bash)

1) Python environment and main packages

```bash
# create and activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate

# install main packages (add versions compatible with your CUDA/PyTorch setup)
pip install torch torchvision transformers pandas tqdm pillow numpy
```

If you maintain a `requirements.txt` file in this project, prefer `pip install -r requirements.txt`.

2) Paths and model weights

- DeBERTa weights used in development in this workspace (example):
  `/media/KB/Segformer_ML/student_resource/src/price_predictor_model_deberta_10epochs`
- Training expects embeddings to live under `./embeddings` (for fast train) and `./embeddings_test` for test.

3) Precompute embeddings (training)

This will extract text and image embeddings for training and save arrays under `./embeddings`.

```bash
python precompute_embeddings.py --csv path/to/train.csv --out_dir ./embeddings
```

4) Replace text embeddings only (useful when you have precomputed image embeddings and a new DeBERTa checkpoint)

```bash
python replace_text_embeddings.py --csv path/to/data.csv --emb_dir ./embeddings --deberta_path /path/to/deberta_checkpoint
```

Notes: the script tries to handle `price` being missing (test CSVs) and supports `image` or `image_link` columns, and `title` or `catalog_content` for text. If you hit a tokenizer `IndexError` while batch-encoding, see the troubleshooting below (sanitize or inspect the offending raw texts).

5) Precompute test embeddings (parity with training)

```bash
python precompute_test_embeddings.py --csv path/to/test.csv --out_dir ./embeddings_test
```

6) Fast training using precomputed embeddings

```bash
python train_mjave_price_fast.py --emb_dir ./embeddings --output_dir ./checkpoints
```

The fast training script reads the numpy arrays from the precompute step. Model checkpoints are saved under the configured output path (see the script for exact filenames: `best_mjave_price_model_fast.pth`, `checkpoint_latest_fast.pth`, ...).

7) Inference

- Precomputed-embeddings inference (fast):

```bash
python test_mjave_inference.py --emb_dir ./embeddings_test --checkpoint ./checkpoints/best_mjave_price_model_fast.pth --out predictions.csv
```

- Live (on-the-fly) inference:

```bash
python test_mjave_inference_live.py --csv path/to/test.csv --checkpoint ./checkpoints/best_mjave_price_model_fast.pth --out predictions.csv
```

8) Evaluate SMAPE (matching training's log-space expectation)

If your predictions are saved as actual price values (not log), convert them with `np.log1p(actual_price)` before using training's `compute_smape`. The provided `calculate_smape.py` is already adapted to re-log actual prices when reading a CSV.

Tips to improve SMAPE (practical suggestions)
- Data cleaning: remove or sanitize corrupted text fields, strip control characters and null bytes, normalize unicode (NFKC), and fill empty titles with a short placeholder like "unknown product".
- Tokenizer sanitization: if you encounter tokenizer internal IndexErrors, print the offending raw strings (the repo scripts include debugging hooks), then: limit length, replace invalid control characters, or remove NUL bytes.
- Regularization: add dropout, early stopping (already used in experiments), weight decay, and consider reducing attention size (the repo used ATTN_SIZE=200 for better generalization).
- Data augmentation: augment images (random crops, color jitter) and apply text augmentations (minor paraphrasing, stopword masking) where appropriate.
- Model variants: try different backbones (EfficientNet/ResNet variants), or freeze/fine-tune DeBERTa selectively if you have limited data.
- Ensembling: average predictions from multiple seeds or checkpoints to reduce variance.

Troubleshooting (common errors & fixes)
- KeyError: 'price' when running replacement scripts — some test CSVs don't contain price. The newest `replace_text_embeddings.py` checks for `price` and will proceed by writing NaN targets where absent.
- Tokenizer IndexError ("list index out of range") during batch-encoding — the scripts now print the first few texts in a failing batch. Fix by sanitizing: remove NULs, control chars, or extremely long/garbled fields. Example sanitize snippet:

```python
def sanitize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace('\x00', ' ').strip()
    s = ' '.join(s.split())  # collapse whitespace
    return s
```

- GPU errors / OOM: lower batch size, use torch.cuda.amp (mixed precision), or move to precomputed-embeddings training (fast path) where batch_size can be larger because features are smaller in memory.

Reproducibility notes
- Keep DeBERTa checkpoint, tokenizer name and MAX_LENGTH consistent between precompute and training. The project expects MAX_LENGTH=256 (254 content tokens after special-token removal).
- When saving and loading checkpoints, prefer using model.state_dict() for portability; some scripts save a full checkpoint — inspect the code to match your loading routine.

Where to look for code
- Main model: `custom_multimodel/mjave_price_model.py` (attention modules, gating, fusion, regression head).
- Embedding precomputation and replacement: `precompute_embeddings.py`, `precompute_test_embeddings.py`, `replace_text_embeddings.py`.
- Training & inference: `train_mjave_price_fast.py`, `test_mjave_inference.py`, `test_mjave_inference_live.py`.

Contact & next steps
- If you want, I can:
  - Re-run the `replace_text_embeddings.py` on your dataset (need permission to run or you can share the failing batch debug output),
  - Add a small `requirements.txt`, or
  - Add a `scripts/` folder with short wrappers for the common commands above.

License
- No license is included by default. Add one if you plan to share the code publicly (MIT/Apache-2.0 are common choices).

Acknowledgements
- The code implements a MJAVE-style multimodal fusion adapted for price regression and uses Transformers and torchvision backbones for the text and image encoders.

---
Generated README — best validation SMAPE reported here: 45.2% (user-reported). Use the troubleshooting & sanitization tips above to investigate tokenization or data-quality issues that commonly increase error.
