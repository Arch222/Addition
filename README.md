# Handwritten Strict Transformer Adder (<100 params)

This implementation is:

- decoder-only transformer
- no training
- autoregressive (`argmax` of model logits at each step)
- no symbolic/carry solver branch at inference
- calibrated for **10-digit + 10-digit** addition

## Parameter budget

- Counted parameters (`nn.Parameter`): **91**
- Trainable parameters: **0**
- Total weight scalars including fixed buffers: **139**

The `<100` target is satisfied on `nn.Parameter` count.

## Architecture

- 1 decoder layer
- hidden size = 3
- attention heads = 4
- KV heads = 1
- head dim = 2
- MLP hidden = 4
- vocab size = 10 (digit tokens only)

The weight design follows the reference-style handwritten setup:

- large constant embedding channel for stable RMSNorm
- RoPE offset-targeted queries
- attention extracts previous/current aligned digits
- MLP implements carry/overflow logic via thresholded linear pieces
- tied embedding decode produces digit logits

## Prompt / output format

Prompt tokens:

`[0] + reverse(a_10_digits) + [0] + [0] + reverse(b_10_digits) + [0]`

Generated tokens:

`11` reversed sum digits (fixed length).

## Generate held-out set

```powershell
python generate_test_cases.py --n-digits 10 --size 100000 --seed 12345 --out data/heldout_autoreg_10digit.jsonl
```

## Evaluate

```powershell
python evaluate.py --cases data/heldout_autoreg_10digit.jsonl --n-digits 10 --batch-size 2048
```

Observed:

- `total_parameters=91`
- `accuracy=1.000000` on `100000` held-out cases

## Quick stress

```powershell
python stress_boundaries.py --digit-sizes 10 --cases-per-size 2000 --batch-size 1024
```

`n_digits > 10` is intentionally unsupported by this handwritten weight set.

## Archived prior version

The earlier fixed-transformer implementation is preserved in:

`archived_fixed_transformer_v1/`
