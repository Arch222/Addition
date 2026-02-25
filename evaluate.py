from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tiny_transformer_adder import format_prompt_tokens, make_model


def load_cases(path: Path, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def generate_batch(model, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    seq = torch.tensor(prompts, dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = model(seq)
        next_tok = torch.argmax(logits, dim=-1)
        next_tok = next_tok[:, -1]
        seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
    return seq.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate handwritten strict decoder-only transformer adder.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/heldout_autoreg_10digit.jsonl"),
    )
    parser.add_argument("--n-digits", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=3)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--intermediate-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    model = make_model(
        n_digits=args.n_digits,
        d_model=args.d_model,
        n_layers=args.n_layers,
        intermediate_size=args.intermediate_size,
    )
    if args.checkpoint is not None and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
    trainable_params = model.count_trainable_parameters()
    total_params = model.count_total_parameters()
    total_weights = model.count_total_weights() if hasattr(model, "count_total_weights") else total_params
    if total_params >= 100:
        raise RuntimeError(
            f"model has {total_params} total parameters, expected < 100"
        )

    cases = load_cases(args.cases, limit=args.limit)
    if not cases:
        raise RuntimeError(f"no test cases found in {args.cases}")

    t0 = time.perf_counter()
    correct = 0
    for start in range(0, len(cases), args.batch_size):
        chunk = cases[start : start + args.batch_size]
        prompts = [
            format_prompt_tokens(row["a"], row["b"], args.n_digits, model.vocab)
            for row in chunk
        ]
        generated = generate_batch(model, prompts, max_new_tokens=args.n_digits + 1)
        for row, seq in zip(chunk, generated):
            pred = model.decode_generated_sum(seq)
            if pred == row["sum"]:
                correct += 1
    dt = time.perf_counter() - t0

    total = len(cases)
    acc = correct / total
    per_sec = total / dt if dt > 0 else float("inf")
    print(f"total_parameters={total_params}")
    print(f"total_weights_including_buffers={total_weights}")
    print(f"trainable_parameters={trainable_params}")
    print(f"cases={total}")
    print(f"correct={correct}")
    print(f"accuracy={acc:.6f}")
    print(f"throughput_cases_per_sec={per_sec:.1f}")


if __name__ == "__main__":
    main()
