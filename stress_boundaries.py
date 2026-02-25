from __future__ import annotations

import argparse
import random

import torch

from tiny_transformer_adder import format_prompt_tokens, make_model


def generate_batch(model, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    seq = torch.tensor(prompts, dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = model(seq)
        next_tok = torch.argmax(logits, dim=-1)
        next_tok = next_tok[:, -1]
        seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
    return seq.tolist()


def run_eval(
    n_digits: int,
    cases: int,
    seed: int,
    d_model: int,
    n_layers: int,
    intermediate_size: int,
    batch_size: int,
) -> float:
    if n_digits != 10:
        raise ValueError("handwritten strict model currently supports n_digits=10 only")
    rng = random.Random(seed)
    model = make_model(
        n_digits=n_digits,
        d_model=d_model,
        n_layers=n_layers,
        intermediate_size=intermediate_size,
    )
    max_val = 10 ** n_digits
    correct = 0
    done = 0
    while done < cases:
        cur = min(batch_size, cases - done)
        batch_pairs = [(rng.randrange(max_val), rng.randrange(max_val)) for _ in range(cur)]
        prompts = [
            format_prompt_tokens(a, b, n_digits, model.vocab)
            for a, b in batch_pairs
        ]
        out = generate_batch(model, prompts, max_new_tokens=n_digits + 1)
        for (a, b), seq in zip(batch_pairs, out):
            pred = model.decode_generated_sum(seq)
            if pred == a + b:
                correct += 1
        done += cur
    return correct / cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary stress for handwritten strict decoder-only transformer adder.")
    parser.add_argument("--digit-sizes", type=int, nargs="+", default=[10])
    parser.add_argument("--cases-per-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--d-model", type=int, default=3)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--intermediate-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    for n in args.digit_sizes:
        try:
            acc = run_eval(
                n_digits=n,
                cases=args.cases_per_size,
                seed=args.seed + n,
                d_model=args.d_model,
                n_layers=args.n_layers,
                intermediate_size=args.intermediate_size,
                batch_size=args.batch_size,
            )
            print(f"n_digits={n} accuracy={acc:.6f}")
        except ValueError as e:
            print(f"n_digits={n} unsupported: {e}")


if __name__ == "__main__":
    main()
