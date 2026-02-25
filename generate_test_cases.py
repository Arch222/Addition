from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

from tiny_transformer_adder import int_to_reversed_digits


def weighted_choice(rng: random.Random, weights: List[float]) -> int:
    total = sum(weights)
    r = rng.random() * total
    cdf = 0.0
    for i, w in enumerate(weights):
        cdf += w
        if r <= cdf:
            return i
    return len(weights) - 1


def sample_digits_autoregressive(rng: random.Random, n_digits: int) -> List[int]:
    # Least-significant digit first. Each next digit depends on the previous digit.
    digits = [rng.randrange(10)]
    for pos in range(1, n_digits):
        prev = digits[-1]
        bias_a = (prev * 7 + 3 + pos) % 10
        bias_b = (prev + pos) % 10
        weights = [1.0] * 10
        weights[bias_a] += 4.0
        weights[bias_b] += 2.0
        digits.append(weighted_choice(rng, weights))
    return digits


def digits_to_int(digits_le: List[int]) -> int:
    out = 0
    place = 1
    for d in digits_le:
        out += int(d) * place
        place *= 10
    return out


def make_heldout_cases(size: int, n_digits: int, seed: int) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    seen: set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    while len(out) < size:
        a_digits = sample_digits_autoregressive(rng, n_digits)
        b_digits = sample_digits_autoregressive(rng, n_digits)
        a = digits_to_int(a_digits)
        b = digits_to_int(b_digits)
        pair = (a, b)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autoregressively generate held-out addition test cases."
    )
    parser.add_argument("--n-digits", type=int, default=10)
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/heldout_autoreg_10digit.jsonl"),
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cases = make_heldout_cases(size=args.size, n_digits=args.n_digits, seed=args.seed)
    with args.out.open("w", encoding="utf-8") as f:
        for a, b in cases:
            s = a + b
            row = {
                "a": a,
                "b": b,
                "sum": s,
                "a_rev": int_to_reversed_digits(a, args.n_digits),
                "b_rev": int_to_reversed_digits(b, args.n_digits),
                "sum_rev": int_to_reversed_digits(s, args.n_digits + 1),
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(cases)} held-out cases to {args.out}")


if __name__ == "__main__":
    main()
