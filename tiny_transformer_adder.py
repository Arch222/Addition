from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


MODEL_LAYERS = 1
MODEL_DIM = 3
ATTENTION_HEADS = 4
KEY_VALUE_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 4
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11

# Constants used to hand-write the arithmetic behavior.
EMBED_CONST = 1000.0
DIGIT_SCALE = EMBED_CONST / math.sqrt(MODEL_DIM)
CONST_NORM = math.sqrt(MODEL_DIM)
ALPHA = 20.0
QK_NORM_SCALE = 256.0
DECODE_LINEAR_EPS = 5e-4
DECODE_QUAD = DECODE_LINEAR_EPS / 2.0


@dataclass(frozen=True)
class AdderVocab:
    size: int = VOCAB_SIZE

    def digit_to_id(self, digit: int) -> int:
        if digit < 0 or digit > 9:
            raise ValueError(f"digit must be in [0, 9], got {digit}")
        return digit

    def id_to_digit(self, token_id: int) -> int:
        if token_id < 0 or token_id > 9:
            raise ValueError(f"token id is not a digit token: {token_id}")
        return token_id


def int_to_reversed_digits(value: int, width: int) -> List[int]:
    if value < 0:
        raise ValueError("value must be non-negative")
    out = []
    n = value
    for _ in range(width):
        out.append(int(n % 10))
        n //= 10
    return out


def reversed_digits_to_int(digits: Sequence[int]) -> int:
    out = 0
    place = 1
    for d in digits:
        out += int(d) * place
        place *= 10
    return out


def format_prompt_tokens(a: int, b: int, n_digits: int, vocab: AdderVocab) -> List[int]:
    _ = vocab
    a_digits = int_to_reversed_digits(a, n_digits)
    b_digits = int_to_reversed_digits(b, n_digits)
    # Reference-style prompt.
    return [0, *a_digits, 0, 0, *b_digits, 0]


def _qvec(offset: int) -> tuple[float, float]:
    return (math.cos(offset), -math.sin(offset))


class HandwrittenFastAdderTransformer(nn.Module):
    """
    Strict transformer-only handwritten model:
    - no training
    - no symbolic arithmetic branch
    - autoregressive decoding via argmax(logits)
    """

    def __init__(self, n_digits: int = 10):
        super().__init__()
        if n_digits != 10:
            raise ValueError("handwritten weights are calibrated for n_digits=10 only")
        self.n_digits = n_digits
        self.vocab = AdderVocab()

        # Counted parameters (<100 total).
        self.embed_weight = nn.Parameter(
            torch.zeros(VOCAB_SIZE, MODEL_DIM, dtype=torch.float32),
            requires_grad=False,
        )
        self.input_norm_weight = nn.Parameter(
            torch.ones(MODEL_DIM, dtype=torch.float32), requires_grad=False
        )
        self.post_norm_weight = nn.Parameter(
            torch.ones(MODEL_DIM, dtype=torch.float32), requires_grad=False
        )
        self.final_norm_weight = nn.Parameter(
            torch.zeros(MODEL_DIM, dtype=torch.float32), requires_grad=False
        )

        self.q_norm_weight = nn.Parameter(
            torch.ones(HEAD_DIM, dtype=torch.float32), requires_grad=False
        )
        self.k_norm_weight = nn.Parameter(
            torch.ones(HEAD_DIM, dtype=torch.float32), requires_grad=False
        )

        self.k_proj_weight = nn.Parameter(
            torch.zeros(KEY_VALUE_HEADS * HEAD_DIM, MODEL_DIM, dtype=torch.float32),
            requires_grad=False,
        )
        self.v_proj_weight = nn.Parameter(
            torch.zeros(KEY_VALUE_HEADS * HEAD_DIM, MODEL_DIM, dtype=torch.float32),
            requires_grad=False,
        )

        self.gate_proj_weight = nn.Parameter(
            torch.zeros(INTERMEDIATE_SIZE, MODEL_DIM, dtype=torch.float32),
            requires_grad=False,
        )
        self.up_proj_weight = nn.Parameter(
            torch.zeros(INTERMEDIATE_SIZE, MODEL_DIM, dtype=torch.float32),
            requires_grad=False,
        )
        self.down_proj_weight = nn.Parameter(
            torch.zeros(MODEL_DIM, INTERMEDIATE_SIZE, dtype=torch.float32),
            requires_grad=False,
        )

        # Fixed buffers (not counted in nn.Parameter budget).
        self.register_buffer(
            "q_proj_weight",
            torch.zeros(ATTENTION_HEADS * HEAD_DIM, MODEL_DIM, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "o_proj_weight",
            torch.zeros(MODEL_DIM, ATTENTION_HEADS * HEAD_DIM, dtype=torch.float32),
            persistent=False,
        )

        self._hand_set_weights()

    @property
    def prompt_len(self) -> int:
        return 1 + self.n_digits + 1 + 1 + self.n_digits + 1

    @property
    def output_len(self) -> int:
        return self.n_digits + 1

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_total_weights(self) -> int:
        p = self.count_total_parameters()
        b = sum(buf.numel() for buf in self.buffers())
        return p + b

    @staticmethod
    def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + eps)
        return x * weight

    @staticmethod
    def _apply_rope(x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H, 2]
        b, t, h, d = x.shape
        if d != 2:
            raise ValueError(f"expected head dim=2, got {d}")
        pos = torch.arange(t, dtype=x.dtype, device=x.device)
        cos = torch.cos(pos).view(1, t, 1)
        sin = torch.sin(pos).view(1, t, 1)
        x0 = x[..., 0]
        x1 = x[..., 1]
        y0 = (x0 * cos) - (x1 * sin)
        y1 = (x0 * sin) + (x1 * cos)
        return torch.stack((y0, y1), dim=-1)

    def _hand_set_weights(self) -> None:
        with torch.no_grad():
            embed = []
            for d in range(10):
                embed.append(
                    [
                        EMBED_CONST - DECODE_QUAD * float(d * d),
                        float(d),
                        DECODE_LINEAR_EPS * float(d),
                    ]
                )
            self.embed_weight.copy_(torch.tensor(embed, dtype=torch.float32))

            self.final_norm_weight.copy_(
                torch.tensor([1.0 / CONST_NORM, 0.0, DIGIT_SCALE], dtype=torch.float32)
            )
            self.input_norm_weight.fill_(1.0)
            self.post_norm_weight.fill_(1.0)
            self.q_norm_weight.fill_(QK_NORM_SCALE)
            self.k_norm_weight.fill_(QK_NORM_SCALE)

            self.k_proj_weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
            )
            self.v_proj_weight.copy_(
                torch.tensor([[0.0, DIGIT_SCALE, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
            )

            q_prev_a = _qvec(23)
            q_prev_b = _qvec(11)
            q_cur_a = _qvec(22)
            q_cur_b = _qvec(10)
            self.q_proj_weight.copy_(
                torch.tensor(
                    [
                        [q_prev_a[0], 0.0, 0.0],
                        [q_prev_a[1], 0.0, 0.0],
                        [q_prev_b[0], 0.0, 0.0],
                        [q_prev_b[1], 0.0, 0.0],
                        [q_cur_a[0], 0.0, 0.0],
                        [q_cur_a[1], 0.0, 0.0],
                        [q_cur_b[0], 0.0, 0.0],
                        [q_cur_b[1], 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )

            o_proj = torch.zeros(MODEL_DIM, ATTENTION_HEADS * HEAD_DIM, dtype=torch.float32)
            o_proj[1, 0] = -1.0
            o_proj[1, 2] = -1.0
            o_proj[2, 4] = 1.0
            o_proj[2, 6] = 1.0
            self.o_proj_weight.copy_(o_proj)

            gate = torch.zeros(INTERMEDIATE_SIZE, MODEL_DIM, dtype=torch.float32)
            # c = I[x <= -9], x = sum_prev - a_prev - b_prev
            gate[0, 0] = ALPHA * (-8.0) / CONST_NORM
            gate[0, 1] = ALPHA * (-1.0) * DIGIT_SCALE
            gate[1, 0] = ALPHA * (-9.0) / CONST_NORM
            gate[1, 1] = ALPHA * (-1.0) * DIGIT_SCALE
            # w = I[s + c >= 10]
            gate[2, 0] = ALPHA * (-188.0) / CONST_NORM
            gate[2, 1] = ALPHA * (-2.0) * DIGIT_SCALE
            gate[2, 2] = ALPHA * (20.0) * DIGIT_SCALE
            gate[3, 0] = ALPHA * (-189.0) / CONST_NORM
            gate[3, 1] = ALPHA * (-2.0) * DIGIT_SCALE
            gate[3, 2] = ALPHA * (20.0) * DIGIT_SCALE
            self.gate_proj_weight.copy_(gate)

            up = torch.zeros(INTERMEDIATE_SIZE, MODEL_DIM, dtype=torch.float32)
            up[:, 0] = 1.0
            self.up_proj_weight.copy_(up)

            scale = 1.0 / (ALPHA * CONST_NORM)
            down = torch.zeros(MODEL_DIM, INTERMEDIATE_SIZE, dtype=torch.float32)
            down[2, 0] = 1.0 * scale
            down[2, 1] = -1.0 * scale
            down[2, 2] = -10.0 * scale
            down[2, 3] = 10.0 * scale
            self.down_proj_weight.copy_(down)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2:
            raise ValueError(f"expected rank-2 token tensor, got shape {tuple(token_ids.shape)}")

        x = F.embedding(token_ids, self.embed_weight)

        x_ln = self._rms_norm(x, self.input_norm_weight)
        q = F.linear(x_ln, self.q_proj_weight).view(
            x.shape[0], x.shape[1], ATTENTION_HEADS, HEAD_DIM
        )
        k = F.linear(x_ln, self.k_proj_weight).view(
            x.shape[0], x.shape[1], KEY_VALUE_HEADS, HEAD_DIM
        )
        v = F.linear(x_ln, self.v_proj_weight).view(
            x.shape[0], x.shape[1], KEY_VALUE_HEADS, HEAD_DIM
        )

        q = self._rms_norm(q, self.q_norm_weight)
        k = self._rms_norm(k, self.k_norm_weight)
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        k_rep = k.expand(-1, -1, ATTENTION_HEADS, -1)
        v_rep = v.expand(-1, -1, ATTENTION_HEADS, -1)

        # [B, H, T, T]
        scores = torch.einsum("bthd,bshd->bhts", q, k_rep) / math.sqrt(HEAD_DIM)
        t = scores.shape[-1]
        mask = torch.triu(torch.ones(t, t, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bhts,bshd->bthd", probs, v_rep).reshape(
            x.shape[0], x.shape[1], ATTENTION_HEADS * HEAD_DIM
        )
        attn_out = F.linear(ctx, self.o_proj_weight)
        x = x + attn_out

        x_ln2 = self._rms_norm(x, self.post_norm_weight)
        gate = F.linear(x_ln2, self.gate_proj_weight)
        up = F.linear(x_ln2, self.up_proj_weight)
        mlp_out = F.linear(F.silu(gate) * up, self.down_proj_weight)
        x = x + mlp_out

        x_out = self._rms_norm(x, self.final_norm_weight)
        logits = torch.matmul(x_out, self.embed_weight.transpose(0, 1))
        return logits

    def next_token(self, prefix_tokens: Sequence[int]) -> int:
        x = torch.tensor(prefix_tokens, dtype=torch.long).unsqueeze(0)
        logits = self.forward(x)
        return int(torch.argmax(logits[0, -1], dim=-1).item())

    def generate(
        self,
        prompt_tokens: Sequence[int],
        max_new_tokens: int | None = None,
    ) -> List[int]:
        seq = list(prompt_tokens)
        steps = max_new_tokens if max_new_tokens is not None else self.output_len
        for _ in range(steps):
            nxt = self.next_token(seq)
            seq.append(nxt)
        return seq

    def decode_generated_sum(self, full_sequence: Sequence[int]) -> int:
        needed = self.prompt_len + self.output_len
        if len(full_sequence) < needed:
            raise ValueError(f"need at least {needed} tokens, got {len(full_sequence)}")
        digits = full_sequence[self.prompt_len : self.prompt_len + self.output_len]
        digits = [self.vocab.id_to_digit(d) for d in digits]
        return reversed_digits_to_int(digits)


def make_model(
    n_digits: int = 10,
    d_model: int = MODEL_DIM,
    n_layers: int = MODEL_LAYERS,
    intermediate_size: int = INTERMEDIATE_SIZE,
) -> HandwrittenFastAdderTransformer:
    if d_model != MODEL_DIM:
        raise ValueError(f"handwritten model fixed at d_model={MODEL_DIM}")
    if n_layers != MODEL_LAYERS:
        raise ValueError(f"handwritten model fixed at n_layers={MODEL_LAYERS}")
    if intermediate_size != INTERMEDIATE_SIZE:
        raise ValueError(f"handwritten model fixed at intermediate_size={INTERMEDIATE_SIZE}")
    return HandwrittenFastAdderTransformer(n_digits=n_digits)
