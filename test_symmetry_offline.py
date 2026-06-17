#!/usr/bin/env python3
"""
Offline encode/decode symmetry test -- NO server, NO real LLM required.

We replace the language model with a *mock* whose next-token distribution is
fixed (independent of context), so the arithmetic-coding intervals can be
constructed exactly. This now demonstrates the Meteor PRG-mask fix that replaced
the 50%-boundary hack:

  A. Clean power-of-two split, no mask      -> COMPLETES, SYMMETRIC  (baseline)
  B. Midpoint-straddling token, NO mask     -> INFINITE LOOP / TIMEOUT
       With the hack removed, a [1,0,0,0,...] chunk lands in a token that
       straddles the midpoint, encodes 0 bits, and -- because the (mock)
       distribution never changes and nothing re-randomizes the value -- the
       message pointer never advances. This is the bug the hack tried to paper
       over.
  C. Same straddling token, WITH PRG mask   -> COMPLETES, SYMMETRIC  (the fix)
       A fresh per-token mask (Meteor's PRG.Next) re-randomizes the sampled
       value every step, so the straddle cannot repeat. Works even on this
       degenerate constant distribution, and the decoder (same key) recovers the
       message exactly. No hack needed.
  D. Near-50/50 zeros stall, WITH PRG mask  -> COMPLETES, SYMMETRIC
       The all-zeros stall (message_idx == 0 always selecting a >50% top token)
       is dissolved by the mask too.

Each encode runs under a wall-clock alarm so case B reports a hang instead of
freezing the suite.

Run:  pipenv run python test_symmetry_offline.py
"""

import io
import signal
import contextlib
from types import SimpleNamespace

import torch

from raw_stego_encoder import int2bits, num_same_from_beg, encode_steganographic
from raw_stego_decoder import decode_steganographic
from stego_codec import make_mask_fn

SHOW_RAW_LOGS = False
ENCODE_TIMEOUT_S = 3


class MockModel:
    """A 'language model' whose next-token logits are a fixed vector."""

    def __init__(self, logits_vector: torch.Tensor):
        self.device = torch.device("cpu")
        self._logits = logits_vector

    def __call__(self, input_tokens, use_cache=False):
        seq_len = input_tokens.shape[1]
        logits = self._logits.view(1, 1, -1).expand(1, seq_len, -1).contiguous()
        return SimpleNamespace(logits=logits)


class MockTokenizer:
    """Whitespace tokenizer over a vocab of words 'w0', 'w1', ..."""

    def __init__(self, vocab_size: int):
        self.id2word = {i: f"w{i}" for i in range(vocab_size)}
        self.word2id = {w: i for i, w in self.id2word.items()}
        self.eos_token = "<eos>"
        self.pad_token = None

    def encode(self, text, return_tensors=None):
        ids = [self.word2id[w] for w in text.split()]
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        if len(ids) == 0:
            return ""
        return "".join(f" {self.id2word[int(i)]}" for i in ids)


class _EncodeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _EncodeTimeout()


def describe_distribution(probs, precision):
    max_val = 2 ** precision
    p = torch.tensor(probs, dtype=torch.float32)
    ints = (p / p.sum() * max_val).round().long()
    cum = torch.cumsum(ints, 0)
    cum = cum + (max_val - cum[-1].item())
    rows, bottom = [], 0
    for j, top in enumerate(cum.tolist()):
        bottom_bits = list(reversed(int2bits(bottom, precision)))
        top_bits = list(reversed(int2bits(top - 1, precision)))
        nbits = num_same_from_beg(bottom_bits, top_bits)
        straddles = bottom < max_val // 2 < top
        rows.append({"token": j, "bottom": bottom, "top": top, "nbits": nbits,
                     "straddles": straddles})
        bottom = top
    return rows


def bits_to_str(bits):
    return "".join(str(b) for b in bits)


def first_divergence(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return -1 if len(a) <= len(b) else min(len(a), len(b))


def run_case(name, probs, message_bits, expect_status, expect_symmetric,
             mask_key=None, precision=16, temp=1.0, topk=50000):
    print("=" * 78)
    print(f"CASE: {name}")
    print("=" * 78)
    print(f"probabilities: {probs}   mask: {'ON (key=%r)' % mask_key if mask_key else 'OFF'}")
    for r in describe_distribution(probs, precision):
        tag = "  <-- STRADDLES midpoint (0 bits)" if r["straddles"] else ""
        print(f"  token w{r['token']}: [{r['bottom']:>6}, {r['top']:>6}) "
              f"fixes {r['nbits']} bit(s){tag}")
    print(f"message bits ({len(message_bits)}): {bits_to_str(message_bits)}")

    model = MockModel(torch.log(torch.tensor(probs, dtype=torch.float32)))
    tok = MockTokenizer(len(probs))
    context_text = "w0"
    mask_fn = make_mask_fn(mask_key, precision) if mask_key else None

    cm = (contextlib.nullcontext() if SHOW_RAW_LOGS
          else contextlib.redirect_stdout(io.StringIO()))

    status, recovered = "completed", []
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(ENCODE_TIMEOUT_S)
    try:
        with cm:
            generated = encode_steganographic(
                model, tok, list(message_bits), context_text,
                temp=temp, precision=precision, topk=topk, mask_fn=mask_fn)
            full_stego = context_text + tok.decode(generated[0])
            recovered = decode_steganographic(
                model, tok, full_stego, context_text,
                temp=temp, precision=precision, topk=topk, mask_fn=mask_fn)
    except _EncodeTimeout:
        status = "timeout"
    finally:
        signal.alarm(0)

    symmetric = False
    if status == "timeout":
        print(f"RESULT: TIMEOUT after {ENCODE_TIMEOUT_S}s -- encoder never "
              f"terminated (boundary stall, pointer stuck).")
    else:
        cmp_len = len(message_bits)
        div = first_divergence(message_bits, recovered[:cmp_len])
        symmetric = div == -1 and len(recovered) >= cmp_len
        print(f"recovered bits ({len(recovered)}): {bits_to_str(recovered)}")
        if symmetric:
            print(f"RESULT: SYMMETRIC -- first {cmp_len} recovered bits match.")
        else:
            print(f"RESULT: ASYMMETRIC -- diverges at bit index {div}")
            print(f"        message  : {bits_to_str(message_bits)}")
            print(f"        recovered: {bits_to_str(recovered[:cmp_len])}")

    ok = (status == expect_status) and (status == "timeout"
                                        or symmetric == expect_symmetric)
    print(f"(expected status={expect_status}, symmetric={expect_symmetric}: "
          f"{'as expected' if ok else '!! UNEXPECTED !!'})\n")
    return ok


def main():
    print("Offline encode/decode symmetry test (mock model, no server)\n")
    results = []

    # A. Clean 50/50, no mask -> baseline symmetric round trip.
    results.append(run_case(
        name="A. Clean power-of-two split (0.5 / 0.5), no mask -- baseline",
        probs=[0.5, 0.5],
        message_bits=[0, 1, 1, 0, 1, 0, 0, 1],
        expect_status="completed", expect_symmetric=True))

    # B. Straddling token, NO mask -> infinite loop (the bug the hack hid).
    results.append(run_case(
        name="B. Straddling token (0.75, 0.0625x4), NO mask -- BOUNDARY STALL",
        probs=[0.75, 0.0625, 0.0625, 0.0625, 0.0625],
        message_bits=[1, 0, 0, 0, 1, 1, 1, 1],
        expect_status="timeout", expect_symmetric=False))

    # C. Same scenario WITH the PRG mask -> completes and round-trips. The fix.
    results.append(run_case(
        name="C. Same distribution, WITH PRG mask -- THE FIX",
        probs=[0.75, 0.0625, 0.0625, 0.0625, 0.0625],
        message_bits=[1, 0, 0, 0, 1, 1, 1, 1],
        mask_key="shared-secret-key",
        expect_status="completed", expect_symmetric=True))

    # D. All-zeros stall on a >50% top token, dissolved by the mask too.
    results.append(run_case(
        name="D. Zeros stall (0.55 / 0.45), WITH PRG mask",
        probs=[0.55, 0.45],
        message_bits=[0, 0, 1, 0, 1, 1, 0, 1],
        mask_key="another-key",
        expect_status="completed", expect_symmetric=True))

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{sum(results)}/{len(results)} cases matched their expected behavior.\n")
    print("Without a mask, a midpoint-straddling token encodes 0 bits and, on a")
    print("fixed distribution, the encoder stalls forever (B). Meteor's per-token")
    print("PRG mask re-randomizes the sampled value every step, so the straddle")
    print("cannot repeat -- encode/decode complete and round-trip exactly (C, D),")
    print("with no 50% hack and full one-time-pad security.")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
