#!/usr/bin/env python3
"""
Offline encode/decode symmetry test -- NO server, NO real LLM required.

We replace the language model with a *mock* whose next-token distribution is
fixed (independent of context). Because the distribution is fully under our
control, we can deterministically construct the exact arithmetic-coding
intervals that exercise the edge cases -- in particular the "50% boundary hack"
in raw_stego_encoder.py.

Background -- the 50% boundary hack (encoder, raw_stego_encoder.py ~L177-200):
  Each token fixes the leading bits shared by its probability interval
  [bottom, top). A token whose interval *straddles* the precision midpoint
  (bottom < 2^(p-1) <= top) shares no prefix and therefore encodes 0 bits.
  To avoid stalling on such a token when the pending chunk looks like
  [1,0,0,0,...], the encoder forces the sampling point to the 75% mark, selects
  whatever token sits there, and advances the message pointer by however many
  prefix bits THAT token's interval happens to fix.

  The decoder (raw_stego_decoder.py) has NO matching special case: it just reads
  the shared-prefix bits of each token it sees. They stay in sync only when the
  hacked token fixes exactly the one bit the encoder meant to send.

This script demonstrates three behaviours:

  A. Clean power-of-two split            -> COMPLETES, SYMMETRIC   (baseline)
  B. Hack OVER-ADVANCES and corrupts     -> COMPLETES, ASYMMETRIC  (data bug)
       The 75% point lands in a narrow interval high in the range that fixes
       *several* prefix bits. The encoder skips that many message bits, but the
       decoder reads those prefix bits literally -- they are not the message.
  C. Hack does not apply -> INFINITE LOOP -> TIMEOUT (hang hazard)
       Any 0-leading chunk that selects a straddling token encodes 0 bits and,
       without the [1,0,0,0] pattern, the hack stays silent: the pointer never
       advances and `while i < len(message_bits)` spins forever. The hack only
       patches the single point idx == 2^(p-1); the rest of a straddling token's
       lower span still hangs.

Each encode is run under a wall-clock alarm so case C reports a hang instead of
freezing the suite.

Run:  pipenv run python test_symmetry_offline.py
"""

import io
import signal
import contextlib
from types import SimpleNamespace

import torch

from raw_stego_encoder import (
    int2bits,
    num_same_from_beg,
    encode_steganographic,
)
from raw_stego_decoder import decode_steganographic

# Set True to see the encoder/decoder's own (very chatty) progress output.
SHOW_RAW_LOGS = False
# Seconds before we declare an encode hung (infinite loop).
ENCODE_TIMEOUT_S = 3


# --------------------------------------------------------------------------- #
# Mock model + tokenizer
# --------------------------------------------------------------------------- #
class MockModel:
    """A 'language model' whose next-token logits are a fixed vector.

    The real functions only ever read logits[:, -1, :], so a context-independent
    distribution is enough to drive the arithmetic coder -- and it makes the
    encoder and decoder see *identical* distributions at every step, which is
    the synchronization the protocol relies on."""

    def __init__(self, logits_vector: torch.Tensor):
        self.device = torch.device("cpu")
        self._logits = logits_vector

    def __call__(self, input_tokens, use_cache=False):
        seq_len = input_tokens.shape[1]
        logits = self._logits.view(1, 1, -1).expand(1, seq_len, -1).contiguous()
        return SimpleNamespace(logits=logits)


class MockTokenizer:
    """Whitespace tokenizer over a vocab of words 'w0', 'w1', ...

    decode() emits a leading space per token so that
    `context_text + decode(generated)` re-tokenizes back to the original token
    stream (the round-trip the decoder depends on)."""

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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _EncodeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _EncodeTimeout()


def describe_distribution(probs, precision):
    """Reproduce the encoder's interval math so we can show, per token, the
    fixed interval [bottom, top) and how many leading bits it pins down."""
    max_val = 2 ** precision
    p = torch.tensor(probs, dtype=torch.float32)
    ints = (p / p.sum() * max_val).round().long()
    cum = torch.cumsum(ints, 0)
    cum = cum + (max_val - cum[-1].item())  # dump leftover mass on the top token
    rows, bottom = [], 0
    for j, top in enumerate(cum.tolist()):
        bottom_bits = list(reversed(int2bits(bottom, precision)))
        top_bits = list(reversed(int2bits(top - 1, precision)))
        nbits = num_same_from_beg(bottom_bits, top_bits)
        straddles = bottom < max_val // 2 < top
        rows.append({
            "token": j, "bottom": bottom, "top": top,
            "frac": (bottom / max_val, top / max_val),
            "fixed_bits": top_bits[:nbits], "straddles": straddles,
        })
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
             precision=16, temp=1.0, topk=50000):
    max_val = 2 ** precision
    print("=" * 78)
    print(f"CASE: {name}")
    print("=" * 78)
    print(f"precision={precision} (max_val={max_val}), temp={temp}")
    print(f"probabilities: {probs}")
    print("Per-token intervals (what each token fixes):")
    for r in describe_distribution(probs, precision):
        lo, hi = r["frac"]
        tag = "  <-- STRADDLES midpoint (encodes 0 bits)" if r["straddles"] else ""
        print(
            f"  token w{r['token']}: [{r['bottom']:>6}, {r['top']:>6}) "
            f"= [{lo:.4f}, {hi:.4f})  fixes {len(r['fixed_bits'])} bit(s): "
            f"{bits_to_str(r['fixed_bits']) or '-'}{tag}"
        )
    print(f"message bits ({len(message_bits)}): {bits_to_str(message_bits)}")

    model = MockModel(torch.log(torch.tensor(probs, dtype=torch.float32)))
    tok = MockTokenizer(len(probs))
    context_text = "w0"

    cm = (contextlib.nullcontext() if SHOW_RAW_LOGS
          else contextlib.redirect_stdout(io.StringIO()))

    status, recovered, gen_ids = "completed", [], []
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(ENCODE_TIMEOUT_S)
    try:
        with cm:
            generated = encode_steganographic(
                model, tok, list(message_bits), context_text,
                temp=temp, precision=precision, topk=topk, verbose=False)
            gen_ids = generated[0].tolist()
            full_stego = context_text + tok.decode(generated[0])
            recovered = decode_steganographic(
                model, tok, full_stego, context_text,
                temp=temp, precision=precision, topk=topk, verbose=False)
    except _EncodeTimeout:
        status = "timeout"
    finally:
        signal.alarm(0)

    symmetric = False
    if status == "timeout":
        print(f"RESULT: TIMEOUT after {ENCODE_TIMEOUT_S}s -- encoder never "
              f"terminated (infinite-loop hazard: pointer stuck, 0 bits/token).")
    else:
        print(f"generated tokens ({len(gen_ids)}): {['w'+str(i) for i in gen_ids]}")
        print(f"recovered bits ({len(recovered)}): {bits_to_str(recovered)}")
        cmp_len = len(message_bits)
        div = first_divergence(message_bits, recovered[:cmp_len])
        symmetric = div == -1 and len(recovered) >= cmp_len
        if symmetric:
            print(f"RESULT: SYMMETRIC -- first {cmp_len} recovered bits match "
                  f"(tail beyond {cmp_len} is expected final-token padding).")
        else:
            print(f"RESULT: ASYMMETRIC -- diverges at bit index {div}")
            print(f"        message  : {bits_to_str(message_bits)}")
            print(f"        recovered: {bits_to_str(recovered[:cmp_len])}")
            print(f"                   {' ' * max(div, 0)}^")

    ok = (status == expect_status) and (symmetric == expect_symmetric or
                                         status == "timeout")
    print(f"(expected status={expect_status}, symmetric={expect_symmetric}: "
          f"{'as expected' if ok else '!! UNEXPECTED !!'})\n")
    return ok


def main():
    print("Offline encode/decode symmetry test (mock model, no server)\n")
    results = []

    # A. Clean 50/50 split: token0=[0,0.5) fixes bit 0, token1=[0.5,1) fixes
    #    bit 1. Nothing straddles, the hack never fires. Fully symmetric.
    results.append(run_case(
        name="A. Clean power-of-two split (0.5 / 0.5) -- baseline",
        probs=[0.5, 0.5],
        message_bits=[0, 1, 1, 0, 1, 0, 0, 1],
        expect_status="completed", expect_symmetric=True))

    # B. Skewed: token0=[0,0.75) straddles the midpoint. A [1,0,0,0,...] chunk
    #    (idx == 0.5) selects it -> 0 bits -> hack fires -> 75% point lands in
    #    the NARROW token1=[0.75,0.8125), which fixes 4 prefix bits "1100".
    #    The encoder advances 4 bits but only meant the single bit 1, so the
    #    decoder reads back 1100 where the message had 1000 -> corruption.
    #    (The over-advance also skips the trailing zeros, which is why it still
    #    terminates instead of hanging like case C.)
    results.append(run_case(
        name="B. Hack over-advances (0.75, 0.0625x4) -- COMPLETES but CORRUPTS",
        probs=[0.75, 0.0625, 0.0625, 0.0625, 0.0625],
        message_bits=[1, 0, 0, 0, 1, 1, 1, 1],
        expect_status="completed", expect_symmetric=False))

    # C. Near 50/50: token0=[0,0.55) straddles. A 0-leading chunk selects it,
    #    encodes 0 bits, and the hack's [1,0,0,0] pattern does NOT match, so the
    #    pointer never advances -> infinite loop. Demonstrates the hang hazard.
    results.append(run_case(
        name="C. Hack can't apply (0.55 / 0.45) -- INFINITE LOOP / hang hazard",
        probs=[0.55, 0.45],
        message_bits=[0, 0, 1, 0, 1, 1, 0, 1],
        expect_status="timeout", expect_symmetric=False))

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{sum(results)}/{len(results)} cases matched their expected behavior.\n")
    print("Findings about the 50% boundary hack vs. the (unmodified) decoder:")
    print("  * Clean, non-straddling distributions round-trip correctly (A).")
    print("  * When the hacked 75% token fixes >1 bit, the encoder over-advances")
    print("    the message pointer and the decoder reads prefix bits that are not")
    print("    the message -> silent corruption (B).")
    print("  * The hack only rescues the single point idx==2^(p-1) with pattern")
    print("    [1,0,0,0]; any other 0-leading chunk on a straddling token still")
    print("    encodes 0 bits and the encoder loops forever (C).")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
