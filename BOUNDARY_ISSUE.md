# The Midpoint-Boundary Issue (and the right way to fix it)

## Summary

When the encoder tries to embed a bit chunk whose value sits on the **50%
boundary** of the probability range — canonically `[1,0,0,0,…]`, which is the
integer `2^(precision-1)`, the exact midpoint — the token it selects almost
always encodes **0 bits**. The encoder then makes no progress on that step. The
custom "75% hack" was an attempt to force progress, but it is both insecure
(predictable token selection) and incorrect (the decoder cannot invert it). This
document explains the bug precisely, how the unmodified `raw_stego_*` functions
handle it, and what the correct fix is.

## How token selection works (recap)

Each step turns the next `precision` message bits into an integer by reading the
**first** bit as the most-significant bit (`raw_stego_encoder.py`,
`bits2int(reversed(chunk))`):

```
chunk = [1,0,0,0, …]  (padded to precision=16)  ->  message_idx = 1000000000000000b = 32768
```

The model's distribution is turned into cumulative integer boundaries over
`[0, max_val)` (here `max_val = 2^16 = 65536`); token `j` owns the interval
`[cum_probs[j-1], cum_probs[j])`. The encoder selects the **one** token whose
interval contains `message_idx`, and that token encodes

```
L = num_same_from_beg(bits(bottom), bits(top-1))     # the shared leading bits
```

bits. The pointer advances by `L`. **There is no search and no rejection** — the
containing token is always found; the only question is how many leading bits its
interval pins down.

## The bug: a token straddling the midpoint fixes 0 bits

`L = 0` happens for exactly one reason: `bottom` and `top-1` differ in their
**most-significant** bit, i.e. the interval straddles the midpoint:

```
bottom < 2^(precision-1) <= top         (bottom starts with 0, top-1 starts with 1)
```

Concretely, a token with probability interval `[0.30, 0.70)`:

```
bottom = 0.30 * 65536 = 19660 = 0100110011001100
top-1  ≈ 0.70 * 65536 = 45874 = 1011001100110010
                                 ^ leading bits already differ  ->  L = 0
```

This is the *only* way to encode 0 bits. Straddling any **lower** boundary (the
25% line, the 12.5% line, …) still leaves the leading bit shared, so it encodes
≥ 1 bit. Only the top-level 50% split kills the leading bit.

Why `[1,0,0,0,…]` is the canonical trigger: it maps to `message_idx` = the exact
midpoint `32768`. Whichever token contains the point `32768` straddles it unless
a cumulative-probability boundary happens to land *exactly* on `32768` — which,
for a real distribution, essentially never happens. So `[1,0,0,0,…]` almost
always lands in a straddling token and encodes nothing. (The same 0-bit outcome
also hits a whole band of values on either side of the midpoint that fall inside
that straddling token, and — separately — an all-zeros chunk `[0,0,0,…]` when
the single highest-probability token has > 50% mass, since `message_idx = 0`
then selects it.)

This is not a corner case the scheme failed to anticipate. Variable-rate coding
is the entire point of Meteor: the number of bits per token is proportional to
the channel's instantaneous entropy, and **0 bits is a legitimate outcome** at a
low-entropy / boundary event (Meteor paper §5.1–§5.2, "the expected number of
bits encoded [is] proportional to the entropy").

## How the unmodified `raw_stego_*` functions handle it

With the hack removed, the raw encoder does the obvious, correct thing:

- `num_bits_encoded == 0`, but the selected token is **still appended** to the
  covertext (`output_tokens = cat(...)`), and the pointer advances by 0
  (`i += 0`). The token is emitted as cover that carries no payload this step.
- Because a real model is *context-dependent*, appending that token changes the
  next distribution, so the following step generally does encode bits and the
  message resumes.

The decoder mirrors this with **no special case**: it sees the same token,
reconstructs the same straddling interval, computes `L = 0`, recovers 0 bits,
and moves on (`raw_stego_decoder.py`). Encoder and decoder stay in sync.

So the raw behavior is **lossless and symmetric** — the "bug" is really
*inefficiency* (a wasted token), not incorrectness. The one way it turns fatal
is a distribution that never changes while always straddling at the needed
point: then the pointer never advances and `while i < len(message_bits)` spins
forever. A real language model does not produce a frozen distribution, but a
*constant* mock does — which is exactly the hang reproduced in
`test_symmetry_offline.py` (case C). See also `EDGE_CASES.md` for full traces.

The deeper reason this is fragile: in the raw functions, `message_idx` is a
**deterministic** function of the message bits. When a step encodes 0 bits the
pointer does not advance, so the *next* step re-presents the *same* bits → the
*same* `message_idx`. Nothing re-randomizes it; escape depends entirely on the
model's context shifting. That determinism is the root fragility.

## Why the 75% hack is the wrong fix

`raw_stego_encoder.py` (~L177–200) detects the `[1,0,0,0]` pattern on a 0-bit
step and forces `message_idx = max_val * 3 // 4`, then advances by whatever that
token fixes. Two independent problems:

1. **It is not invertible — silent corruption.** The forced sampling point is
   not derived from the message, so the selected interval's prefix is no longer
   a prefix of the message bits. The decoder (which has no matching rule) reads
   that prefix literally. When the hacked token fixes more than one bit, the
   recovered bits ≠ the message bits. Demonstrated in `test_symmetry_offline.py`
   case B: `10001111` decodes as `11001111`.
2. **It is predictable — a security leak.** Biasing selection toward the 75%
   region makes the stegotext distinguishable from honest model output, which
   breaks the indistinguishability property that is the whole purpose of the
   scheme (Meteor paper §3.1, Definition 1). As the user notes, it "introduces a
   little bit of predictability."

It also doesn't even cover the related all-zeros stall, since the pattern check
only matches `[1,0,0,0]`.

## Recommended solution

### Primary: restore the Meteor PRG mask (the design the implementation dropped)

The reference algorithm already solves this — it XORs a fresh pseudorandom
**mask** into the sampling value every step (Meteor paper §5.2, Algorithms 5 & 6,
realized with HMAC_DRBG in §6):

```
Encode (Alg 5), each iteration:        Decode (Alg 6), each token:
  mask ← PRG.Next(k_prg)                 R    ← Recover(H, c_i)
  r    ← m[n : n+β] ⊕ mask                x_i  ← Prefix(R)
  c_i  ← Sample(H, r)                     mask ← PRG.Next(k_prg)
  n_i  ← LenPrefix(Recover(H, c_i))       x    ← x ‖ (x_i ⊕ mask[0:|x_i|])
  n   += n_i                              H    ← H ‖ c_i
```

Two properties of this design fix our problem directly:

- **`PRG.Next` is called once per emitted token, not per bit consumed.** When a
  step encodes 0 bits (`n_i = 0`), `n` does not advance — but the next step draws
  a *fresh* mask, so `r = m[n:n+β] ⊕ mask` is a new, uniformly random value. The
  straddling token is therefore **not** deterministically re-selected: the 0-bit
  event cannot repeat forever, and no `[1,0,0,0]`-style special case is ever
  needed. Encoder and decoder both advance the PRG exactly once per token, so
  their mask streams stay in lockstep regardless of how many bits each token
  carried.
- **It restores security.** `r` is now a one-time-pad-masked value, uniformly
  distributed, so the selected tokens are distributed exactly like honest model
  samples (Meteor's proof of security). The current direct-bit encoding has *no*
  such guarantee — it leans entirely on the client-side ciphertext being
  pseudorandom, and even that does not help on a 0-bit reuse (the same bits are
  re-read unchanged), which is why the stall exists at all.

Implementation note for this codebase: encode and decode must share the PRG seed
`k_prg`, just as they already must share `model_id`/`temp`/`precision`/`topk`.
The Nostr app already exchanges a shared secret between sender and recipient, so
`k_prg` can be derived from it (e.g. HKDF of the shared key) and passed alongside
the other encode/decode parameters. This is the same key the (symmetric) Meteor
scheme assumes. The client-side encryption can stay; the PRG mask is an
orthogonal, server-side layer.

### Interim (if a crypto change is too large right now)

Keep the raw functions' native 0-bit handling (emit the token, advance 0 — it is
already symmetric) and simply **delete the hack**. Add a guard against
pathological low-entropy runs that is *decoder-aware and recoverable*: e.g. cap
the number of consecutive 0-bit tokens and, on hitting the cap, abort/retry with
a different starting context rather than forcing a biased selection. Any rule
that changes selection on the encoder side **must** be reproducible by the
decoder from the stegotext alone — the hack's fatal flaw is that it is not.

### Do not

Re-introduce any encoder-only forced-selection rule (the 75% hack or variants).
It corrupts data and leaks structure.

## TL;DR

The midpoint straddle producing 0 bits is expected behavior, and the raw
encoder/decoder already handle it correctly and symmetrically by emitting a
0-bit carrier token. The real defect is that this implementation removed
Meteor's per-token PRG mask, which is what (a) keeps a 0-bit event from
deterministically repeating and (b) makes the scheme secure in the first place.
Restore the mask; retire the hack.
