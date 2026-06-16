# Stego-text completion: encoding *complete essays*

## The problem

Meteor's encoding loop (`meteor-stego-paper.md`, **Algorithm 5**) is:

```
c ← ε,  n ← 0
while n < |m| do
    mask ← PRG.Next(k_prg)
    r    ← m[n : n+β] ⊕ mask
    c_i  ← Sample(H, r)
    n    ← n + LenPrefix(Recover(H, c_i))
    c    ← c ‖ c_i
Output c
```

It stops **the instant the message fits** (`n ≥ |m|`). Our server inherited that
behavior: `/encode` returned a stego-text just long enough to carry the framed
message and then halted — often mid-sentence, mid-thought. The result does not
read like a finished article, which is exactly what StegaNote needs (a complete
Nostr post/essay), and a truncated stub is itself a weak tell.

We want two things at once:

1. **Encode complete essays.** Keep generating natural text after the message is
   done, until the model has actually *finished* the piece.
2. **Decode reliably from a complete essay**, without "reading past" the message
   into the trailing cover text.

## What Meteor tells us about (2)

Look at the decoder, **Algorithm 6**:

```
for i ∈ {0 … |c|−1} do
    x_i  ← Prefix(Recover(H, c_i))
    mask ← PRG.Next(k_prg)
    x    ← x ‖ (x_i ⊕ mask[0:|x_i|])
Output x
```

Meteor's decoder walks **every token of `c`** and concatenates the recovered
bits. It has no internal "stop" signal — it relies on the caller to know how
long `m` is. In plain Meteor, `c` *is* exactly the message, so "process all
tokens" and "stop at the message" coincide. The moment we append cover text,
they diverge: the decoder would keep going and append garbage bits from the
cover tokens.

So the algorithmic question "how do I know when to stop decoding?" has a clean
answer: **the message must carry its own length, in-band.** This repo already
does that — it does *not* feed the raw plaintext to the stego layer. It wraps it
in a small self-describing **frame** first. The next section spells that frame
out in full, because it is the key to both questions.

## The frame: commit + length prefixes (what the encoded bits actually are)

### Where this happens: the server, not the client

The client interface (StegaNote / the React app) sends the server only two
things: the **plaintext `message`** and the shared **`key`** (hex). It does
*not* build the frame and does *not* compute the commit. All framing is done
**server-side**, in `main.py`'s `/encode` handler:

```python
# main.py  /encode
message_bits = text_to_message_bits(message)   # <-- builds commit + length + payload
mask_fn      = make_mask_fn(key, precision)
...
encode_steganographic(model, tokenizer, message_bits, start_text, ..., mask_fn=mask_fn)
```

So by the time the Meteor encoder runs, `message_bits` is already the *framed*
bit stream — the commit and length prefixes are part of `message_bits`, and the
per-token PRG mask is applied to *all* of those bits uniformly (the prefixes are
masked exactly like the payload). The mirror happens in `/decode`: the server
recovers the framed bits and calls `message_bits_to_text` to peel the frame back
off and return the plaintext `message` + `integrity_ok`. The client never sees
the commit or length — it only ever deals in plaintext.

### The exact layout (`stego_codec.text_to_message_bits`)

```
byte offset:  0          4            6                         6 + N
              ├──────────┼────────────┼──────────────────────────┤
              │  commit  │  length N  │        payload           │
              │ 4 bytes  │  2 bytes   │   N bytes (UTF-8 msg)     │
              │sha256[:4]│ big-endian │                          │
              └──────────┴────────────┴──────────────────────────┘
bit offset:   0         32           48                    48 + 8·N

  commit  = sha256(utf8_message)[:4]      # integrity tag / wrong-key detector
  length  = len(utf8_message) as uint16   # number of payload bytes, big-endian
  payload = the UTF-8 message bytes
```

Each **byte** is expanded to 8 bits **LSB-first** (matching `int2bits`/`bits2int`
used everywhere in the codec and the raw modules). So:

- `bits[0:32]`   — the 4 commit bytes
- `bits[32:48]`  — the 2 length bytes; `bits[32:40]` is the **high** byte
  (`N >> 8`), `bits[40:48]` is the **low** byte (`N & 0xFF`)
- `bits[48 : 48 + 8·N]` — the N payload bytes

The header is always **48 bits** (6 bytes), so the payload always begins at bit
48. That fixed offset is what lets the decoder read the length before it has seen
any payload.

### Worked example

`text_to_message_bits("Hi")` (N = 2 payload bytes) produces a **64-bit** frame
(`48 + 8·2`):

```
commit  bits[0:32]  = 0110110010011100 1111011110110011   (sha256("Hi")[:4] = 36 39 ef cd)
length  bits[32:48] = 00000000 01000000
                       └ hi byte = 0x00      └ lo byte LSB-first = 0x02  -> N = 2
payload bits[48:64] = "H" then "i"  (2 bytes, LSB-first each)
```

`text_to_message_bits("Attack at dawn")` (N = 14) produces a **160-bit** frame:
`commit = fa 43 ee 53`, `length = 00 0e` (→ 14), payload = the 14 UTF-8 bytes,
total `48 + 8·14 = 160` bits.

## The design

### Encoder — honest cover continuation, terminated by EOS (`complete_text=True`)

`encode_steganographic` takes `complete_text` (plus a `max_cover_tokens` safety
cap). It runs in two phases:

- **Message phase** — unchanged. Exactly Meteor's Algorithm 5 with the per-token
  PRG mask: consume the framed `message_bits` in order until they are exhausted.
  To the encoder there is nothing special about the prefix bits — it just walks
  the flat bit list, so it encodes the **commit bits (`bits[0:32]`) first, then
  the length bits (`bits[32:48]`), then the payload bits (`bits[48:]`)**, each
  through the same masked-sampling step (`r = chunk ⊕ mask`, pick the token whose
  interval contains `r`, advance by the shared-prefix length). A token typically
  carries a few bits, so the 48-bit header is spread across the first handful of
  tokens; the payload bits follow seamlessly. No token boundary is aligned to the
  commit/length/payload boundaries, and nothing marks them in the cover text —
  the structure exists only in the recovered bit stream.
- **Cover phase** — once `i ≥ len(message_bits)`, keep sampling tokens, but with
  **fresh random coins** instead of message bits:

  ```python
  message_idx = secrets.randbelow(2**precision)   # uniform r ∈ [0, 2^β)
  ```

  This is *exactly* an ordinary multinomial draw from the model (Meteor's
  `Sample` with random `r`, i.e. the unmodified generative process — see the
  paper's Algorithm 7 discussion). Because the cover tokens are drawn from the
  true model distribution, they are indistinguishable from normal output: a
  censor cannot find the boundary where the message ended. No mask is applied
  (there is nothing to hide), and the decoder never reads these bits anyway.

### How do you know the essay is "complete"? — the EOS token

The standard, model-native signal that generation is finished is the
**end-of-sequence (EOS) token**. A generative model is trained to emit it when
the content is complete (a finished document, the end of a turn, etc.), and it
is precisely what the library's own `model.generate()` stops on. So the cover
phase does not guess at "a natural-looking boundary" — it **samples normally and
stops when the model itself emits EOS**:

```python
if not in_message and sel_id in eos_ids:   # the model says it's done
    break                                   # essay complete (EOS not appended)
```

`eos_ids` is gathered from both the tokenizer and the model's `generation_config`
(`_eos_token_ids`), so multi-terminator models like Llama-3 Instruct
(`<|end_of_text|>` **and** `<|eot_id|>`) are handled. Because cover tokens are
drawn with uniform random coins, EOS is selected with exactly its true
probability — completion happens organically, when the model decides.

**Safety cap + clean trim.** One caveat: *base* (non-instruct) models such as
`meta-llama/Llama-3.2-1B` are trained to continue text indefinitely and rarely
emit EOS in free generation. To bound runtime we keep a `max_cover_tokens` cap.
If the cap is reached without an EOS, we **trim the continuation back to the last
sentence boundary** (tracked via `_ends_sentence` during sampling) so the essay
still ends on a complete sentence rather than mid-word:

```python
if hit_cover_cap and last_boundary_len is not None:
    output_tokens = output_tokens[:, :last_boundary_len]
```

Trimming cover tokens is always safe: the decoder only reads up to the framed
message length, which lies far before any cover token, so removing a cover tail
cannot affect recovery. (Note this is a *fallback for a clean ending*, not the
primary stop — the primary stop is EOS, which is what makes the essay genuinely
"complete." For reliably self-terminating essays, point the server at an
instruct-tuned model.)

The PRG token counter (`token_index`, which indexes the mask) advances once per
token across both phases, exactly as the decoder counts tokens — so the mask
streams stay in lockstep over the message-bearing prefix.

`main.py`'s `/encode` passes `complete_text=True`. The low-level default is
`complete_text=False`, so `test_symmetry_offline.py` and any direct callers keep
the minimal Meteor behavior.

### Decoder — stop at the frame boundary (`done_fn`)

`decode_steganographic` takes an optional `done_fn(message_bits) -> bool`,
checked after each token. The server passes `stego_codec.frame_is_complete`:

```python
def frame_bit_length(bits):           # None until the 6-byte header is in
    if len(bits) < 48: return None
    n = (bits2int(bits[32:40]) << 8) | bits2int(bits[40:48])
    return 48 + 8 * n

def frame_is_complete(bits):
    target = frame_bit_length(bits)
    return target is not None and len(bits) >= target
```

**How the prefixes determine "up to what length to decode" — step by step.**
The decoder recovers bits one token at a time (each token yields 0..β unmasked
bits, appended to a running `message_bits` list). After *every* token it calls
`done_fn(message_bits)`, i.e. `frame_is_complete`, which works in two stages:

1. **Until ≥ 48 bits are recovered**, `frame_bit_length` returns `None` (the
   length field hasn't fully arrived yet). The decoder keeps consuming tokens.
   This is why the header is a *fixed* 48 bits and the length sits at a *fixed*
   offset: the decoder can always find it without knowing anything in advance.
2. **The moment bit 48 is reached**, the length field `bits[32:48]` is fully
   recovered (and already unmasked — the prefixes are de-masked exactly like
   payload bits). `frame_bit_length` reads it as
   `N = (bits2int(bits[32:40]) << 8) | bits2int(bits[40:48])` and computes the
   **target total** `48 + 8·N`. Now the decoder knows the exact end of the
   message.
3. **Each subsequent token** adds more bits; once `len(message_bits) ≥ 48 + 8·N`,
   `frame_is_complete` returns `True` and the decoder **breaks immediately**. It
   never runs the model over the cover-text continuation that follows. (The
   commit `bits[0:32]` is *not* used to find the end — only the length is. The
   commit is checked afterward by `message_bits_to_text` to flag a wrong key.)

So the length prefix is the entire answer to "decode up to what length": read N
from a fixed offset, stop at `48 + 8·N` bits. The commit is orthogonal — it
verifies the result, it does not bound it.

This both implements "decode without reading beyond the message" **and** makes
decode faster (it processes only the message-bearing prefix, not the whole
essay).

When the target falls mid-token (the token that pushes the count to/over
`48 + 8·N`), the surplus bits of that last decoded token are harmless:
`message_bits_to_text` slices exactly `bits[0 : 48 + 8·N]` and ignores the rest.

`done_fn` defaults to `None` (decode every token), preserving the behavior of
the offline test, which round-trips raw unframed bits and must not interpret the
first 48 bits as a length header.

## Putting it together

```
CLIENT                    SERVER (/encode)
message,key ──▶ text_to_message_bits ▶ [commit(32b)‖len(16b)‖payload(8N b)] bits
                                                     │ Meteor encode (per-token mask)
                                                     ▼
                                          message tokens ─┐
                  cover phase: honest random sampling ▶ … until model emits EOS … <stop>
                                  (safety cap → trim to last sentence)
                                                     ▼
                          stego_text ◀── returned to CLIENT ──────────────────────────┘

CLIENT                    SERVER (/decode)
stego_text,key ──▶ Meteor decode (unmask), after each token check frame_is_complete:
                     • <48 bits  -> length unknown, keep going
                     • =48 bits  -> read N from bits[32:48], target = 48+8N
                     • ≥48+8N    -> STOP (cover tokens never decoded)
                   message_bits_to_text ▶ verify commit ▶ message + integrity_ok ──▶ CLIENT
```

## Why this is secure / correct

- **Security.** The message tokens are produced by unchanged Meteor (one-time-pad
  mask over the sampling value). The cover tokens are honest draws from the same
  model with uniform randomness — including the EOS draw that ends the essay — so
  the entire stegotext, message portion and continuation alike, is distributed
  like ordinary model output. The boundary leaves no statistical seam. The
  frame's commit is a `sha256` prefix of the plaintext, not a marker an observer
  can locate without the key.
- **Correctness.** Every message bit is encoded as a fixed shared-prefix bit of
  some message-phase token and recovered identically by the decoder (Meteor
  correctness). The length prefix deterministically marks the end, so the
  decoder recovers exactly the message regardless of how much cover text follows
  (and regardless of whether that cover was trimmed).

## Knobs

| Parameter | Where | Default | Meaning |
|---|---|---|---|
| `complete_text` | `encode_steganographic` | `False` (server passes `True`) | enable the cover continuation |
| `max_cover_tokens` | `encode_steganographic` | `512` | safety cap if the model never emits EOS; on the cap, trim to the last sentence boundary |
| `done_fn` | `decode_steganographic` | `None` (server passes `frame_is_complete`) | stop once the framed message is recovered |

## Tests

- `test_symmetry_offline.py` — unchanged; still 4/4 (verifies the low-level
  default behavior, `complete_text=False`, is untouched).
- Mock round-trip (no real model): with `complete_text=True` the encoder emits
  more tokens than the message needs; decoding with `done_fn=frame_is_complete`
  recovers the message and stops at the frame boundary, while a full decode of
  every token recovers the same message — confirming the cover continuation is
  ignored either way. (The mock has no EOS token, so it also exercises the
  safety-cap + sentence-trim fallback.)
