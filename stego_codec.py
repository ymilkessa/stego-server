#!/usr/bin/env python3
"""
Plaintext framing + per-token PRG mask for the Meteor-style stego server.

The server now receives a *plaintext message* plus an *encryption key* instead
of a pre-built ciphertext. Two pieces live here:

  * `make_mask_fn(key, precision)` -> the PRG that yields a fresh mask per token.
    This realizes Meteor's `PRG.Next` (paper Algorithm 5/6). Both encode and
    decode call it with the same key, so their mask streams stay synchronized.
    The mask is what re-randomizes the sampling value every token, removing the
    50%-boundary stall and giving the scheme its one-time-pad security.

  * `text_to_message_bits` / `message_bits_to_text` -> a self-describing frame so
    the decoder knows exactly where the message ends and can detect a wrong key.

Frame layout (before stego encoding), bytes:
    [ 4 : commit = sha256(utf8)[:4] ][ 2 : length N, big-endian ][ N : utf8 ]
Each byte is expanded LSB-first (matching bits2int/int2bits in the raw modules).
"""

import hmac
import hashlib

from raw_stego_encoder import int2bits, bits2int


def derive_prg_key(key) -> bytes:
    """Normalize an arbitrary key (hex string, str, or bytes) to 32 PRG bytes.

    Decode must use the identical key, so this must be deterministic."""
    if isinstance(key, bytes):
        raw = key
    else:
        raw = str(key).encode("utf-8")
    return hashlib.sha256(raw).digest()


def make_mask_fn(key, precision):
    """Return mask_fn(token_index) -> list of `precision` pseudorandom bits.

    Stateless and indexed by token: mask(t) = bits of HMAC-SHA256(prg_key, t).
    Indexing by token (rather than a running stream) makes encoder/decoder
    synchronization trivial and robust to 0-bit tokens."""
    prg_key = derive_prg_key(key)

    def mask_fn(token_index):
        bits = []
        counter = 0
        # 32 bytes (256 bits) per HMAC block; loop only if precision is huge.
        while len(bits) < precision:
            msg = (token_index & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "big") + \
                  counter.to_bytes(4, "big")
            block = hmac.new(prg_key, msg, hashlib.sha256).digest()
            for byte in block:
                bits.extend(int2bits(byte, 8))
            counter += 1
        return bits[:precision]

    return mask_fn


def text_to_message_bits(text: str):
    """Frame a plaintext string into a flat list of message bits."""
    data = text.encode("utf-8")
    if len(data) > 0xFFFF:
        raise ValueError("message too long (max 65535 UTF-8 bytes)")
    commit = hashlib.sha256(data).digest()[:4]
    header = commit + bytes([(len(data) >> 8) & 0xFF, len(data) & 0xFF])
    bits = []
    for byte in header + data:
        bits.extend(int2bits(byte, 8))
    return bits


def message_bits_to_text(bits):
    """Inverse of `text_to_message_bits`.

    Returns (text, integrity_ok). integrity_ok is False when the recovered bytes
    are too short, fail the commit check, or are not valid UTF-8 -- all strong
    signals of a wrong key or corrupted stegotext."""
    def byte_at(bit_offset):
        return bits2int(bits[bit_offset:bit_offset + 8])

    # 6-byte header (4 commit + 2 length) = 48 bits minimum
    if len(bits) < 48:
        return ("", False)
    commit = bytes(byte_at(o) for o in range(0, 32, 8))
    n = (byte_at(32) << 8) | byte_at(40)
    if len(bits) < 48 + n * 8:
        return ("", False)
    data = bytes(byte_at(48 + 8 * k) for k in range(n))
    ok = hashlib.sha256(data).digest()[:4] == commit
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return (data.decode("utf-8", errors="replace"), False)
    return (text, ok)
