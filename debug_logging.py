#!/usr/bin/env python3
"""Console debug logging for the stego server (enabled with --debug).

When the server is started with ``--debug`` every /encode and /decode request is
narrated to the console: the full request body, the message-bit sequence as it
is built up *token by token*, the moment the message is fully hidden and the
encoder switches to writing honest cover text, and the final result.

This is purely diagnostic -- it does NOT change the stego output. It only
*observes*, by hanging off the per-token ``step_hook`` that ``encode_steganographic``
and ``decode_steganographic`` already expose (the same seam the GUI uses). That
keeps the raw encoder/decoder untouched and guarantees the logs reflect exactly
what the real code path did.
"""

from stego_codec import frame_is_complete


def bits_str(bits):
    """Render a bit list as a compact "010110..." string."""
    return "".join(str(b) for b in bits)


def chain_hooks(*hooks):
    """Combine several step hooks into one (skipping any that are None).

    Lets the debug hook coexist with the GUI hook: both get every payload.
    """
    hooks = [h for h in hooks if h is not None]
    if not hooks:
        return None
    if len(hooks) == 1:
        return hooks[0]

    def combined(payload):
        for hook in hooks:
            hook(payload)

    return combined


def _sep(char="="):
    print(char * 80)


def log_request(endpoint, data):
    """Dump every field of an incoming request body."""
    print()
    _sep()
    print(f"[DEBUG] {endpoint} request")
    _sep()
    if not isinstance(data, dict):
        print(f"  (non-dict body): {data!r}")
        return
    for key, value in data.items():
        if isinstance(value, str):
            print(f"  {key} (str, len={len(value)}): {value!r}")
        else:
            print(f"  {key}: {value!r}")


class EncodeDebugLogger:
    """Per-token step hook that narrates an /encode run.

    Tracks the running message-bit sequence, announces the switch from the
    hidden message to the cover-text continuation, and reports cover progress.
    """

    def __init__(self, total_message_bits, framed_bits):
        self.total = total_message_bits
        self.cover_started = False
        self.cover_count = 0
        print(f"[DEBUG][encode] framed message = {total_message_bits} bits "
              f"(4-byte commit + 2-byte length + payload)")
        print(f"[DEBUG][encode] full message bit sequence to embed:")
        print(f"               {bits_str(framed_bits)}")

    def hook(self, payload):
        if not payload.get("cover"):
            so_far = payload["bits_after"]
            bits = payload["message_bits"][:so_far]
            added = payload.get("num_bits_encoded", 0)
            print(f"[DEBUG][encode] token {payload['step']:>4}: "
                  f"+{added} bit(s)  ->  {so_far}/{self.total} message bits")
            print(f"               bits so far: {bits_str(bits)}")
        else:
            if not self.cover_started:
                self.cover_started = True
                print()
                _sep("-")
                print(f"[DEBUG][encode] >>> MESSAGE FULLY ENCODED "
                      f"({self.total}/{self.total} bits). Switching to cover "
                      f"text to complete the article. <<<")
                _sep("-")
            self.cover_count += 1
            print(f"[DEBUG][encode] cover token {self.cover_count} "
                  f"(carries no message bits)")


def log_encode_result(stego_text, response):
    """Log the final stego text and stats produced by /encode."""
    _sep()
    print("[DEBUG] /encode complete")
    _sep()
    stats = response.get("stats", {})
    print(f"  output tokens: {stats.get('output_tokens')}")
    print(f"  message bits embedded: {stats.get('message_bits')}")
    print(f"  starter_length: {response.get('starter_length')}")
    print(f"  final stego_text (len={len(stego_text)}):")
    print(f"    {stego_text!r}")
    _sep()
    print()


class DecodeDebugLogger:
    """Per-token step hook that narrates a /decode run.

    Accumulates the recovered bits, prints them after each token, and announces
    the point at which the length-prefixed frame is complete (after which the
    decoder ignores the remaining cover tokens).
    """

    def __init__(self):
        self.bits = []
        self.frame_done = False

    def hook(self, payload):
        self.bits.extend(payload.get("recovered_bits", []))
        word = payload.get("selected_word", "")
        added = payload.get("num_bits_encoded", 0)
        final = "  [final token]" if payload.get("is_final") else ""
        print(f"[DEBUG][decode] token {payload['step']:>4}: {word!r}  "
              f"+{added} bit(s)  ->  {len(self.bits)} bits total{final}")
        print(f"               bits so far: {bits_str(self.bits)}")
        if not self.frame_done and frame_is_complete(self.bits):
            self.frame_done = True
            print()
            _sep("-")
            print(f"[DEBUG][decode] >>> FRAMED MESSAGE COMPLETE at "
                  f"{len(self.bits)} bits. Remaining tokens are cover text and "
                  f"will be ignored. <<<")
            _sep("-")


def log_decode_result(message, integrity_ok, response):
    """Log the recovered message and integrity result produced by /decode."""
    _sep()
    print("[DEBUG] /decode complete")
    _sep()
    stats = response.get("stats", {})
    print(f"  recovered bits: {stats.get('recovered_bits')}")
    print(f"  integrity_ok: {integrity_ok}")
    print(f"  recovered message (len={len(message)}):")
    print(f"    {message!r}")
    if not integrity_ok:
        print("  NOTE: integrity check FAILED -- wrong key or corrupted stego text.")
    _sep()
    print()
