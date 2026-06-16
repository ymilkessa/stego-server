#!/usr/bin/env python3
"""
Optional step-by-step encode/decode visualizer for the steganographic server.

Enabled with `python main.py --add-gui`. The Flask server runs on a background
thread; this Tkinter GUI runs on the main thread (required on macOS). When an
/encode or /decode request arrives (and the GUI is not already busy), the
encoder/decoder calls back into `GuiBridge.hook` once per token, which hands
the step to the GUI and blocks until the user advances. The same window serves
both directions; a single session lock guarantees one request is visualized at
a time (a concurrent request gets HTTP 409).

Encode flow per step:
  1. The model's candidate tokens and their binary probability ranges are shown
     together with the random bit value (message ⊕ per-token mask) that drives
     the selection. The selected token is hidden. The user inspects, clicks
     "Next".
  2. "Next" reveals which token that random value selected and the bits encoded.
  3. "Next" again resumes the encoder, which computes the following step.

Decode flow per step (the reverse): the full stego-text is shown at the top with
the current word highlighted. Each step shows the model's candidates; "Next"
reveals which candidate is the actual next word in the stego-text and how many
message bits that token choice recovers.

"Finish" fast-forwards without pausing and shows the final result (the stego
text when encoding, the recovered message when decoding).
"""

import queue
import threading

import tkinter as tk
from tkinter import ttk


# --------------------------------------------------------------------------- #
# Thread bridge between the Flask/encoder thread and the Tk main thread
# --------------------------------------------------------------------------- #
class GuiBridge:
    """Thread-safe channel: encoder thread -> GUI thread (a queue of events),
    GUI thread -> encoder thread (a 'proceed' event)."""

    def __init__(self):
        self.q = queue.Queue()
        self._proceed = threading.Event()
        self._finishing = threading.Event()
        # Only one interactive encode at a time.
        self.session_lock = threading.Lock()

    # ---- called from the encoder / Flask thread ------------------------- #
    def begin_session(self, info):
        self._finishing.clear()
        self._proceed.clear()
        self.q.put(("begin", info))

    def hook(self, payload):
        """Invoked by encode_steganographic once per step. Blocks until the
        user advances, unless we're fast-forwarding."""
        if self._finishing.is_set():
            self.q.put(("step", payload))
            return
        self._proceed.clear()
        self.q.put(("step", payload))
        self._proceed.wait()

    def end_session(self, final_text):
        self.q.put(("end", final_text))

    # ---- called from the GUI thread ------------------------------------- #
    def user_proceed(self):
        self._proceed.set()

    def user_finish(self):
        self._finishing.set()
        self._proceed.set()

    def is_finishing(self):
        return self._finishing.is_set()


# --------------------------------------------------------------------------- #
# The Tk application
# --------------------------------------------------------------------------- #
class StegoGui:
    POLL_MS = 80

    def __init__(self, root, bridge, port):
        self.root = root
        self.bridge = bridge
        self.port = port
        self.current = None     # current step payload awaiting user
        self.phase = 0          # 0 idle, 1 options shown, 2 selection revealed
        self.item_to_rank = {}
        self.mode = "encode"    # "encode" or "decode" — set on each session
        self._bits_word = "encoded"
        self.stego_text = ""    # decode: full stego text shown at the top
        self.starter_len = 0    # decode: chars of starter prefix (greyed out)
        self.decoded_bits = []  # decode: running list of recovered message bits

        root.title("Meteor stego — step visualizer (encode / decode)")
        root.geometry("980x720")
        self._build()
        self._set_idle()
        self.root.after(self.POLL_MS, self._poll)

    # ---- layout --------------------------------------------------------- #
    def _build(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        counters = ttk.Frame(top)
        counters.pack(side="left", anchor="w")
        self.step_var = tk.StringVar(value="Steps: 0")
        self.bits_var = tk.StringVar(value="Bits encoded: 0")
        ttk.Label(counters, textvariable=self.step_var,
                  font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(counters, textvariable=self.bits_var,
                  font=("TkDefaultFont", 13)).pack(anchor="w")

        self.finish_btn = ttk.Button(top, text="Finish ⏭  (fast-forward)",
                                     command=self._on_finish)
        self.finish_btn.pack(side="right", anchor="e")

        # Core body --------------------------------------------------------
        body = ttk.LabelFrame(self.root, text="Current encoding step", padding=8)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        self.top_label_var = tk.StringVar(
            value="Bit sequence being encoded "
            "(grey = done, bold = current window, green = just encoded):")
        ttk.Label(body, textvariable=self.top_label_var).pack(anchor="w")
        self.bits_text = tk.Text(body, height=4, wrap="char",
                                 font=("Menlo", 12))
        self.bits_text.pack(fill="x", pady=(2, 8))
        self.bits_text.tag_configure("done", foreground="#999999")
        self.bits_text.tag_configure("window", font=("Menlo", 12, "bold"))
        self.bits_text.tag_configure("just", foreground="#0a7d00",
                                     font=("Menlo", 12, "bold"))
        self.bits_text.configure(state="disabled")

        ttk.Label(body, text="Model output options — by probability "
                  "(probability range shown in binary):").pack(anchor="w")

        tree_frame = ttk.Frame(body)
        tree_frame.pack(fill="both", expand=True)
        cols = ("word", "prob", "low", "high", "fixes")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                 height=10)
        headings = {
            "word": ("token", 150),
            "prob": ("prob", 80),
            "low": ("range low (bin)", 200),
            "high": ("range high (bin)", 200),
            "fixes": ("would fix", 150),
        }
        for c, (label, width) in headings.items():
            self.tree.heading(c, text=label)
            self.tree.column(c, width=width, anchor="w")
        self.tree.tag_configure("selected", background="#fff2a8")
        vs = ttk.Scrollbar(tree_frame, orient="vertical",
                           command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")

        self.reveal_var = tk.StringVar(value="")
        self.reveal_lbl = ttk.Label(body, textvariable=self.reveal_var,
                                    font=("TkDefaultFont", 13))
        self.reveal_lbl.pack(anchor="w", pady=(8, 0))

        self.hack_var = tk.StringVar(value="")
        self.hack_lbl = tk.Label(body, textvariable=self.hack_var,
                                 fg="#3060a0", font=("Menlo", 11))
        self.hack_lbl.pack(anchor="w")

        self.next_btn = ttk.Button(body, text="Next ▸", command=self._on_next)
        self.next_btn.pack(anchor="e", pady=6)

        # Bottom status / output ------------------------------------------
        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        self.status_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.status_var,
                  foreground="#444").pack(anchor="w")
        self.output_text = tk.Text(bottom, height=5, wrap="word",
                                   font=("Menlo", 11))
        self.output_text.pack(fill="x", pady=(4, 0))
        self.output_text.configure(state="disabled")

    # ---- state helpers -------------------------------------------------- #
    def _set_idle(self):
        self.current = None
        self.phase = 0
        self.next_btn.configure(state="disabled", text="Next ▸")
        self.finish_btn.configure(state="disabled")
        self.status_var.set(
            f"Waiting for an /encode or /decode request… "
            f"(POST to http://localhost:{self.port})")

    def _reset_counters(self, info):
        self.mode = info.get("mode", "encode")
        self._bits_word = "decoded" if self.mode == "decode" else "encoded"
        self.step_var.set("Steps: 0")
        self.bits_var.set(f"Bits {self._bits_word}: 0")
        self.decoded_bits = []
        self._clear_step()
        self._set_output("")
        if self.mode == "decode":
            self.stego_text = info.get("stego_text", "")
            self.starter_len = len(info.get("start_text", ""))
            self.top_label_var.set(
                "Stego-text being decoded "
                "(grey = starter, bold = current word, green = just decoded):")
            preview = self.stego_text[:60]
            self.status_var.set(f"Decoding started — stego_text={preview!r}…")
        else:
            self.top_label_var.set(
                "Bit sequence being encoded "
                "(grey = done, bold = current window, green = just encoded):")
            msg = info.get("message", "")
            st = info.get("start_text", "")
            self.status_var.set(f"Encoding started — message={msg!r}  "
                                f"start_text={st!r}")
        self.finish_btn.configure(state="normal")

    def _clear_step(self):
        self.bits_text.configure(state="normal")
        self.bits_text.delete("1.0", "end")
        self.bits_text.configure(state="disabled")
        self.tree.delete(*self.tree.get_children())
        self.item_to_rank.clear()
        self.reveal_var.set("")
        self.hack_var.set("")

    def _set_output(self, text):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")

    # ---- queue polling -------------------------------------------------- #
    def _poll(self):
        try:
            while True:
                kind, data = self.bridge.q.get_nowait()
                self._handle(kind, data)
        except queue.Empty:
            pass
        self.root.after(self.POLL_MS, self._poll)

    def _handle(self, kind, data):
        if kind == "begin":
            self._reset_counters(data)
        elif kind == "step":
            self._handle_step(data)
        elif kind == "end":
            self._handle_end(data)

    def _handle_step(self, payload):
        # Keep counters live even while fast-forwarding.
        self.step_var.set(f"Steps: {payload['step']}")
        # The hook fires exactly once per token; accumulate the recovered bits
        # here so they stream into the bottom pane (and survive fast-forward).
        if self.mode == "decode":
            self.decoded_bits.extend(payload.get("recovered_bits", []))
        if self.bridge.is_finishing():
            self.bits_var.set(f"Bits {self._bits_word}: {payload['bits_after']}")
            self.status_var.set("Fast-forwarding…")
            if self.mode == "decode":
                self._show_decoded_bits(payload['bits_after'])
            return
        self.current = payload
        self.phase = 1
        self._render_step(payload, reveal=False)
        if self.mode == "decode":
            self.next_btn.configure(state="normal",
                                    text="Next ▸  (reveal next word)")
            self.status_var.set(
                f"Step {payload['step']}: {payload['total_candidates']} candidate "
                f"token(s). Which is the next word? Click Next to reveal.")
        else:
            self.next_btn.configure(state="normal",
                                    text="Next ▸  (reveal selection)")
            self.status_var.set(
                f"Step {payload['step']}: {payload['total_candidates']} candidate "
                f"token(s). Inspect the ranges + random value, then click Next.")

    def _handle_end(self, final_text):
        self._clear_step()
        if self.mode == "decode":
            self.status_var.set("Decoding complete. Recovered message:")
        else:
            self.status_var.set("Encoding complete. Final stego text:")
        self._set_output(final_text)
        self._set_idle()
        # keep counters as-is so the user can read the totals

    # ---- rendering ------------------------------------------------------ #
    def _render_step(self, payload, reveal):
        self.bits_var.set(
            f"Bits {self._bits_word}: "
            f"{payload['bits_after'] if reveal else payload['bits_before']}")
        if self.mode == "decode":
            self._render_stego(payload, reveal)
        else:
            self._render_bits(payload, reveal)
        self._render_candidates(payload, reveal)
        if self.mode == "decode":
            self._render_reveal_decode(payload, reveal)
            # Bottom pane: bits recovered so far. Before the reveal, exclude the
            # current (still-hidden) token; on reveal, include it.
            n = payload["bits_after"] if reveal else payload["bits_before"]
            self._show_decoded_bits(n)
        else:
            self._render_reveal_encode(payload, reveal)

    def _show_decoded_bits(self, n):
        """Decode mode: write the first `n` recovered bits to the bottom pane,
        grouped into bytes for readability."""
        bits = self.decoded_bits[:n]
        grouped = " ".join("".join(map(str, bits[i:i + 8]))
                           for i in range(0, len(bits), 8))
        self._set_output(f"Decoded bits so far ({len(bits)}):\n{grouped}")

    def _render_reveal_encode(self, payload, reveal):
        # The random value (message ⊕ per-token mask) drives the selection, so
        # it is shown in BOTH phases — only the resulting token is hidden until
        # the user clicks Next.
        if payload.get("mask") is not None:
            self.hack_var.set(
                f"random value sampled (message ⊕ mask): "
                f"{''.join(map(str, payload['coding_chunk']))}   "
                f"[mask: {''.join(map(str, payload['mask']))}]")
        else:
            self.hack_var.set("")
        if reveal:
            sel = payload["candidates_by_rank"].get(payload["selection_idx"])
            word = sel["word"] if sel else "?"
            enc = "".join(map(str, payload["encoded_bits"])) or "(none — 0 bits)"
            self.reveal_var.set(
                f"Selected token: {word!r}    →    encoded message bits: {enc}    "
                f"({payload['num_bits_encoded']} bit(s))")
        else:
            self.reveal_var.set("Selected token hidden — click Next to reveal "
                                "which token the random value picks.")

    def _render_reveal_decode(self, payload, reveal):
        # The per-token decoding mask is key-derived (independent of which token
        # comes next), so it is shown in BOTH phases. On reveal we also show the
        # full unmasking: the bits read off the token ⊕ mask = message bits.
        mask = payload.get("mask")
        if mask is None:
            self.hack_var.set("")
        elif reveal:
            masked = "".join(map(str, payload.get("masked_bits", [])))
            msg = "".join(map(str, payload["recovered_bits"]))
            applied = "".join(map(str, mask))[:len(masked)]
            self.hack_var.set(
                f"unmask: read {masked or '∅'} ⊕ mask {applied or '∅'} = "
                f"message {msg or '∅'}   "
                f"[full token mask: {''.join(map(str, mask))}]")
        else:
            self.hack_var.set(
                f"decoding mask (key-derived, token {payload['step']}): "
                f"{''.join(map(str, mask))}")
        if reveal:
            bits = "".join(map(str, payload["recovered_bits"])) or "(none — 0 bits)"
            extra = " (final token — all remaining bits)" if payload.get("is_final") else ""
            self.reveal_var.set(
                f"Next word: {payload.get('selected_word', '?')!r} "
                f"(rank {payload['selection_idx'] + 1})    →    "
                f"decoded {len(payload['recovered_bits'])} bit(s){extra}: {bits}")
        else:
            self.reveal_var.set("Next word hidden — click Next to reveal which "
                                "candidate is the next word in the stego-text.")

    def _render_stego(self, payload, reveal):
        """Decode mode: show the full stego-text with the current word marked."""
        text = self.stego_text
        self.bits_text.configure(state="normal")
        self.bits_text.delete("1.0", "end")
        self.bits_text.insert("1.0", text)
        if self.starter_len:
            self.bits_text.tag_add("done", "1.0", f"1.0+{self.starter_len}c")
        start = payload.get("done_len", self.starter_len)
        wlen = payload.get("word_len", 0)
        if wlen:
            tag = "just" if reveal else "window"
            self.bits_text.tag_add(tag, f"1.0+{start}c", f"1.0+{start + wlen}c")
        self.bits_text.configure(state="disabled")

    def _render_bits(self, payload, reveal):
        bits = payload["message_bits"]
        pos = payload["bits_before"]
        after = payload["bits_after"]
        win_end = pos + len(payload["chunk"])
        self.bits_text.configure(state="normal")
        self.bits_text.delete("1.0", "end")
        for idx, b in enumerate(bits):
            if idx < pos:
                tag = "done"
            elif reveal and pos <= idx < after:
                tag = "just"
            elif idx < win_end:
                tag = "window"
            else:
                tag = ()
            self.bits_text.insert("end", str(b), tag)
        self.bits_text.configure(state="disabled")

    def _render_candidates(self, payload, reveal):
        self.tree.delete(*self.tree.get_children())
        self.item_to_rank.clear()
        by_rank = {}
        for c in payload["candidates"]:
            by_rank[c["rank"]] = c
            fixes = (f"{c['fixes']} bit(s): {c['prefix']}"
                     if c["fixes"] else "0 bits (straddles)")
            item = self.tree.insert(
                "", "end",
                values=(c["word"], f"{c['prob']:.4f}",
                        c["lo_bits"], c["hi_bits"], fixes))
            self.item_to_rank[item] = c["rank"]
            if reveal and c["rank"] == payload["selection_idx"]:
                self.tree.item(item, tags=("selected",))
                self.tree.selection_set(item)
                self.tree.see(item)
        payload["candidates_by_rank"] = by_rank

    # ---- buttons -------------------------------------------------------- #
    def _on_next(self):
        if self.current is None:
            return
        if self.phase == 1:
            self.phase = 2
            self._render_step(self.current, reveal=True)
            self.next_btn.configure(text="Next ▸  (next step)")
            if self.mode == "decode":
                self.status_var.set("Next word revealed. Click Next for the "
                                    "following token.")
            else:
                self.status_var.set("Selection revealed. Click Next for the "
                                    "following step.")
        else:
            self.next_btn.configure(state="disabled", text="…")
            self.current = None
            self.phase = 0
            self.bridge.user_proceed()

    def _on_finish(self):
        self.finish_btn.configure(state="disabled")
        self.next_btn.configure(state="disabled", text="…")
        self.current = None
        self.phase = 0
        self.status_var.set("Fast-forwarding…")
        self.bridge.user_finish()


def run_gui(bridge, port):
    """Build the Tk app and run its mainloop (blocks; call on the main thread)."""
    root = tk.Tk()
    StegoGui(root, bridge, port)
    root.mainloop()
