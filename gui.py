#!/usr/bin/env python3
"""
Optional step-by-step encoding visualizer for the steganographic server.

Enabled with `python main.py --add-gui`. The Flask server runs on a background
thread; this Tkinter GUI runs on the main thread (required on macOS). When an
/encode request arrives, the encoder calls back into `GuiBridge.hook` once per
step, which hands the step to the GUI and blocks until the user advances.

Flow per step:
  1. The model's candidate tokens and their binary probability ranges are shown
     (the selection is hidden). The user inspects, then clicks "Next".
  2. "Next" reveals the selected token and the bits actually encoded.
  3. "Next" again resumes the encoder, which computes the following step.

"Finish" fast-forwards without pausing and shows the final stego text.
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

        root.title("Meteor stego encoder — step visualizer")
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

        ttk.Label(body, text="Bit sequence being encoded "
                  "(grey = done, bold = current window, green = just encoded):"
                  ).pack(anchor="w")
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
                                 fg="#b00000", font=("TkDefaultFont", 12, "bold"))
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
            f"Waiting for an /encode request… "
            f"(POST to http://localhost:{self.port}/encode)")

    def _reset_counters(self, info):
        self.step_var.set("Steps: 0")
        self.bits_var.set("Bits encoded: 0")
        self._clear_step()
        self._set_output("")
        ct = info.get("ciphertext", "")
        st = info.get("start_text", "")
        self.status_var.set(f"Encoding started — ciphertext={ct!r}  "
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
        if self.bridge.is_finishing():
            self.bits_var.set(f"Bits encoded: {payload['bits_after']}")
            self.status_var.set("Fast-forwarding…")
            return
        self.current = payload
        self.phase = 1
        self._render_step(payload, reveal=False)
        self.next_btn.configure(state="normal", text="Next ▸  (reveal selection)")
        self.status_var.set(
            f"Step {payload['step']}: {payload['total_candidates']} candidate "
            f"token(s). Inspect the ranges, then click Next.")

    def _handle_end(self, final_text):
        self._clear_step()
        self.status_var.set("Encoding complete. Final stego text:")
        self._set_output(final_text)
        self._set_idle()
        # keep counters as-is so the user can read the totals

    # ---- rendering ------------------------------------------------------ #
    def _render_step(self, payload, reveal):
        self.bits_var.set(
            f"Bits encoded: {payload['bits_after'] if reveal else payload['bits_before']}")
        self._render_bits(payload, reveal)
        self._render_candidates(payload, reveal)
        if reveal:
            sel = payload["candidates_by_rank"].get(payload["selection_idx"])
            word = sel["word"] if sel else "?"
            enc = "".join(map(str, payload["encoded_bits"])) or "(none — 0 bits)"
            self.reveal_var.set(
                f"Selected token: {word!r}    →    encoded bits: {enc}    "
                f"({payload['num_bits_encoded']} bit(s))")
            if payload["hack_applied"]:
                self.hack_var.set(
                    "⚠ 50% boundary hack applied: sampling point was forced to "
                    "75%, so these encoded bits may not match the message bits.")
        else:
            self.reveal_var.set("Selection hidden — click Next to reveal.")
            self.hack_var.set("")

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
