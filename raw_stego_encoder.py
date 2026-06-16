#!/usr/bin/env python3
"""
Raw steganographic encoder using Llama-3 model
Encodes already-encrypted ciphertext into natural language text using arithmetic coding principles
No cryptographic keys or nonces required - works with pre-encrypted data
"""

import os
import secrets
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

def bits2int(bits):
    """Convert bit array to integer (LSB first)"""
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2**i)
    return res

def int2bits(inp, num_bits):
    """Convert integer to bit array (LSB first)"""
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]

def num_same_from_beg(bits1, bits2):
    """Count number of identical bits from the beginning"""
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i

def hex_to_bits(hex_string):
    """Convert hexadecimal string to bit array without length prefix"""
    # Remove any whitespace and ensure even length
    hex_string = hex_string.replace(' ', '').replace('\n', '').replace('\t', '')

    if len(hex_string) % 2 != 0:
        raise ValueError(f"Hex string must have even length, got {len(hex_string)} characters")

    # Convert hex string to bytes
    try:
        hex_bytes = bytes.fromhex(hex_string)
    except ValueError as e:
        raise ValueError(f"Invalid hexadecimal string: {e}")

    # Convert each byte to bits (LSB first for each byte)
    message_bits = []
    for byte in hex_bytes:
        byte_bits = int2bits(byte, 8)
        message_bits.extend(byte_bits)

    return message_bits


def _ends_sentence(piece):
    """Heuristic: does this decoded token text end a sentence? Only used as a
    fallback to trim a clean ending if the model never emits EOS before the
    safety cap -- the primary completion signal is the model's own EOS token."""
    s = piece.rstrip(" \t\"')]}>”’")
    return s.endswith((".", "!", "?", "\n"))


def _eos_token_ids(tokenizer, model):
    """Collect every token id that means 'generation is complete'. This is the
    standard way a generative model signals it is done (exactly what
    transformers' model.generate() stops on). Llama-3 Instruct in particular has
    several terminators (e.g. <|end_of_text|> and <|eot_id|>), so we gather them
    from both the tokenizer and the model's generation_config."""
    ids = set()
    for src in (getattr(tokenizer, "eos_token_id", None),
                getattr(getattr(model, "generation_config", None),
                        "eos_token_id", None)):
        if isinstance(src, int):
            ids.add(src)
        elif isinstance(src, (list, tuple, set)):
            ids.update(int(x) for x in src if x is not None)
    return ids


def encode_steganographic(model, tokenizer, message_bits, context_text,
                         temp=1.0, precision=16, topk=50000, verbose=False,
                         step_hook=None, mask_fn=None, complete_text=False,
                         max_cover_tokens=512):
    """
    Encode message bits into text using steganographic arithmetic coding

    Args:
        model: The language model
        tokenizer: The tokenizer
        message_bits: List of bits to encode
        context_text: Initial context string
        temp: Temperature for sampling
        precision: Precision for arithmetic coding
        topk: Top-k cutoff for vocabulary
        verbose: Show detailed token selection information
        step_hook: Optional callable(step_payload: dict). Invoked once per
            encoding step with everything needed to visualize that step
            (candidate tokens, their binary probability ranges, the selection,
            and the bits encoded). The call may block (e.g. the GUI waits for
            the user), which pauses encoding until it returns.
        mask_fn: Optional callable(token_index: int) -> list[int] returning
            `precision` mask bits for the given token. Implements Meteor's
            per-token PRG mask: the value used to sample is the message chunk
            XOR this mask. Must be the SAME function (same key) used by the
            decoder. If None, bits are encoded directly (insecure; can stall on
            a 50%-boundary token).
        complete_text: When True, keep sampling cover tokens AFTER the message
            has been fully encoded, so the output reads like a complete essay
            instead of stopping mid-thought. Meteor's Algorithm 5 stops the
            instant `n >= |m|`; this flag adds an honest-sampling continuation
            that runs until the model signals completion by emitting its EOS
            token (the standard generation-complete signal, exactly what
            transformers' model.generate() stops on). The continuation carries
            no message bits -- each cover token is sampled from fresh random
            coins, i.e. exactly an ordinary draw from the model -- so it stays
            indistinguishable from normal output and the decoder ignores it (the
            length-prefixed frame marks where the message ends). See
            STEGO_TEXT_COMPLETION.md.
        max_cover_tokens: Safety cap on cover tokens, bounding runtime if the
            model never emits EOS (common for non-instruct base models, which
            are trained to continue text indefinitely). If the cap is hit, the
            continuation is trimmed back to the last sentence boundary so the
            essay still ends cleanly rather than mid-word. Only used when
            complete_text.

    Returns:
        Generated text tokens (continuation of context)
    """
    # Tokenize context
    context_tokens = tokenizer.encode(context_text, return_tensors="pt")
    context_tokens = context_tokens.to(model.device)

    # Limit context length to avoid memory issues
    if context_tokens.shape[1] > 1022:
        context_tokens = context_tokens[:, -1022:]

    max_val = 2**precision
    threshold = 1 / max_val

    output_tokens = context_tokens.clone()
    eos_ids = _eos_token_ids(tokenizer, model)

    print(f"Encoding {len(message_bits)} bits into steganographic text...")
    if complete_text:
        print(f"complete_text=ON: continuing until the model emits EOS "
              f"{sorted(eos_ids) or '(none configured)'} "
              f"(safety cap {max_cover_tokens} cover tokens).")

    if verbose:
        print("\n" + "="*80)
        print("VERBOSE TOKEN SELECTION DETAILS")
        print("="*80)
        print(f"Precision: {precision} bits, Max value: {max_val}")
        print(f"Temperature: {temp}, Top-k: {topk}")
        print("="*80)
        print()

    def build_distribution():
        """Run the model on the current context and return the quantized
        cumulative-probability bins (cum_probs), the matching token ids
        (indices), and the sorted probabilities. Identical for message and
        cover tokens -- every token resets the interval to [0, max_val)."""
        outputs = model(output_tokens, use_cache=False)
        logits = outputs.logits[:, -1, :]  # logits for the last position
        logits_temp = logits / temp
        probs_temp = F.softmax(logits_temp, dim=-1)
        probs_sorted, idx = torch.sort(probs_temp, dim=-1, descending=True)
        probs_sorted = probs_sorted.squeeze(0)
        idx = idx.squeeze(0)
        cutoff_mask = probs_sorted >= threshold
        k = min(max(2, cutoff_mask.sum().item()), topk)
        probs_int = probs_sorted[:k]
        idx = idx[:k]
        probs_int = probs_int / probs_int.sum() * max_val
        probs_int = probs_int.round().long()
        cum = probs_int.cumsum(0)
        overfill_index = (cum > max_val).nonzero()
        if len(overfill_index) > 0:
            cum = cum[:overfill_index[0]]
            idx = idx[:overfill_index[0]]
        if len(cum) > 0:
            cum += max_val - cum[-1]
        return cum, idx, probs_sorted

    def build_candidates(cum_probs, indices, probs_sorted, selection_idx):
        """Shape up to 60 candidate rows for the step visualizer."""
        max_display = 60
        n_cand = len(cum_probs)
        show = list(range(min(n_cand, max_display)))
        if selection_idx not in show:
            show.append(selection_idx)
        candidates = []
        for j in show:
            lo = cum_probs[j - 1].item() if j > 0 else 0
            hi = cum_probs[j].item()
            lo_bits = list(reversed(int2bits(lo, precision)))
            hi_bits = list(reversed(int2bits(hi - 1, precision)))
            shared = num_same_from_beg(lo_bits, hi_bits)
            candidates.append({
                "rank": j,
                "word": tokenizer.decode([indices[j].item()]),
                "lo": lo,
                "hi": hi,
                "lo_bits": "".join(map(str, lo_bits)),
                "hi_bits": "".join(map(str, hi_bits)),
                "fixes": shared,
                "prefix": "".join(map(str, hi_bits[:shared])),
                "prob": probs_sorted[j].item() if j < len(probs_sorted) else 0.0,
            })
        return candidates, n_cand

    with torch.no_grad():
        i = 0                 # message bits consumed so far
        token_index = 0       # token counter (drives the PRG mask)
        cover_tokens = 0      # cover tokens added after the message
        last_boundary_len = None  # output length at the last sentence end (cover)
        hit_cover_cap = False     # True if cover stopped on the safety cap
        while True:
            in_message = i < len(message_bits)

            # Once the message is fully encoded, either stop (Meteor's default)
            # or keep going to complete the essay with honest cover sampling.
            if not in_message and not complete_text:
                break

            if in_message:
                # ---- message token: encode the next chunk of message bits ----
                message_chunk = message_bits[i:i+precision]
                actual_bits = len(message_chunk)
                if actual_bits < precision:
                    message_chunk = message_chunk + ([0] * (precision - actual_bits))
                original_chunk = list(message_chunk)
                # Meteor masking (Algorithm 5): XOR a fresh per-token mask into
                # the value used to sample. A new mask every token re-randomizes
                # the sampling point and removes the 50%-boundary stall.
                if mask_fn is not None:
                    mask_bits = list(mask_fn(token_index))
                    coding_chunk = [b ^ m for b, m in zip(message_chunk, mask_bits)]
                else:
                    mask_bits = None
                    coding_chunk = message_chunk
                message_idx = bits2int(list(reversed(coding_chunk)))
            else:
                # ---- cover token: sample honestly from fresh random coins -----
                # No message bits remain; r is drawn uniformly at random, which
                # is exactly an ordinary multinomial draw from the model. The
                # decoder never reads these (the frame's length prefix stops it).
                actual_bits = 0
                original_chunk = None
                mask_bits = None
                message_idx = secrets.randbelow(max_val)
                coding_chunk = list(reversed(int2bits(message_idx, precision)))

            cum_probs, indices, probs_sorted = build_distribution()

            # Find which cumulative-probability bin contains the sampling value
            selection_idx = 0
            for j in range(len(cum_probs)):
                if cum_probs[j].item() > message_idx:
                    selection_idx = j
                    break

            sel_id = indices[selection_idx].item()

            # The model signals it has finished by emitting EOS. This is the
            # standard completion criterion, so during cover sampling we treat it
            # as the essay being done and stop (without appending the EOS token).
            if not in_message and sel_id in eos_ids:
                if verbose:
                    print("Cover: model emitted EOS -> essay complete.\n")
                break

            # Calculate interval boundaries and the consumed prefix length
            new_int_bottom = cum_probs[selection_idx-1].item() if selection_idx > 0 else 0
            new_int_top = cum_probs[selection_idx].item()
            new_int_bottom_bits = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits = list(reversed(int2bits(new_int_top-1, precision)))
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits, new_int_top_bits)

            # Append the selected token to the running output
            selected_token = indices[selection_idx].unsqueeze(0).unsqueeze(0)
            output_tokens = torch.cat((output_tokens, selected_token), dim=1)

            # Verbose output for this token selection
            if verbose:
                selected_token_text = tokenizer.decode([sel_id])
                prob_bottom = new_int_bottom / max_val
                prob_top = new_int_top / max_val
                prob_width = prob_top - prob_bottom
                token_prob = probs_sorted[selection_idx].item()
                token_number = output_tokens.shape[1] - context_tokens.shape[1]
                if in_message:
                    encoded_bits = original_chunk[:num_bits_encoded]
                    bits_remaining = len(message_bits) - (i + num_bits_encoded)
                    print(f"Token {token_number:3d}: '{selected_token_text}' (ID: {sel_id})")
                    print(f"  Message bits: {encoded_bits}")
                    print(f"  Bits encoded: {num_bits_encoded}/{actual_bits} (position {i} -> {i + num_bits_encoded})")
                    print(f"  Bits remaining: {bits_remaining}/{len(message_bits)}")
                    print(f"  Token rank: {selection_idx + 1}/{len(indices)} (prob: {token_prob:.6f})")
                    print(f"  Interval: [{new_int_bottom}, {new_int_top}) = [{prob_bottom:.6f}, {prob_top:.6f}) width: {prob_width:.6f}")
                    if mask_bits is not None:
                        print(f"  Mask: {''.join(map(str, mask_bits))}  ->  masked sampling value (idx): {message_idx}")
                    else:
                        print(f"  Message index: {message_idx}")
                else:
                    print(f"Token {token_number:3d}: '{selected_token_text}' (ID: {sel_id})  [cover]")
                    print(f"  Cover token {cover_tokens + 1} (no message bits); random sampling value: {message_idx}")
                print()

            # GUI/step hook: surface everything about this step.
            if step_hook is not None:
                candidates, n_cand = build_candidates(
                    cum_probs, indices, probs_sorted, selection_idx)
                if in_message:
                    step_hook({
                        "step": token_index + 1,
                        "cover": False,
                        "precision": precision,
                        "message_bits": list(message_bits),
                        "pos": i,
                        "chunk": original_chunk,
                        "mask": list(mask_bits) if mask_bits is not None else None,
                        "coding_chunk": list(coding_chunk),
                        "selection_idx": selection_idx,
                        "num_bits_encoded": num_bits_encoded,
                        "encoded_bits": list(message_bits[i:i + num_bits_encoded]),
                        "candidates": candidates,
                        "total_candidates": n_cand,
                        "bits_before": i,
                        "bits_after": i + num_bits_encoded,
                        "total_message_bits": len(message_bits),
                    })
                else:
                    # Cover step: no message bits move, but show the honest draw.
                    done = len(message_bits)
                    step_hook({
                        "step": token_index + 1,
                        "cover": True,
                        "precision": precision,
                        "message_bits": list(message_bits),
                        "pos": done,
                        "chunk": list(coding_chunk),
                        "mask": None,
                        "coding_chunk": list(coding_chunk),
                        "selection_idx": selection_idx,
                        "num_bits_encoded": 0,
                        "encoded_bits": [],
                        "candidates": candidates,
                        "total_candidates": n_cand,
                        "bits_before": done,
                        "bits_after": done,
                        "total_message_bits": len(message_bits),
                    })

            token_index += 1

            if in_message:
                i += num_bits_encoded
                if not verbose and i > 0 and i % 50 == 0:
                    print(f"Encoded {i}/{len(message_bits)} bits...")
            else:
                cover_tokens += 1
                # Remember the last clean sentence end, so that if we hit the
                # safety cap (model never emitted EOS) we can trim back to it.
                if _ends_sentence(tokenizer.decode([sel_id])):
                    last_boundary_len = output_tokens.shape[1]
                if cover_tokens >= max_cover_tokens:
                    if verbose:
                        print(f"Cover: reached safety cap "
                              f"max_cover_tokens={max_cover_tokens} without EOS; "
                              f"stopping.\n")
                    hit_cover_cap = True
                    break

    # If the model never emitted EOS and we stopped on the safety cap, trim the
    # dangling partial sentence so the essay ends on a clean boundary.
    if hit_cover_cap and last_boundary_len is not None:
        if verbose:
            trimmed = output_tokens.shape[1] - last_boundary_len
            print(f"Trimming {trimmed} trailing token(s) back to the last "
                  f"sentence boundary for a clean ending.\n")
        output_tokens = output_tokens[:, :last_boundary_len]

    # Return only the generated tokens (excluding context)
    generated_tokens = output_tokens[:, context_tokens.shape[1]:]

    if verbose:
        print("="*80)
        print("ENCODING SUMMARY")
        print("="*80)
        print(f"Total message bits encoded: {i}/{len(message_bits)}")
        print(f"Cover tokens appended: {cover_tokens}")
        print(f"Total tokens generated: {generated_tokens.shape[1]}")
        if generated_tokens.shape[1] > 0:
            print(f"Bits per token (avg over all tokens): {i/generated_tokens.shape[1]:.2f}")
        print("="*80)
        print()

    return generated_tokens
