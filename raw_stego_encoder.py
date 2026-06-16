#!/usr/bin/env python3
"""
Raw steganographic encoder using Llama-3 model
Encodes already-encrypted ciphertext into natural language text using arithmetic coding principles
No cryptographic keys or nonces required - works with pre-encrypted data
"""

import os
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

def encode_steganographic(model, tokenizer, message_bits, context_text,
                         temp=1.0, precision=16, topk=50000, verbose=False,
                         step_hook=None, mask_fn=None):
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
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive
    threshold = 1 / max_val
    
    output_tokens = context_tokens.clone()
    
    print(f"Encoding {len(message_bits)} bits into steganographic text...")
    
    if verbose:
        print("\n" + "="*80)
        print("VERBOSE TOKEN SELECTION DETAILS")
        print("="*80)
        print(f"Precision: {precision} bits, Max value: {max_val}")
        print(f"Temperature: {temp}, Top-k: {topk}")
        print("="*80)
        print()
    
    with torch.no_grad():
        i = 0
        token_index = 0
        while i < len(message_bits):
            # Get message bits for this iteration
            message_chunk = message_bits[i:i+precision]
            actual_bits = len(message_chunk)

            if actual_bits == 0:
                break  # No more bits to encode

            # Pad zeros to trailing bits of message chunk
            if actual_bits < precision:
                message_chunk = message_chunk + ([0] * (precision - actual_bits))

            # Snapshot the raw message chunk (pre-mask) for visualization
            original_chunk = list(message_chunk)

            # Meteor masking (paper Algorithm 5): XOR a fresh per-token mask into
            # the value used to sample from the distribution. Drawing a new mask
            # every token re-randomizes the sampling point even when the previous
            # step consumed 0 bits, which removes the 50%-boundary stall (no hack
            # needed) and gives the scheme its one-time-pad security. The PRG is
            # advanced exactly once per token so the decoder stays in lockstep.
            if mask_fn is not None:
                mask_bits = list(mask_fn(token_index))
                coding_chunk = [b ^ m for b, m in zip(message_chunk, mask_bits)]
            else:
                mask_bits = None
                coding_chunk = message_chunk

            # Get model predictions (disable caching for compatibility)
            outputs = model(output_tokens, use_cache=False)
            logits = outputs.logits[:, -1, :]  # Get logits for last token
            
            # Apply temperature and get probabilities
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=-1)
            
            # Sort by probability (descending)
            probs_temp_sorted, indices = torch.sort(probs_temp, dim=-1, descending=True)
            probs_temp_sorted = probs_temp_sorted.squeeze(0)
            indices = indices.squeeze(0)
            
            # Find cutoff point
            cutoff_mask = probs_temp_sorted >= threshold
            k = min(max(2, cutoff_mask.sum().item()), topk)
            
            # Take top-k tokens
            probs_temp_int = probs_temp_sorted[:k]
            indices = indices[:k]
            
            # Rescale to integer range
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * max_val
            
            # Round to integers
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)
            
            # Handle overflow
            overfill_index = (cum_probs > max_val).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                indices = indices[:overfill_index[0]]
            
            # Add missing mass to top
            if len(cum_probs) > 0:
                cum_probs += max_val - cum_probs[-1]
            
            # The (masked) value used to sample from the distribution
            message_idx = bits2int(list(reversed(coding_chunk)))
            
            # Find which cumulative probability bin contains our message index
            selection_idx = 0
            for j, cum_prob in enumerate(cum_probs):
                if cum_prob > message_idx:
                    selection_idx = j
                    break
            
            # Calculate interval boundaries
            new_int_bottom = cum_probs[selection_idx-1].item() if selection_idx > 0 else cur_interval[0]
            new_int_top = cum_probs[selection_idx].item()
            
            # Convert to bits and find common prefix
            new_int_bottom_bits = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits = list(reversed(int2bits(new_int_top-1, precision)))
            
            # Count bits that can be consumed. A straddling interval yields 0
            # here; with masking that is fine -- the next token draws a fresh
            # mask and re-randomizes, so the message still makes progress. (This
            # is where the old 50%-boundary hack used to live; it is gone.)
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits, new_int_top_bits)

            # Select the token
            selected_token = indices[selection_idx].unsqueeze(0).unsqueeze(0)
            
            # Update for next iteration
            output_tokens = torch.cat((output_tokens, selected_token), dim=1)
            
            # Verbose output for this token selection
            if verbose:
                selected_token_id = indices[selection_idx].item()
                selected_token_text = tokenizer.decode([selected_token_id])
                
                # Get the actual message bits that were encoded this step
                encoded_bits = original_chunk[:num_bits_encoded]
                
                # Calculate probability bounds
                prob_bottom = new_int_bottom / max_val
                prob_top = new_int_top / max_val
                prob_width = prob_top - prob_bottom
                
                # Get token rank and probability
                token_prob = probs_temp_sorted[selection_idx].item()
                
                # Calculate remaining bits
                bits_remaining = len(message_bits) - (i + num_bits_encoded)
                
                # Get the generated text so far (excluding context)
                generated_tokens_so_far = output_tokens[:, context_tokens.shape[1]:]
                generated_text_so_far = tokenizer.decode(generated_tokens_so_far[0], skip_special_tokens=True)
                
                token_number = output_tokens.shape[1] - context_tokens.shape[1]
                print(f"Token {token_number:3d}: '{selected_token_text}' (ID: {selected_token_id})")
                print(f"  Message bits: {encoded_bits}")
                print(f"  Bits encoded: {num_bits_encoded}/{actual_bits} (position {i} -> {i + num_bits_encoded})")
                print(f"  Bits remaining: {bits_remaining}/{len(message_bits)} ({bits_remaining/len(message_bits)*100:.1f}% left)")
                print(f"  Token rank: {selection_idx + 1}/{len(indices)} (prob: {token_prob:.6f})")
                print(f"  Interval: [{new_int_bottom}, {new_int_top}) = [{prob_bottom:.6f}, {prob_top:.6f}) width: {prob_width:.6f}")
                if mask_bits is not None:
                    print(f"  Mask: {''.join(map(str, mask_bits))}  ->  masked sampling value (idx): {message_idx}")
                else:
                    print(f"  Message index: {message_idx} (from bits: {list(reversed(coding_chunk))})")
                print(f"  Stego-note: {generated_text_so_far}")
                print()
            
            # GUI/step hook: surface everything about this step so a visualizer
            # can show the candidate tokens, their binary probability ranges,
            # the selection, and the bits actually consumed.
            if step_hook is not None:
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
                        "prob": probs_temp_sorted[j].item() if j < len(probs_temp_sorted) else 0.0,
                    })
                step_hook({
                    "step": token_index + 1,
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

            i += num_bits_encoded
            token_index += 1

            # Progress indicator (only if not verbose to avoid clutter)
            if not verbose and i % 50 == 0:
                print(f"Encoded {i}/{len(message_bits)} bits...")
    
    # Return only the generated tokens (excluding context)
    generated_tokens = output_tokens[:, context_tokens.shape[1]:]
    
    if verbose:
        print("="*80)
        print("ENCODING SUMMARY")
        print("="*80)
        print(f"Total bits encoded: {i}/{len(message_bits)}")
        print(f"Total tokens generated: {generated_tokens.shape[1]}")
        print(f"Bits per token (avg): {i/generated_tokens.shape[1]:.2f}")
        print("="*80)
        print()
    
    return generated_tokens

