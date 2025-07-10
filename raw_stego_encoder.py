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
                         temp=1.0, precision=16, topk=50000, verbose=False):
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
        while i < len(message_bits):
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
            
            # Cutoff low probabilities
            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range
            
            # Find cutoff point
            cutoff_mask = probs_temp_sorted >= cur_threshold
            if cutoff_mask.sum() == 0:
                k = 2  # Minimum 2 tokens
            else:
                k = min(max(2, cutoff_mask.sum().item()), topk)
            
            # Take top-k tokens
            probs_temp_int = probs_temp_sorted[:k]
            indices = indices[:k]
            
            # Rescale to integer range
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            
            # Round to integers
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)
            
            # Handle overflow
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                indices = indices[:overfill_index[0]]
            
            # Add missing mass to top
            if len(cum_probs) > 0:
                cum_probs += cur_int_range - cum_probs[-1]
            
            # Convert to position in range
            cum_probs += cur_interval[0]
            
            # Get message bits for this iteration (no masking - direct encoding)
            message_chunk = message_bits[i:i+precision]
            actual_bits = len(message_chunk)
            
            if actual_bits == 0:
                break  # No more bits to encode
            
            # Convert message bits to selection index (no padding)
            message_idx = bits2int(list(reversed(message_chunk)))
            
            # Scale the index to the current range based on actual bit count
            if actual_bits < precision:
                max_val_for_chunk = 2**actual_bits
                scale_factor = cur_int_range // max_val_for_chunk
                message_idx = message_idx * scale_factor
            
            # Special handling for 50% boundary conditions (patterns like [1,0,...])
            # Check if this maps to close to 50% of the current range. e.g. 1,0,0,0,1.
            is_fifty_percent = False
            if message_chunk[0] == 1 and all(bit == 0 for bit in message_chunk[1:4]):
                is_fifty_percent = True
                # Shift to 75% point to encode just the first '1' bit
                message_idx = cur_int_range * 3 // 4  # 75% point
                actual_bits = 1  # Only encode the first bit
                message_chunk = [1]  # Only process the first bit
            
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
            
            # Count bits that can be consumed
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits, new_int_top_bits)
            
            # Select the token
            selected_token = indices[selection_idx].unsqueeze(0).unsqueeze(0)
            
            # Update for next iteration
            output_tokens = torch.cat((output_tokens, selected_token), dim=1)
            
            # Verbose output for this token selection
            if verbose:
                selected_token_id = indices[selection_idx].item()
                selected_token_text = tokenizer.decode([selected_token_id])
                
                # Get the actual bits that were encoded
                if num_bits_encoded > 0:
                    encoded_bits = message_chunk[:num_bits_encoded]
                else:
                    encoded_bits = message_chunk
                
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
                if is_fifty_percent:
                    print(f"  Message index: {message_idx} (50% boundary detected - using 75% point for first bit)")
                elif actual_bits < precision:
                    print(f"  Message index: {message_idx} (scaled from {bits2int(list(reversed(message_chunk)))} for {actual_bits} bits)")
                else:
                    print(f"  Message index: {message_idx} (from bits: {list(reversed(message_chunk))})")
                print(f"  Stego-note: {generated_text_so_far}")
                print()
            
            i += num_bits_encoded
            
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

