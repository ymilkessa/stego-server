#!/usr/bin/env python3
"""
Raw steganographic decoder using Llama-3 model
Decodes already-encrypted ciphertext from natural language text using arithmetic coding principles
No cryptographic keys or nonces required - works with pre-encrypted data
"""

import os
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

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

def bits_to_hex(bits):
    """Convert bit array to hexadecimal string without length prefix handling"""
    if len(bits) == 0:
        print("Warning: No bits to decode")
        return ""
    
    # Use all available bits (may include garbage bits at the end)
    message_bits = bits
    
    # Pad message bits to multiple of 8
    while len(message_bits) % 8 != 0:
        message_bits.append(0)
    
    print(f"Converting {len(message_bits)} bits to hex string ({len(message_bits)//8} bytes)")
    
    # Convert each 8-bit group to a byte
    bytes_data = []
    for i in range(0, len(message_bits), 8):
        byte_bits = message_bits[i:i+8]
        byte_value = bits2int(byte_bits)
        bytes_data.append(byte_value)
    
    # Always return as hex string (to match encoder input format)
    return ''.join(f'{b:02x}' for b in bytes_data)

def setup_model():
    """Initialize the Llama-3 model and tokenizer"""
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not found in environment variables")
    
    print("Loading Llama-3 model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Model loaded on device: {model.device}")
    return model, tokenizer

def decode_steganographic(model, tokenizer, stego_text, context_text, 
                         temp=1.0, precision=16, topk=50000, verbose=False):
    """
    Decode message bits from steganographic text using arithmetic coding
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        stego_text: The steganographic text to decode
        context_text: Initial context string (same as used in encoding)
        temp: Temperature for sampling (same as used in encoding)
        precision: Precision for arithmetic coding (same as used in encoding)
        topk: Top-k cutoff for vocabulary (same as used in encoding)
        verbose: Show detailed token decoding information
    
    Returns:
        List of decoded message bits
    """
    # Tokenize the full steganographic text
    full_tokens = tokenizer.encode(stego_text, return_tensors="pt")
    full_tokens = full_tokens.to(model.device)
    
    # Tokenize just the context to find where generated text starts
    context_tokens = tokenizer.encode(context_text, return_tensors="pt")
    context_tokens = context_tokens.to(model.device)
    
    # Limit context length to avoid memory issues
    if context_tokens.shape[1] > 1022:
        context_tokens = context_tokens[:, -1022:]
    
    # Extract the generated tokens (everything after context)
    if full_tokens.shape[1] <= context_tokens.shape[1]:
        print("Warning: Steganographic text appears to be shorter than or equal to context")
        return []
    
    generated_tokens = full_tokens[:, context_tokens.shape[1]:]
    
    print(f"Context tokens: {context_tokens.shape[1]}")
    print(f"Generated tokens to decode: {generated_tokens.shape[1]}")
    
    max_val = 2**precision
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive
    
    message_bits = []
    
    print("Decoding steganographic text...")
    
    if verbose:
        print("\n" + "="*80)
        print("VERBOSE TOKEN DECODING DETAILS")
        print("="*80)
        print(f"Precision: {precision} bits, Max value: {max_val}")
        print(f"Temperature: {temp}, Top-k: {topk}")
        print("="*80)
        print()
    
    with torch.no_grad():
        for token_idx in range(generated_tokens.shape[1]):
            # Reconstruct context up to current position
            current_context = torch.cat([context_tokens, generated_tokens[:, :token_idx]], dim=1)
            
            # Get model predictions for the current position (disable caching for compatibility)
            outputs = model(current_context, use_cache=False)
            logits = outputs.logits[:, -1, :]  # Get logits for last token
            
            # Apply temperature and get probabilities
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=-1)
            
            # Sort by probability (descending)
            probs_temp_sorted, indices = torch.sort(probs_temp, dim=-1, descending=True)
            probs_temp_sorted = probs_temp_sorted.squeeze(0)
            indices = indices.squeeze(0)
            
            # Cutoff low probabilities (same logic as encoder)
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
                k = overfill_index[0].item()
            
            # Add missing mass to top
            if len(cum_probs) > 0:
                cum_probs += cur_int_range - cum_probs[-1]
            
            # Convert to position in range
            cum_probs += cur_interval[0]
            
            # Find which token was actually selected
            actual_token = generated_tokens[0, token_idx].item()
            
            # Find the rank of the actual token in our probability-sorted list
            try:
                selection_idx = (indices == actual_token).nonzero()[0].item()
            except IndexError:
                # Token not found in top-k, this is an error condition
                print(f"Warning: Token {actual_token} not found in top-{k} at position {token_idx}")
                # Try to handle gracefully by using rank 0
                selection_idx = 0
            
            # Calculate interval boundaries
            new_int_bottom = cum_probs[selection_idx-1].item() if selection_idx > 0 else cur_interval[0]
            new_int_top = cum_probs[selection_idx].item()
            
            # Convert to bits and find common prefix
            new_int_bottom_bits = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits = list(reversed(int2bits(new_int_top-1, precision)))
            
            # Count bits that were encoded
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits, new_int_top_bits)
            
            # Extract the bits that were encoded (no unmasking - direct decoding)
            if token_idx == generated_tokens.shape[1] - 1:  # Last token
                # For the last token, we need to determine how many bits were actually encoded
                # This is tricky without knowing the exact original bit count
                # For now, use all bottom bits (this may include some extra bits)
                recovered_bits = new_int_bottom_bits
            else:
                # For other tokens, use only the fixed bits
                recovered_bits = new_int_top_bits[:num_bits_encoded]
            
            # Verbose output for this token decoding
            if verbose:
                actual_token_text = tokenizer.decode([actual_token])
                
                # Calculate probability bounds
                prob_bottom = new_int_bottom / max_val
                prob_top = new_int_top / max_val
                prob_width = prob_top - prob_bottom
                
                # Get token rank and probability
                token_prob = probs_temp_sorted[selection_idx].item() if selection_idx < len(probs_temp_sorted) else 0.0
                
                is_final_token = token_idx == generated_tokens.shape[1] - 1
                
                print(f"Token {token_idx + 1:3d}: '{actual_token_text}' (ID: {actual_token})")
                print(f"  Token rank: {selection_idx + 1}/{k} (prob: {token_prob:.6f})")
                print(f"  Interval: [{new_int_bottom}, {new_int_top}) = [{prob_bottom:.6f}, {prob_top:.6f}) width: {prob_width:.6f}")
                print(f"  Recovered bits: {recovered_bits}")
                if is_final_token:
                    print(f"  Final token: using all bottom bits (may include extra bits)")
                print(f"  Bits recovered: {len(recovered_bits)} (total so far: {len(message_bits) + len(recovered_bits)})")
                print()
            
            # Add the recovered bits to our message
            message_bits.extend(recovered_bits)
            
            # Progress indicator (only if not verbose to avoid clutter)
            if not verbose and (token_idx + 1) % 10 == 0:
                print(f"Decoded token {token_idx + 1}/{generated_tokens.shape[1]}, recovered {len(message_bits)} bits so far...")
    
    print(f"Total recovered bits: {len(message_bits)}")
    
    if verbose:
        print("="*80)
        print("DECODING SUMMARY")
        print("="*80)
        print(f"Total bits decoded: {len(message_bits)}")
        print(f"Total tokens processed: {generated_tokens.shape[1]}")
        print(f"Bits per token (avg): {len(message_bits)/generated_tokens.shape[1]:.2f}")
        print("="*80)
        print()
    
    return message_bits
