#!/usr/bin/env python3
"""
Steganographic HTTP Server
Provides REST API endpoints for encoding and decoding steganographic text
"""

import os
import argparse
import threading
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Default model ID - can be overridden by requests
# default_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# default_model_id = "openai-community/gpt2"
default_model_id = "meta-llama/Llama-3.2-1B"

# Internal steering prompt prepended to the MODEL context (never shown to the
# user). Base models like Llama-3.2-1B tend to ramble forever without emitting
# EOS; this nudges the model to actually finish so the cover-text completion
# terminates promptly. Both /encode and /decode prepend the IDENTICAL string so
# the model conditions on exactly the same context in each direction. It is
# never part of the returned stego text and is excluded from every length and
# offset (it is passed via the encoder/decoder `context_prefix` argument, which
# keeps all user-facing accounting relative to the un-prefixed text).
INTERNAL_PROMPT = "Complete this very short article in less than five paragraphs.\n\n"

# Import the encoding/decoding functions from the raw modules
from raw_stego_encoder import encode_steganographic
from raw_stego_decoder import decode_steganographic
from stego_codec import (make_mask_fn, text_to_message_bits,
                         message_bits_to_text, frame_is_complete)
from debug_logging import (chain_hooks, log_request, EncodeDebugLogger,
                          log_encode_result, DecodeDebugLogger,
                          log_decode_result)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model cache - stores loaded models to avoid reloading
model_cache = {}

# Server-side debugging flag - set to True when you want to debug
verbose = False

# --debug request logging flag (set when started with --debug). When True, each
# /encode and /decode request is narrated to the console: the full request body,
# the message-bit sequence built up token by token, the switch to cover text,
# and the final output. See debug_logging.py. Distinct from `verbose` (raw
# per-token model internals) and from Flask's own DEBUG reloader.
DEBUG_LOG = False

# Optional step-by-step GUI bridge (set when started with --add-gui)
GUI_BRIDGE = None


def set_gui_bridge(bridge):
    """Register the GUI bridge so /encode drives the step visualizer."""
    global GUI_BRIDGE
    GUI_BRIDGE = bridge


def set_debug_log(enabled):
    """Enable/disable the --debug request logging."""
    global DEBUG_LOG
    DEBUG_LOG = enabled


def get_model(model_id=None):
    """Get model and tokenizer, loading and caching if necessary
    
    Args:
        model_id: Hugging Face model identifier. If None, uses default_model_id
    
    Returns:
        tuple: (model, tokenizer)
    """
    if model_id is None:
        model_id = default_model_id
    
    # Check if model is already cached
    if model_id in model_cache:
        print(f"Using cached model: {model_id}")
        return model_cache[model_id]
    
    # Load model if not in cache
    print(f"Loading model: {model_id}")
    
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not found in environment variables")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Model loaded on device: {model.device}")
    
    # Cache the model and tokenizer
    model_cache[model_id] = (model, tokenizer)
    print(f"Model cached: {model_id}")
    
    return model, tokenizer


def preload_default_model():
    """Preload the default model at startup"""
    try:
        print("Preloading default model...")
        get_model(default_model_id)
        print("Default model preloaded successfully!")
    except Exception as e:
        print(f"Warning: Could not preload default model: {e}")
        print("Model will be loaded on first request.")


def get_cache_status():
    """Get information about cached models"""
    return {
        "cached_models": list(model_cache.keys()),
        "cache_size": len(model_cache),
        "default_model_cached": default_model_id in model_cache
    }


@app.route('/encode', methods=['POST'])
def encode_endpoint():
    """
    Encode endpoint: POST /encode

    The server encrypts the plaintext itself: it derives a per-token PRG mask
    from `key` (Meteor's PRG.Next) and XOR-masks the message as it samples
    tokens. Send the plaintext and the shared key, NOT a pre-built ciphertext.

    JSON Body:
    {
        "message": "Attack at dawn",         // plaintext to hide
        "key": "3f9a...",                     // shared encryption key (hex string)
        "start_text": "Hello world...",       // starting text
        "model_id": "meta-llama/Llama-3.2-1B",  // optional, default from server
        "temp": 1.2,                          // optional, default 1.2
        "precision": 16,                      // optional, default 16
        "topk": 50000,                        // optional, default 50000
        "complete_essay": 1                   // optional; 1 = keep writing until
                                              //   EOS (full article). Absent or
                                              //   != 1 (default) = stop as soon
                                              //   as the message is encoded.
    }

    Response:
    {
        "success": true,
        "stego_text": "Hello world companies like...",
        "starter_length": 25,                 // length of starting text in characters
        "config": { "model_id": ..., "temp": 1.2, "precision": 16, "topk": 50000, "complete_essay": false },
        "stats": {
            "message_chars": 14,
            "message_bytes": 14,
            "message_bits": 160,              // includes 6-byte commit+length frame
            "output_tokens": 42
        }
    }
    """
    try:
        # Parse JSON request
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        if DEBUG_LOG:
            log_request('/encode', data)

        # Extract required parameters
        message = data.get('message')
        key = data.get('key')
        start_text = data.get('start_text')

        if message is None:
            return jsonify({"success": False, "error": "Missing required parameter: message"}), 400
        if not key:
            return jsonify({"success": False, "error": "Missing required parameter: key"}), 400
        if not start_text:
            return jsonify({"success": False, "error": "Missing required parameter: start_text"}), 400

        # Extract optional parameters
        model_id = data.get('model_id', default_model_id)
        temp = data.get('temp', 1.2)
        precision = data.get('precision', 16)
        topk = data.get('topk', 50000)

        # complete_essay: only when explicitly set to 1 does the encoder keep
        # sampling cover tokens until the model emits EOS (a full article, which
        # can be slow). Absent or anything other than 1 -> stop the moment the
        # message is fully encoded and return just that text (the fast default).
        complete_text = str(data.get('complete_essay')) == '1'

        # Get model from cache (loads if not cached)
        model, tokenizer = get_model(model_id)

        # Frame the plaintext (commit + length + bytes) and build the PRG mask.
        message_bits = text_to_message_bits(message)
        mask_fn = make_mask_fn(key, precision)

        # When the GUI is attached, drive it step-by-step (one encode at a time).
        gui = GUI_BRIDGE
        step_hook = None
        if gui is not None:
            if not gui.session_lock.acquire(blocking=False):
                return jsonify({"success": False,
                                "error": "GUI is busy visualizing another encode"}), 409
            gui.begin_session({"mode": "encode", "message": message,
                               "start_text": start_text})
            step_hook = gui.hook

        # Attach the debug step hook (alongside the GUI hook if present).
        if DEBUG_LOG:
            enc_logger = EncodeDebugLogger(len(message_bits), message_bits)
            step_hook = chain_hooks(step_hook, enc_logger.hook)

        try:
            # Encode steganographically using imported function
            print(f"Encoding steganographic text...")
            generated_tokens = encode_steganographic(
                model, tokenizer, message_bits, start_text,
                temp=temp, precision=precision, topk=topk, verbose=verbose,
                step_hook=step_hook, mask_fn=mask_fn, complete_text=complete_text,
                context_prefix=INTERNAL_PROMPT
            )
            print(f"Just finished encoding steganographic text...")

            # Decode and create full steganographic text
            generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            full_stego_text = start_text + generated_text

            if gui is not None:
                gui.end_session(full_stego_text)
        finally:
            if gui is not None:
                gui.session_lock.release()

        # Prepare response
        response = {
            "success": True,
            "stego_text": full_stego_text,
            "starter_length": len(start_text),
            "config": {
                "model_id": model_id,
                "temp": temp,
                "precision": precision,
                "topk": topk,
                "complete_essay": complete_text
            },
            "stats": {
                "message_chars": len(message),
                "message_bytes": len(message.encode('utf-8')),
                "message_bits": len(message_bits),
                "output_tokens": generated_tokens.shape[1]
            }
        }

        print(f"Response: {response}")

        if DEBUG_LOG:
            log_encode_result(full_stego_text, response)

        return jsonify(response)

    except Exception as e:
        # Always surface the full traceback to the console: a crash mid-encode
        # (e.g. the model hitting EOS early -> degenerate distribution) is
        # exactly what we most need to see when debugging.
        print("\n" + "!" * 80)
        print(f"[encode] ENCODING FAILED: {type(e).__name__}: {e}")
        print("!" * 80)
        traceback.print_exc()
        print("!" * 80 + "\n")
        error_msg = f"Encoding error: {str(e)}"
        if verbose or DEBUG_LOG:
            error_msg += f"\n{traceback.format_exc()}"
        return jsonify({"success": False, "error": error_msg}), 500

@app.route('/decode', methods=['POST'])
def decode_endpoint():
    """
    Decode endpoint: POST /decode

    Recovers the plaintext directly. Pass the SAME `key` used to encode; the
    server regenerates the per-token mask and unmasks the recovered bits.

    JSON Body:
    {
        "stego_text": "Hello world companies like...",  // full steganographic text
        "starter_length": 25,                            // number of characters in starting text
        "key": "3f9a...",                                // same shared key used to encode
        "model_id": "meta-llama/Llama-3.2-1B",  // optional, default from server
        "temp": 1.2,                                     // optional, default 1.2
        "precision": 16,                                 // optional, default 16
        "topk": 50000                                    // optional, default 50000
    }

    Response:
    {
        "success": true,
        "message": "Attack at dawn",   // recovered plaintext
        "integrity_ok": true,          // false => wrong key or corrupted stegotext
        "config": { "model_id": ..., "temp": 1.2, "precision": 16, "topk": 50000 },
        "stats": { "input_length": 245, "generated_tokens": 42, "recovered_bits": 320 }
    }
    """
    try:
        # Parse JSON request
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        if DEBUG_LOG:
            log_request('/decode', data)

        # Extract required parameters
        stego_text = data.get('stego_text')
        starter_length = data.get('starter_length')
        key = data.get('key')

        if not stego_text:
            return jsonify({"success": False, "error": "Missing required parameter: stego_text"}), 400
        if starter_length is None:
            return jsonify({"success": False, "error": "Missing required parameter: starter_length"}), 400
        if not key:
            return jsonify({"success": False, "error": "Missing required parameter: key"}), 400

        # Extract optional parameters
        model_id = data.get('model_id', default_model_id)
        temp = data.get('temp', 1.2)
        precision = data.get('precision', 16)
        topk = data.get('topk', 50000)

        # Validate starter_length
        if not isinstance(starter_length, int) or starter_length < 1:
            return jsonify({"success": False, "error": "starter_length must be a positive integer"}), 400
        if starter_length >= len(stego_text):
            return jsonify({"success": False, "error": "starter_length must be less than the total stego_text length"}), 400

        # Extract start_text using the provided starter_length
        start_text = stego_text[:starter_length]

        # Get model from cache (loads if not cached)
        model, tokenizer = get_model(model_id)

        # Decode steganographically, undoing the per-token mask with the key.
        mask_fn = make_mask_fn(key, precision)

        # When the GUI is attached, drive it step-by-step (one request at a time,
        # shared with /encode via the same session lock).
        gui = GUI_BRIDGE
        step_hook = None
        if gui is not None:
            if not gui.session_lock.acquire(blocking=False):
                return jsonify({"success": False,
                                "error": "GUI is busy visualizing another request"}), 409
            gui.begin_session({"mode": "decode", "stego_text": stego_text,
                               "start_text": start_text})
            step_hook = gui.hook

        # Attach the debug step hook (alongside the GUI hook if present).
        if DEBUG_LOG:
            dec_logger = DecodeDebugLogger()
            step_hook = chain_hooks(step_hook, dec_logger.hook)

        try:
            recovered_bits = decode_steganographic(
                model, tokenizer, stego_text, start_text,
                temp=temp, precision=precision, topk=topk, verbose=verbose,
                mask_fn=mask_fn, step_hook=step_hook, done_fn=frame_is_complete,
                context_prefix=INTERNAL_PROMPT
            )

            if not recovered_bits:
                if gui is not None:
                    gui.end_session("(no bits recovered)")
                return jsonify({"success": False, "error": "No bits recovered from steganographic text"}), 400

            # Parse the framed plaintext back out of the recovered bits
            message, integrity_ok = message_bits_to_text(recovered_bits)

            if gui is not None:
                tag = "" if integrity_ok else "  [integrity check FAILED — wrong key?]"
                gui.end_session(f"{message}{tag}")
        finally:
            if gui is not None:
                gui.session_lock.release()

        # Prepare response
        response = {
            "success": True,
            "message": message,
            "integrity_ok": integrity_ok,
            "config": {
                "model_id": model_id,
                "temp": temp,
                "precision": precision,
                "topk": topk
            },
            "stats": {
                "input_length": len(stego_text),
                "generated_tokens": len(stego_text) - len(start_text),  # Approximate
                "recovered_bits": len(recovered_bits)
            }
        }

        if DEBUG_LOG:
            log_decode_result(message, integrity_ok, response)

        return jsonify(response)

    except Exception as e:
        # Always surface the full traceback to the console, mirroring /encode.
        print("\n" + "!" * 80)
        print(f"[decode] DECODING FAILED: {type(e).__name__}: {e}")
        print("!" * 80)
        traceback.print_exc()
        print("!" * 80 + "\n")
        error_msg = f"Decoding error: {str(e)}"
        if verbose or DEBUG_LOG:
            error_msg += f"\n{traceback.format_exc()}"
        return jsonify({"success": False, "error": error_msg}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if default model is cached, if not try to load it
        model, tokenizer = get_model()
        cache_info = get_cache_status()
        
        return jsonify({
            "success": True,
            "status": "healthy",
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
            "device": str(model.device) if model else None,
            "default_model_id": default_model_id,
            "cache_info": cache_info
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "default_model_id": default_model_id,
            "cache_info": get_cache_status()
        }), 500

@app.route('/cache', methods=['GET'])
def cache_status():
    """Get cache status endpoint"""
    return jsonify({
        "success": True,
        "cache_info": get_cache_status()
    })

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear model cache endpoint"""
    try:
        global model_cache
        cache_info_before = get_cache_status()
        
        # Clear the cache
        model_cache.clear()
        
        cache_info_after = get_cache_status()
        
        return jsonify({
            "success": True,
            "message": "Cache cleared successfully",
            "before": cache_info_before,
            "after": cache_info_after
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    return jsonify({
        "name": "Steganographic API Server",
        "version": "1.0.0",
        "default_model_id": default_model_id,
        "cache_info": get_cache_status(),
        "endpoints": {
            "POST /encode": "Encode a plaintext message (+ key) into steganographic text",
            "POST /decode": "Decode steganographic text (+ key) back to the plaintext message",
            "GET /health": "Health check endpoint",
            "GET /cache": "Get cache status",
            "POST /cache/clear": "Clear model cache",
            "GET /": "This documentation"
        },
                 "example_encode": {
             "url": "/encode",
             "method": "POST",
             "body": {
                 "message": "Attack at dawn",
                 "key": "3f9a8c2b1d4e5f60718293a4b5c6d7e8",
                 "start_text": "Hello world",
                 "model_id": default_model_id,
                 "temp": 1.2,
                 "precision": 16,
                 "topk": 50000
             }
         },
         "example_decode": {
             "url": "/decode",
             "method": "POST",
             "body": {
                 "stego_text": "Hello world companies like...",
                 "starter_length": 25,
                 "key": "3f9a8c2b1d4e5f60718293a4b5c6d7e8",
                 "model_id": default_model_id,
                 "temp": 1.2,
                 "precision": 16,
                 "topk": 50000
             }
         }
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Steganographic API Server")
    parser.add_argument('--add-gui', action='store_true',
                        help="Launch a step-by-step encoding visualizer window")
    parser.add_argument('--debug', action='store_true',
                        help="Log each /encode and /decode request to the console: "
                             "the full request body, the message-bit sequence built "
                             "up token by token, the switch to cover text, and the "
                             "final output.")
    args = parser.parse_args()

    set_debug_log(args.debug)

    print("Starting Steganographic API Server...")
    print(f"Default model: {default_model_id}")
    if args.debug:
        print("Debug request logging: ON (--debug)")

    # Preload default model for faster first requests
    preload_default_model()

    # Run the server
    port = int(os.getenv('PORT', 3000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    print(f"Server starting on port {port}")
    print(f"Server URL: http://localhost:{port}")
    print("Available endpoints:")
    print("  POST /encode - Encode ciphertext to steganographic text")
    print("  POST /decode - Decode steganographic text to ciphertext")
    print("  GET /health - Health check")
    print("  GET /cache - Get cache status")
    print("  POST /cache/clear - Clear model cache")
    print("  GET / - API documentation")

    if args.add_gui:
        # Tkinter must own the main thread (required on macOS), so the Flask
        # server runs on a background thread and the GUI blocks here.
        from gui import GuiBridge, run_gui

        bridge = GuiBridge()
        set_gui_bridge(bridge)

        server_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=port, debug=False,
                                   use_reloader=False, threaded=True),
            daemon=True,
        )
        server_thread.start()

        print("GUI mode: a visualizer window is open. Send an /encode request "
              "to step through encoding.")
        run_gui(bridge, port)
    else:
        app.run(host='0.0.0.0', port=port, debug=debug)
