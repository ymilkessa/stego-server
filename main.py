#!/usr/bin/env python3
"""
Steganographic HTTP Server
Provides REST API endpoints for encoding and decoding steganographic text
"""

import os
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

# Import the encoding/decoding functions from the raw modules
from raw_stego_encoder import hex_to_bits, encode_steganographic
from raw_stego_decoder import bits_to_hex, decode_steganographic

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model cache - stores loaded models to avoid reloading
model_cache = {}

# Server-side debugging flag - set to True when you want to debug
verbose = False


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
    
    JSON Body:
    {
        "ciphertext": "5361486a4b31593d",  // hex string
        "start_text": "Hello world...",    // starting text
        "model_id": "meta-llama/Llama-3.2-1B",  // optional, default from server
        "temp": 1.2,                       // optional, default 1.2
        "precision": 16,                   // optional, default 16
        "topk": 50000                      // optional, default 50000
    }
    
    Response:
    {
        "success": true,
        "stego_text": "Hello world companies like...",
        "starter_length": 25,              // length of starting text in characters
        "config": {
            "model_id": "meta-llama/Llama-3.2-1B",
            "temp": 1.2,
            "precision": 16,
            "topk": 50000
        },
        "stats": {
            "input_hex_length": 16,
            "input_bytes": 8,
            "message_bits": 64,
            "output_tokens": 25
        }
    }
    """
    try:
        # Parse JSON request
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        print(f"Data: {data}")
        
        # Extract required parameters
        ciphertext_hex = data.get('ciphertext')
        start_text = data.get('start_text')
        
        if not ciphertext_hex:
            return jsonify({"success": False, "error": "Missing required parameter: ciphertext"}), 400
        if not start_text:
            return jsonify({"success": False, "error": "Missing required parameter: start_text"}), 400
        
        # Extract optional parameters
        model_id = data.get('model_id', default_model_id)
        temp = data.get('temp', 1.2)
        precision = data.get('precision', 16)
        topk = data.get('topk', 50000)
        
        # Get model from cache (loads if not cached)
        model, tokenizer = get_model(model_id)
        
        # Convert hex to bits
        message_bits = hex_to_bits(ciphertext_hex)
        
        # Encode steganographically using imported function
        print(f"Encoding steganographic text...")
        generated_tokens = encode_steganographic(
            model, tokenizer, message_bits, start_text, 
            temp=temp, precision=precision, topk=topk, verbose=verbose
        )
        print(f"Just finished encoding steganographic text...")

        
        # Decode and create full steganographic text
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        full_stego_text = start_text + generated_text
        
        # Prepare response
        response = {
            "success": True,
            "stego_text": full_stego_text,
            "starter_length": len(start_text),
            "config": {
                "model_id": model_id,
                "temp": temp,
                "precision": precision,
                "topk": topk
            },
            "stats": {
                "input_hex_length": len(ciphertext_hex),
                "input_bytes": len(ciphertext_hex) // 2,
                "message_bits": len(message_bits),
                "output_tokens": generated_tokens.shape[1]
            }
        }

        print(f"Response: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Encoding error: {str(e)}"
        if verbose:
            error_msg += f"\n{traceback.format_exc()}"
        return jsonify({"success": False, "error": error_msg}), 500

@app.route('/decode', methods=['POST'])
def decode_endpoint():
    """
    Decode endpoint: POST /decode
    
    JSON Body:
    {
        "stego_text": "Hello world companies like...",  // full steganographic text
        "starter_length": 25,                            // number of characters in starting text
        "model_id": "meta-llama/Llama-3.2-1B",  // optional, default from server
        "temp": 1.2,                                     // optional, default 1.2
        "precision": 16,                                 // optional, default 16
        "topk": 50000                                    // optional, default 50000
    }
    
    Response:
    {
        "success": true,
        "ciphertext": "5361486a4b31593d",  // recovered hex string
        "config": {
            "model_id": "meta-llama/Llama-3.2-1B",
            "temp": 1.2,
            "precision": 16,
            "topk": 50000
        },
        "stats": {
            "input_length": 245,
            "generated_tokens": 25,
            "recovered_bits": 64,
            "output_hex_length": 16,
            "output_bytes": 8
        }
    }
    """
    try:
        # Parse JSON request
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        # Extract required parameters
        stego_text = data.get('stego_text')
        starter_length = data.get('starter_length')
        
        if not stego_text:
            return jsonify({"success": False, "error": "Missing required parameter: stego_text"}), 400
        if starter_length is None:
            return jsonify({"success": False, "error": "Missing required parameter: starter_length"}), 400
        
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
        
        # Decode steganographically using imported function
        recovered_bits = decode_steganographic(
            model, tokenizer, stego_text, start_text,
            temp=temp, precision=precision, topk=topk, verbose=verbose
        )
        
        if not recovered_bits:
            return jsonify({"success": False, "error": "No bits recovered from steganographic text"}), 400
        
        # Convert bits back to hex
        recovered_hex = bits_to_hex(recovered_bits)
        
        # Prepare response
        response = {
            "success": True,
            "ciphertext": recovered_hex,
            "config": {
                "model_id": model_id,
                "temp": temp,
                "precision": precision,
                "topk": topk
            },
            "stats": {
                "input_length": len(stego_text),
                "generated_tokens": len(stego_text) - len(start_text),  # Approximate
                "recovered_bits": len(recovered_bits),
                "output_hex_length": len(recovered_hex),
                "output_bytes": len(recovered_hex) // 2
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Decoding error: {str(e)}"
        if verbose:
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
            "POST /encode": "Encode hexadecimal ciphertext into steganographic text",
            "POST /decode": "Decode steganographic text back to hexadecimal ciphertext",
            "GET /health": "Health check endpoint",
            "GET /cache": "Get cache status",
            "POST /cache/clear": "Clear model cache",
            "GET /": "This documentation"
        },
                 "example_encode": {
             "url": "/encode",
             "method": "POST",
             "body": {
                 "ciphertext": "5361486a4b31593d",
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
                 "model_id": default_model_id,
                 "temp": 1.2,
                 "precision": 16,
                 "topk": 50000
             }
         }
    })

if __name__ == '__main__':
    print("Starting Steganographic API Server...")
    print(f"Default model: {default_model_id}")
    
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
    
    app.run(host='0.0.0.0', port=port, debug=debug)
