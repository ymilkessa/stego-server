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
default_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Default precision (bits)
default_precision = 8

# Import the encoding/decoding functions from the raw modules
from raw_stego_encoder import hex_to_bits, encode_steganographic
from raw_stego_decoder import bits_to_hex, decode_steganographic

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model (loaded once on startup)
model = None
tokenizer = None

# Server-side debugging flag - set to True when you want to debug
verbose = True


def setup_model(model_id=None):
    """Initialize the model and tokenizer
    
    Args:
        model_id: Hugging Face model identifier. If None, uses default_model_id
    
    Returns:
        tuple: (model, tokenizer)
    """
    if model_id is None:
        model_id = default_model_id
        
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not found in environment variables")
    
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Model loaded on device: {model.device}")
    return model, tokenizer


@app.route('/encode', methods=['POST'])
def encode_endpoint():
    """
    Encode endpoint: POST /encode
    
    JSON Body:
    {
        "ciphertext": "5361486a4b31593d",  // hex string
        "start_text": "Hello world...",    // starting text
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  // optional, default from server
        "temp": 1.2,                       // optional, default 1.2
        "precision": 8,                    // optional, default 8
        "topk": 50000                      // optional, default 50000
    }
    
    Response:
    {
        "success": true,
        "stego_text": "Hello world companies like...",
        "starter_length": 25,              // length of starting text in characters
        "config": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "temp": 1.2,
            "precision": 8,
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
        precision = data.get('precision', default_precision)
        topk = data.get('topk', 50000)
        
        # Setup model with specified model_id
        model, tokenizer = setup_model(model_id)
        
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
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  // optional, default from server
        "temp": 1.2,                                     // optional, default 1.2
        "precision": 8,                                  // optional, default 8
        "topk": 50000                                    // optional, default 50000
    }
    
    Response:
    {
        "success": true,
        "ciphertext": "5361486a4b31593d",  // recovered hex string
        "config": {
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "temp": 1.2,
            "precision": 8,
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
        precision = data.get('precision', default_precision)
        topk = data.get('topk', 50000)
        
        # Validate starter_length
        if not isinstance(starter_length, int) or starter_length < 1:
            return jsonify({"success": False, "error": "starter_length must be a positive integer"}), 400
        if starter_length >= len(stego_text):
            return jsonify({"success": False, "error": "starter_length must be less than the total stego_text length"}), 400
        
        # Extract start_text using the provided starter_length
        start_text = stego_text[:starter_length]
        
        # Setup model with specified model_id
        model, tokenizer = setup_model(model_id)
        
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
        model, tokenizer = setup_model()
        return jsonify({
            "success": True,
            "status": "healthy",
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
            "device": str(model.device) if model else None,
            "default_model_id": default_model_id
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "default_model_id": default_model_id
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    return jsonify({
        "name": "Steganographic API Server",
        "version": "1.0.0",
        "default_model_id": default_model_id,
        "endpoints": {
            "POST /encode": "Encode hexadecimal ciphertext into steganographic text",
            "POST /decode": "Decode steganographic text back to hexadecimal ciphertext",
            "GET /health": "Health check endpoint",
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
                 "precision": default_precision,
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
                 "precision": default_precision,
                 "topk": 50000
             }
         }
    })

if __name__ == '__main__':
    print("Starting Steganographic API Server...")
    print(f"Default model: {default_model_id}")
    print("Loading model on startup...")
    
    try:
        setup_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first request.")
    
    # Run the server
    port = int(os.getenv('PORT', 3000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"Server starting on port {port}")
    print(f"Server URL: http://localhost:{port}")
    print("Available endpoints:")
    print("  POST /encode - Encode ciphertext to steganographic text")
    print("  POST /decode - Decode steganographic text to ciphertext")
    print("  GET /health - Health check")
    print("  GET / - API documentation")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
