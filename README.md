# Server for StegoNote

Server for encoding and decoding steganographic texts for the [StegaNote](https://steganote.com) app. The app, and this server, are based on the Meteor protocol.

## Setup

### 1. Install Dependencies

This project uses Pipenv for dependency management. Install dependencies with:

```bash
pipenv install
```

### 2. Environment Variables

Create a `.env` file with your Hugging Face token:

```bash
HUGGING_FACE_HUB_TOKEN=your_token_here
PORT=3000
DEBUG=False
```

### 3. Start the Server

```bash
pipenv run python main.py
```

The server will start on `http://localhost:3000` by default.

## API Endpoints

### POST /encode

Encode hexadecimal ciphertext into steganographic text.

**Request Body:**
```json
{
  "ciphertext": "48656c6c6f20576f726c64",
  "start_text": "The weather today is quite nice and ",
  "model_id": "meta-llama/Llama-3.2-1B",
  "temp": 1.2,
  "precision": 16,
  "topk": 50000
}
```

**Parameters:**
- `ciphertext` (required): Hexadecimal string to encode
- `start_text` (required): Starting text for the steganographic output
- `model_id` (optional): Hugging Face model identifier (default: server's default model)
- `temp` (optional): Temperature for sampling (default: 1.2)
- `precision` (optional): Precision for arithmetic coding (default: 16)
- `topk` (optional): Top-k cutoff for vocabulary (default: 50000)

**Response:**
```json
{
  "success": true,
  "stego_text": "The weather today is quite nice and companies like Microsoft are developing...",
  "starter_length": 36,
  "config": {
    "model_id": "meta-llama/Llama-3.2-1B",
    "temp": 1.2,
    "precision": 16,
    "topk": 50000
  },
  "stats": {
    "input_hex_length": 22,
    "input_bytes": 11,
    "message_bits": 88,
    "output_tokens": 25
  }
}
```

### POST /decode

Decode steganographic text back to hexadecimal ciphertext.

**Request Body:**
```json
{
  "stego_text": "The weather today is quite nice and companies like Microsoft are developing...",
  "starter_length": 36,
  "model_id": "meta-llama/Llama-3.2-1B",
  "temp": 1.2,
  "precision": 16,
  "topk": 50000
}
```

**Parameters:**
- `stego_text` (required): Full steganographic text to decode
- `starter_length` (required): Number of characters in the starting text (from encode response)
- `model_id` (optional): Hugging Face model identifier (must match encoding, default: server's default model)
- `temp` (optional): Temperature (must match encoding, default: 1.2)
- `precision` (optional): Precision (must match encoding, default: 16)
- `topk` (optional): Top-k cutoff (must match encoding, default: 50000)

**Response:**
```json
{
  "success": true,
  "ciphertext": "48656c6c6f20576f726c64",
  "config": {
    "model_id": "meta-llama/Llama-3.2-1B",
    "temp": 1.2,
    "precision": 16,
    "topk": 50000
  },
  "stats": {
    "input_length": 245,
    "generated_tokens": 25,
    "recovered_bits": 88,
    "output_hex_length": 22,
    "output_bytes": 11
  }
}
```

### GET /health

Check server health and model status.

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true,
  "device": "cuda:0",
  "default_model_id": "meta-llama/Llama-3.2-1B",
  "cache_info": {
    "cached_models": ["meta-llama/Llama-3.2-1B"],
    "cache_size": 1,
    "default_model_cached": true
  }
}
```

### GET /cache

Get information about cached models.

**Response:**
```json
{
  "success": true,
  "cache_info": {
    "cached_models": ["meta-llama/Llama-3.2-1B"],
    "cache_size": 1,
    "default_model_cached": true
  }
}
```

### POST /cache/clear

Clear all cached models to free up memory.

**Response:**
```json
{
  "success": true,
  "message": "Cache cleared successfully",
  "before": {
    "cached_models": ["meta-llama/Llama-3.2-1B"],
    "cache_size": 1,
    "default_model_cached": true
  },
  "after": {
    "cached_models": [],
    "cache_size": 0,
    "default_model_cached": false
  }
}
```

### GET /

Get API documentation and examples.

## Usage Examples

### Using curl

**Encode:**
```bash
curl -X POST http://localhost:3000/encode \
  -H "Content-Type: application/json" \
  -d '{
    "ciphertext": "48656c6c6f20576f726c64",
    "start_text": "The weather today is quite nice and ",
    "model_id": "meta-llama/Llama-3.2-1B"
  }'
```

**Decode:**
```bash
curl -X POST http://localhost:3000/decode \
  -H "Content-Type: application/json" \
  -d '{
    "stego_text": "The weather today is quite nice and companies like Microsoft are developing...",
    "starter_length": 36,
    "model_id": "meta-llama/Llama-3.2-1B"
  }'
```

### Using Python requests

```python
import requests
import json

# Encode
encode_data = {
    "ciphertext": "48656c6c6f20576f726c64",
    "start_text": "The weather today is quite nice and ",
    "model_id": "meta-llama/Llama-3.2-1B"
}
response = requests.post("http://localhost:3000/encode", json=encode_data)
result = response.json()
stego_text = result["stego_text"]
starter_length = result["starter_length"]
model_id = result["config"]["model_id"]

# Decode
decode_data = {
    "stego_text": stego_text,
    "starter_length": starter_length,
    "model_id": model_id
}
response = requests.post("http://localhost:3000/decode", json=decode_data)
result = response.json()
recovered_ciphertext = result["ciphertext"]
```

## Troubleshooting

**Model Loading Issues:**
- Ensure your Hugging Face token is valid and set in the `.env` file
- Check that you have access to the model on Huggingface
- Verify PyTorch CUDA installation if using GPU
- Use `GET /cache` to check which models are currently loaded
- Use `POST /cache/clear` to free memory if experiencing out-of-memory issues
