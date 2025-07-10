# Server for StegoNote

A Flask-based HTTP server that provides REST API endpoints for encoding and decoding steganographic text using LLMs.

## Features

- **POST /encode**: Encode hexadecimal ciphertext into natural language steganographic text
- **POST /decode**: Decode steganographic text back to hexadecimal ciphertext  
- **GET /health**: Health check endpoint
- **GET /cache**: Get model cache status
- **POST /cache/clear**: Clear model cache
- **GET /**: API documentation
- CORS enabled for cross-origin requests
- Intelligent model caching system for optimal performance
- Default model preloaded on startup for faster first requests
- Support for multiple models with automatic caching
- Comprehensive error handling and logging

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
  "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
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
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
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
  "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
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
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
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
  "default_model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
  "cache_info": {
    "cached_models": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
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
    "cached_models": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
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
    "cached_models": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
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

## Testing

Run the test suite to verify the server is working correctly:

```bash
pipenv run python test_server.py
```

This will test all endpoints and verify that encoding/decoding works properly.

## Usage Examples

### Using curl

**Encode:**
```bash
curl -X POST http://localhost:3000/encode \
  -H "Content-Type: application/json" \
  -d '{
    "ciphertext": "48656c6c6f20576f726c64",
    "start_text": "The weather today is quite nice and ",
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  }'
```

**Decode:**
```bash
curl -X POST http://localhost:3000/decode \
  -H "Content-Type: application/json" \
  -d '{
    "stego_text": "The weather today is quite nice and companies like Microsoft are developing...",
    "starter_length": 36,
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
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
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
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

## Important Notes

1. **Parameter Consistency**: All parameters used for encoding (model_id, temp, precision, topk) must be exactly the same for decoding to work correctly.

2. **Model Caching**: Models are cached after first use for optimal performance. The default model is preloaded on startup, so first requests are fast. Different models are cached separately.

3. **Memory Usage**: The DeepSeek model requires significant GPU memory. Ensure you have adequate VRAM available.

4. **Error Handling**: The server includes comprehensive error handling and will return detailed error messages for debugging.

5. **CORS**: Cross-origin requests are enabled by default for web application integration.

## Model Caching System

The server implements an intelligent caching system to optimize performance:

### How It Works
- **Default Model Preloading**: The default model is loaded automatically on server startup
- **Lazy Loading**: Additional models are loaded only when first requested
- **Persistent Caching**: Models stay in memory between requests for fast subsequent access
- **Multi-Model Support**: Different models are cached separately, allowing you to switch between them efficiently

### Performance Benefits
- **Fast First Requests**: Default model is ready immediately
- **No Redundant Loading**: Same models are never loaded twice
- **Memory Efficient**: Only requested models are kept in memory
- **Scalable**: Can handle multiple different models simultaneously

### Cache Management
- **View Cache Status**: Use `GET /cache` to see which models are cached
- **Clear Cache**: Use `POST /cache/clear` to free memory when needed
- **Automatic Cleanup**: Cache persists until server restart or manual clearing

## Troubleshooting

**Model Loading Issues:**
- Ensure your Hugging Face token is valid and set in the `.env` file
- Check that you have sufficient GPU memory available
- Verify PyTorch CUDA installation if using GPU
- Use `GET /cache` to check which models are currently loaded
- Use `POST /cache/clear` to free memory if experiencing out-of-memory issues

**Encoding/Decoding Failures:**
- Ensure all parameters match between encoding and decoding
- Check that the starter_length exactly matches the value returned from encoding
- Verify the hex string is properly formatted (even length, valid hex characters)

**Server Startup Issues:**
- Check that the port is not already in use
- Verify all dependencies are installed correctly with `pipenv install`
- Review the server logs for specific error messages 