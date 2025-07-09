# Server for StegoNote

A Flask-based HTTP server that provides REST API endpoints for encoding and decoding steganographic text using LLMs.

## Features

- **POST /encode**: Encode hexadecimal ciphertext into natural language steganographic text
- **POST /decode**: Decode steganographic text back to hexadecimal ciphertext  
- **GET /health**: Health check endpoint
- **GET /**: API documentation
- CORS enabled for cross-origin requests
- Model loaded once on startup for efficiency
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
  "temp": 1.2,
  "precision": 16,
  "topk": 50000
}
```

**Parameters:**
- `ciphertext` (required): Hexadecimal string to encode
- `start_text` (required): Starting text for the steganographic output
- `temp` (optional): Temperature for sampling (default: 1.2)
- `precision` (optional): Precision for arithmetic coding (default: 16)
- `topk` (optional): Top-k cutoff for vocabulary (default: 50000)

**Response:**
```json
{
  "success": true,
  "stego_text": "The weather today is quite nice and companies like Microsoft are developing...",
  "starter_length": 36,
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
  "temp": 1.2,
  "precision": 16,
  "topk": 50000
}
```

**Parameters:**
- `stego_text` (required): Full steganographic text to decode
- `starter_length` (required): Number of characters in the starting text (from encode response)
- `temp` (optional): Temperature (must match encoding, default: 1.2)
- `precision` (optional): Precision (must match encoding, default: 16)
- `topk` (optional): Top-k cutoff (must match encoding, default: 50000)

**Response:**
```json
{
  "success": true,
  "ciphertext": "48656c6c6f20576f726c64",
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
  "device": "cuda:0"
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
    "start_text": "The weather today is quite nice and "
  }'
```

**Decode:**
```bash
curl -X POST http://localhost:3000/decode \
  -H "Content-Type: application/json" \
  -d '{
    "stego_text": "The weather today is quite nice and companies like Microsoft are developing...",
    "starter_length": 36
  }'
```

### Using Python requests

```python
import requests
import json

# Encode
encode_data = {
    "ciphertext": "48656c6c6f20576f726c64",
    "start_text": "The weather today is quite nice and "
}
response = requests.post("http://localhost:3000/encode", json=encode_data)
result = response.json()
stego_text = result["stego_text"]
starter_length = result["starter_length"]

# Decode
decode_data = {
    "stego_text": stego_text,
    "starter_length": starter_length
}
response = requests.post("http://localhost:3000/decode", json=decode_data)
result = response.json()
recovered_ciphertext = result["ciphertext"]
```

## Important Notes

1. **Parameter Consistency**: All parameters used for encoding (temp, precision, topk) must be exactly the same for decoding to work correctly.

2. **Model Loading**: The model is loaded once on startup. First requests may be slower while the model initializes.

3. **Memory Usage**: The DeepSeek model requires significant GPU memory. Ensure you have adequate VRAM available.

4. **Error Handling**: The server includes comprehensive error handling and will return detailed error messages for debugging.

5. **CORS**: Cross-origin requests are enabled by default for web application integration.

## Troubleshooting

**Model Loading Issues:**
- Ensure your Hugging Face token is valid and set in the `.env` file
- Check that you have sufficient GPU memory available
- Verify PyTorch CUDA installation if using GPU

**Encoding/Decoding Failures:**
- Ensure all parameters match between encoding and decoding
- Check that the starter_length exactly matches the value returned from encoding
- Verify the hex string is properly formatted (even length, valid hex characters)

**Server Startup Issues:**
- Check that the port is not already in use
- Verify all dependencies are installed correctly with `pipenv install`
- Review the server logs for specific error messages 