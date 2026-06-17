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

## Running with Docker

Docker lets the small language model be **downloaded once and reused** across
restarts. The model weights are not baked into the image; on first run they are
fetched into the HuggingFace cache (`HF_HOME=/models`), which is mounted as a
named Docker volume (`stego-models`). Stopping or removing the container keeps
that volume intact — only the explicit cleanup command below wipes the model.

You still need a `.env` with `HUGGING_FACE_HUB_TOKEN` (see above); it is passed
to the container at runtime and never stored in the image.

```bash
make build        # build the image
make up           # start the server (first run downloads the model)
make logs         # follow logs — watch the model load
make down         # stop the container; the model stays on disk
```

The server listens on `http://localhost:3000`. Set `PORT` in your shell to map a
different host port (`PORT=8080 make up`).

### Wiping the model from disk

Closing the program does **not** remove the model. To delete the downloaded
weights explicitly:

```bash
make clean-model  # stop the container and delete ONLY the model volume
```

Or remove everything (container, image, and the model volume):

```bash
make clean
```

Equivalent raw commands if you'd rather not use `make`:

```bash
docker compose up -d          # start
docker compose down           # stop (model kept)
docker volume rm stego-models # wipe the downloaded model
```

### Step-by-step GUI (`--add-gui`) in the container

The Tkinter visualizer also runs in the container, but Tkinter has no display of
its own — it draws its window on an **X server running on your host**. So the
container has to be pointed at that X server (`DISPLAY`) and the host has to
authorize the connection. The image already includes the Tk libraries.

**macOS** — Docker runs in a Linux VM and can't use the Mac's native windowing,
so you need [XQuartz](https://www.xquartz.org/):

1. Install and launch XQuartz.
2. In XQuartz → Preferences → Security, enable **"Allow connections from network
   clients"**, then restart XQuartz.
3. Allow the connection and run:

   ```bash
   xhost + 127.0.0.1
   DISPLAY=host.docker.internal:0 make gui
   ```

**Linux** — share the host's X11 socket (already mounted by the overlay) and
authorize Docker:

```bash
xhost +local:docker
make gui
```

`make gui` runs in the **foreground** (Tk must own the main thread, and the
window stays attached to your terminal). Send an `/encode` request from another
terminal to step through encoding. Under the hood it adds an overlay compose
file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gui.yml up
```

> If you see `couldn't connect to display`, the X server isn't reachable or
> isn't authorized — recheck the `xhost`/XQuartz network-clients steps above.

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
