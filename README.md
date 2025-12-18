# SAM Audio Service

Audio source separation using Meta's [Segment Anything Audio (SAM-Audio)](https://github.com/facebookresearch/sam-audio) model. Isolate any sound from audio using text descriptions.

## Features

- **Text-prompted separation**: Describe what you want to isolate (e.g., "Drums", "Vocals", "A man speaking")
- **Multiple model sizes**: small, base, large
- **RunPod Serverless**: Deploy as a scalable serverless endpoint

## Local Development

### Setup

```bash
cd sam_service

# Create virtual environment with Python 3.11+
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install torch torchaudio transformers scipy soundfile torchcodec torchdiffeq descript-audiotools eva-decord

# Install Facebook Research packages
uv pip install --no-deps git+https://github.com/facebookresearch/sam-audio.git
uv pip install --no-deps git+https://github.com/facebookresearch/perception_models.git@unpin-deps
uv pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git
uv pip install --no-deps git+https://github.com/facebookresearch/dacvae.git
uv pip install --no-deps git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f
uv pip install iopath
```

### Hugging Face Authentication

The SAM Audio model requires Hugging Face authentication:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Request access to [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large)
3. Login via CLI:
   ```bash
   huggingface-cli login
   ```

### Run Locally

```bash
# Convert audio to WAV if needed
ffmpeg -i test.mp3 -ar 16000 test.wav

# Run the test script
python test.py
```

## RunPod Serverless Deployment

### Build Docker Image

```bash
# Build the image
docker build -t sam-audio-serverless .

# Tag for your registry
docker tag sam-audio-serverless your-registry/sam-audio-serverless:latest

# Push to registry
docker push your-registry/sam-audio-serverless:latest
```

### Deploy to RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create a new endpoint
3. Select your Docker image
4. Configure GPU (recommended: RTX 3090 or better for `large` model)
5. **Important:** Set environment variables:
   - `HF_TOKEN`: Your Hugging Face access token (required for gated model access)

#### Getting Your HF_TOKEN

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" access
3. Request access to the model at [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large)
4. Add the token as `HF_TOKEN` in your RunPod endpoint environment variables

### API Usage

#### Request Format

```json
{
  "input": {
    "audio_url": "https://example.com/audio.wav",
    "description": "Drums",
    "model_size": "large"
  }
}
```

Or with base64 audio:

```json
{
  "input": {
    "audio_base64": "base64_encoded_audio_data",
    "description": "Vocals",
    "model_size": "large"
  }
}
```

#### Response Format

```json
{
  "target_base64": "base64_encoded_isolated_audio",
  "residual_base64": "base64_encoded_residual_audio",
  "sample_rate": 16000,
  "description": "Drums",
  "status": "success"
}
```

### Example Python Client

```python
import runpod
import base64

runpod.api_key = "your_api_key"

# Create endpoint instance
endpoint = runpod.Endpoint("your_endpoint_id")

# Run separation
result = endpoint.run_sync({
    "input": {
        "audio_url": "https://example.com/song.wav",
        "description": "Drums"
    }
})

# Decode and save result
if result.get("status") == "success":
    target_audio = base64.b64decode(result["target_base64"])
    with open("drums_isolated.wav", "wb") as f:
        f.write(target_audio)
```

## Supported Descriptions

SAM-Audio can isolate various sounds:

- **Instruments**: "Drums", "Guitar", "Piano", "Bass", "Violin"
- **Vocals**: "Vocals", "A man speaking", "A woman singing"
- **Effects**: "Applause", "Crowd noise", "Wind"
- **General**: Any natural language description of the sound

## Model Sizes

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| small | ~4GB | Fast | Good |
| base | ~8GB | Medium | Better |
| large | ~16GB | Slower | Best |

## License

SAM-Audio is licensed under the [SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE).

