"""
RunPod Serverless Handler for SAM Audio (Segment Anything Audio)
Separates audio sources based on text descriptions.

Input format:
{
    "input": {
        "audio_url": "https://example.com/audio.wav",  # URL to audio file
        # OR
        "audio_base64": "base64_encoded_audio_data",   # Base64 encoded audio
        "description": "Drums",                         # What sound to isolate
        "model_size": "large"                           # Optional: small, base, large (default: large)
    }
}

Output format:
{
    "target_base64": "base64_encoded_isolated_audio",
    "residual_base64": "base64_encoded_residual_audio",
    "sample_rate": 16000,
    "description": "Drums"
}
"""

import runpod
import torch
import torchaudio
import base64
import os
import tempfile
import requests
from io import BytesIO

# Global model cache - loaded once on cold start
MODEL = None
PROCESSOR = None
DEVICE = None


def load_model(model_size: str = "large"):
    """Load the SAM Audio model and processor."""
    global MODEL, PROCESSOR, DEVICE
    
    if MODEL is not None and PROCESSOR is not None:
        return MODEL, PROCESSOR, DEVICE
    
    from sam_audio import SAMAudio, SAMAudioProcessor
    from huggingface_hub import login
    
    # Authenticate with Hugging Face if token is available
    # hf_token = os.environ.get("HF_TOKEN")
    # if hf_token:
    #     print("Authenticating with Hugging Face...")
    #     login(token=hf_token)
    # else:
    #     print("Warning: HF_TOKEN not set. Model download may fail for gated models.")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAM Audio model (size: {model_size}) on device: {DEVICE}")
    
    model_id = f"facebook/sam-audio-{model_size}"
    MODEL = SAMAudio.from_pretrained(model_id).to(DEVICE).eval()
    PROCESSOR = SAMAudioProcessor.from_pretrained(model_id)
    
    print("Model loaded successfully!")
    return MODEL, PROCESSOR, DEVICE


def download_audio(url: str, temp_dir: str) -> str:
    """Download audio from URL to a temporary file."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    # Determine file extension from URL or content type
    content_type = response.headers.get("content-type", "")
    if "wav" in url.lower() or "wav" in content_type:
        ext = ".wav"
    elif "mp3" in url.lower() or "mp3" in content_type or "mpeg" in content_type:
        ext = ".mp3"
    elif "flac" in url.lower() or "flac" in content_type:
        ext = ".flac"
    else:
        ext = ".wav"  # Default to wav
    
    temp_path = os.path.join(temp_dir, f"input_audio{ext}")
    with open(temp_path, "wb") as f:
        f.write(response.content)
    
    return temp_path


def decode_base64_audio(audio_base64: str, temp_dir: str) -> str:
    """Decode base64 audio to a temporary file."""
    audio_data = base64.b64decode(audio_base64)
    temp_path = os.path.join(temp_dir, "input_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(audio_data)
    return temp_path


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode audio file to base64."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    """
    Process audio separation job using SAM Audio.
    
    Expected input:
    - audio_url OR audio_base64: The audio to process
    - description: Text description of the sound to isolate (e.g., "Drums", "Vocals")
    - model_size: Optional model size (small, base, large). Default: large
    """
    job_input = job["input"]
    
    # Validate input
    if "audio_url" not in job_input and "audio_base64" not in job_input:
        return {"error": "Missing 'audio_url' or 'audio_base64' in input"}
    
    if "description" not in job_input:
        return {"error": "Missing 'description' in input"}
    
    description = job_input["description"]
    model_size = job_input.get("model_size", "large")
    
    if model_size not in ["small", "base", "large"]:
        return {"error": f"Invalid model_size '{model_size}'. Must be: small, base, or large"}
    
    # Load model
    runpod.serverless.progress_update(job, "Loading model...")
    model, processor, device = load_model(model_size)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Get audio file
            runpod.serverless.progress_update(job, "Downloading/decoding audio...")
            
            if "audio_url" in job_input:
                audio_path = download_audio(job_input["audio_url"], temp_dir)
            else:
                audio_path = decode_base64_audio(job_input["audio_base64"], temp_dir)
            
            # Convert to WAV if needed (using torchaudio)
            runpod.serverless.progress_update(job, "Processing audio...")
            
            # Process with SAM Audio
            runpod.serverless.progress_update(job, f"Separating audio: '{description}'...")
            
            batch = processor(audios=[audio_path], descriptions=[description]).to(device)
            
            with torch.inference_mode():
                result = model.separate(batch)
            
            # Save separated audio
            runpod.serverless.progress_update(job, "Encoding results...")
            
            sample_rate = processor.audio_sampling_rate
            
            target_path = os.path.join(temp_dir, "target.wav")
            residual_path = os.path.join(temp_dir, "residual.wav")
            
            torchaudio.save(target_path, result.target.cpu(), sample_rate)
            torchaudio.save(residual_path, result.residual.cpu(), sample_rate)
            
            # Encode to base64
            target_base64 = encode_audio_to_base64(target_path)
            residual_base64 = encode_audio_to_base64(residual_path)
            
            return {
                "target_base64": target_base64,
                "residual_base64": residual_base64,
                "sample_rate": sample_rate,
                "description": description,
                "status": "success"
            }
            
        except requests.RequestException as e:
            return {"error": f"Failed to download audio: {str(e)}"}
        except Exception as e:
            return {"error": f"Processing error: {str(e)}"}


# Pre-load model on cold start for faster inference
# This runs when the container starts
if os.environ.get("RUNPOD_POD_ID"):
    print("Running on RunPod - pre-loading model...")
    try:
        load_model("large")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")


# Start the serverless worker
runpod.serverless.start({"handler": handler})

