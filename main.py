# app/main.py
import io
from pathlib import Path

import torch
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = FastAPI(
    title="Wav2Vec2 Transcription Service",
    description="Upload an audio file and receive a text transcript from a fine-tuned Wav2Vec2 model.",
    version="1.0.0",
)

# ──────── 1) Resolve MODEL_DIR in a foolproof way ─────────
# __file__ is something like ".../my_asr_service/app/main.py"
BASE_DIR = Path(__file__).parent.parent        # → ".../my_asr_service"
MODEL_DIR = BASE_DIR / "model" / "wav2vec2-finetuned"
print(f"Loading processor & model from: {MODEL_DIR}")

# ──────── 2) Load Wav2Vec2Processor & Wav2Vec2ForCTC ────────
# Make sure that `preprocessor_config.json` and the rest live in MODEL_DIR.
processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR))

if torch.cuda.is_available():
    model = model.to("cuda")
    print("Model moved to CUDA")
else:
    print("CUDA not available; using CPU")

# ──────── 3) Define the /transcribe endpoint ────────────
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts a WAV/MP3/FLAC file as multipart/form-data,
    resamples to 16kHz if needed, runs the Wav2Vec2 model,
    and returns the transcription as JSON.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    filename = file.filename.lower()
    if not (filename.endswith(".wav") or filename.endswith(".mp3") or filename.endswith(".flac")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .wav, .mp3, or .flac."
        )

    # Read raw bytes from the upload and decode via soundfile
    try:
        audio_bytes = await file.read()
        bio = io.BytesIO(audio_bytes)
        audio_input, sr = sf.read(bio, dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio file: {e}")

    # If sample rate ≠ 16 kHz, resample
    target_sr = 16_000
    if sr != target_sr:
        try:
            audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Resampling error: {e}")

    # Prepare input for the processor (it wants a float32 NumPy array @ 16kHz)
    inputs = processor(
        audio_input,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    # Move to GPU if available
    input_values = inputs.input_values
    if torch.cuda.is_available():
        input_values = input_values.to("cuda")

    # Run the model (no_grad)
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return {"filename": file.filename, "transcription": transcription}


@app.get("/")
async def read_root():
    return {
        "message": "Welcome! POST /transcribe with an audio file to get back text."
    }