# Automatic-Speech-Recognition


This repository contains the implementation for  Assignment, focusing on Automatic Speech Recognition (ASR) using the LibriSpeech dataset and the Wav2Vec2 model. The project includes data exploration, baseline evaluation, fine-tuning, and deployment of the model as a FastAPI service.

## Objective
The goal is to:
1. Explore the LibriSpeech dataset (`train-clean-100` and `test-clean` subsets).
2. Evaluate a pretrained Wav2Vec2 model (`facebook/wav2vec2-base-960h`) by computing the Word Error Rate (WER) on the `test-clean` subset.
3. Fine-tune the model on `train-clean-100` and compare WER with the baseline.
4. a FastAPI service for audio transcription.

## Repository Structure
- `Automatic_Speech_Recognition.py`: Jupyter notebook containing the full workflow (data exploration, evaluation, fine-tuning, and model saving).
- `app.py`: FastAPI application script for serving the fine-tuned Wav2Vec2 model.
- 'model':File.
- `test_transcribe.py`: Script to test the FastAPI `/transcribe` endpoint.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: This file, explaining setup, usage, and results.

## Setup Instructions
### Prerequisites
- Python 3.8+
- A machine with a GPU (recommended for faster training; CPU works but is slower)
- Google Drive (for storing dataset and model checkpoints, if using Colab)
- Git installed for cloning the repository
- Internet access for downloading dependencies and pretrained models
## Result
WER score = 0.038

