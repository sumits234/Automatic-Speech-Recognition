# -*- coding: utf-8 -*-
"""Automatic Speech Recognition.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zrXtqwdFgrcXTCYY2ujqNjUMBrS7Lqb8
"""

# === Cell 1: Mount Google Drive ===
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

!pip install --upgrade transformers

# We'll use: torchaudio, datasets, transformers, evaluate, soundfile, fastapi, uvicorn

!pip install --quiet torchaudio datasets transformers evaluate soundfile fastapi uvicorn

#  Define Paths & Imports
import os
import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,

)
from datasets import load_metric, Dataset, Audio
import numpy as np
import random
import evaluate

# === Cell 3: Extract both train-clean-100 and test-clean ===

# 3.1 Ensure target local folder exists
!rm -rf /content/LibriSpeech           # Start fresh to avoid duplication
!mkdir -p /content/LibriSpeech

# 3.2 Extract train-clean-100.tar.gz
# This will extract /content/LibriSpeech/train-clean-100
!tar -xzf "/content/drive/MyDrive/LibriSpeech/train-clean-100.tar.gz" -C /content

# 3.3 Extract test-clean.tar.gz
# This will extract /content/LibriSpeech/test-clean
!tar -xzf "/content/drive/MyDrive/LibriSpeech/test-clean.tar.gz" -C /content

# 3.4 After extraction, everything should now be in:
# /content/LibriSpeech/train-clean-100/
# /content/LibriSpeech/test-clean/

# 3.5 Confirm folder structure
!echo " Extraction complete. Contents of /content/LibriSpeech:"
!ls -l /content/LibriSpeech

# === Cell 4: Confirm the LOCAL folders and set paths for the rest of the notebook ===
import os

LIBRISPEECH_ROOT = "/content/LibriSpeech"
TRAIN_SUBSET     = os.path.join(LIBRISPEECH_ROOT, "train-clean-100")
TEST_SUBSET      = os.path.join(LIBRISPEECH_ROOT, "test-clean")

print("Contents of /content/LibriSpeech:\n", os.listdir(LIBRISPEECH_ROOT), "\n")
print("TRAIN_SUBSET =", TRAIN_SUBSET, "→", "" if os.path.isdir(TRAIN_SUBSET) else "NOT FOUND")
print("TEST_SUBSET  =", TEST_SUBSET,  "→", "" if os.path.isdir(TEST_SUBSET)  else " NOT FOUND")

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("Ready to proceed with Dataset exploration and model training.")

# == — Basic Exploration of train-clean-100 & test-clean ===
import soundfile as sf
from pathlib import Path
from torchaudio.datasets import LIBRISPEECH

def gather_stats(path):
    n_files = 0
    total_duration = 0.0
    for flac in Path(path).rglob("*.flac"):
        n_files += 1
        info = sf.info(str(flac))
        total_duration += info.frames / info.samplerate
    return n_files, total_duration

# The on-disk structure is /content/LibriSpeech/train-clean-100 and /content/LibriSpeech/test-clean
TRAIN_SUBSET = "/content/LibriSpeech/train-clean-100"
TEST_SUBSET  = "/content/LibriSpeech/test-clean"

# 1. Count files & total duration
train_n, train_dur = gather_stats(TRAIN_SUBSET)
test_n,  test_dur  = gather_stats(TEST_SUBSET)

print(f"Train-clean-100: {train_n} audio files, ≈{train_dur/3600:.2f} hours")
print(f"Test-clean:      {test_n} audio files, ≈{test_dur/3600:.2f} hours")

# 2. Sample transcript lengths from train-clean-100 via torchaudio’s LIBRISPEECH loader
#    Here we set root="/content" so that LIBRISPEECH can find "/content/LibriSpeech/train-clean-100".
loader_train = LIBRISPEECH(root="/content", url="train-clean-100", download=False)
lengths = []
for i, (_, _, transcript, *_ ) in enumerate(loader_train):
    if i >= 100:
        break
    lengths.append(len(transcript.split()))

import numpy as _np
print(f"Sample transcripts (n=100): avg words={_np.mean(lengths):.1f}, min={_np.min(lengths)}, max={_np.max(lengths)}")

# === Cell 7: Task 2 — Load Pretrained Wav2Vec2 & Compute Baseline WER ===
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Check for the test-clean directory
!ls "/content/LibriSpeech"

LIBRISPEECH_ROOT = "/content"
TEST_SUBSET      = "/content/LibriSpeech/test-clean"

!ls "/content/LibriSpeech/test-clean/0121/121726"

# List the top‐level inside test-clean
!ls "/content/LibriSpeech/test-clean"

# Then, for example, list the contents of speaker “121”:
!ls "/content/LibriSpeech/test-clean/121"

!pip install jiwer

import pandas as pd

# Baseline WER on test-clean  ===
LIBRISPEECH_ROOT = "/content"
TEST_SUBSET      = "/content/LibriSpeech/test-clean"

# 2) Load pretrained model + processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Loaded pretrained Wav2Vec2 model and processor.\n")

# 3) Build a pandas DataFrame for test-clean
test_rows = []
loader_test = LIBRISPEECH(root=LIBRISPEECH_ROOT, url="test-clean", download=False)

for waveform, sr, transcript, speaker_id, chapter_id, utterance_id in loader_test:
    # ──► Use speaker_id and chapter_id as plain ints (no zero-padding in folder names)
    spk_folder = str(speaker_id)
    chp_folder = str(chapter_id)
    # ──► Zero-pad only the utterance_id to 4 digits for the filename
    filename = f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac"
    flac_path = os.path.join(TEST_SUBSET, spk_folder, chp_folder, filename)
    test_rows.append({"audio_path": flac_path, "transcript": transcript})

test_df = pd.DataFrame(test_rows)
dataset_test = Dataset.from_pandas(test_df)
dataset_test = dataset_test.cast_column("audio_path", Audio(sampling_rate=16000))
dataset_test = dataset_test.rename_column("audio_path", "audio")
dataset_test = dataset_test.rename_column("transcript", "text")
print(f" Built dataset_test with {len(dataset_test)} utterances.\n")

# 4) Preprocessing function: waveform → input_values; transcript → labels
def prepare_batch(batch):
    audio = batch["audio"]["array"]
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

dataset_test = dataset_test.map(
    prepare_batch,
    remove_columns=["audio", "text"],
    num_proc=4
)
print(" Preprocessing complete (mapped input_values + labels).\n")

# 5) Run inference & compute WER in batches (with padding)
wer_metric = evaluate.load("wer")
predictions = []
references = []
batch_size = 8

for i in range(0, len(dataset_test), batch_size):
    batch = dataset_test[i : i + batch_size]

    # Pad input_values correctly
    input_features = [{"input_values": x} for x in batch["input_values"]]
    padded_inputs = processor.feature_extractor.pad(
        input_features,
        padding=True,
        return_tensors="pt"
    )
    input_values = padded_inputs["input_values"].to(model.device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    preds = processor.batch_decode(pred_ids)

    # Decode references (labels)
    label_ids = batch["labels"]
    references_batch = processor.batch_decode(label_ids, group_tokens=False)

    predictions.extend(preds)
    references.extend(references_batch)

# Final WER score
baseline_wer = wer_metric.compute(predictions=predictions, references=references)
print(f"\n Baseline WER (pretrained {model_name}): {baseline_wer:.3f}\n")

# Show a few examples
for idx in [15, 100, 1000]:
    print(f"— idx={idx}")
    print(f"  REF = {references[idx]}")
    print(f"  HYP = {predictions[idx]}\n")

# === Cell H: Save Model and Processor for FastAPI Deployment ===

# Set the output path (can be your Drive or any local folder)
export_dir = "/content/drive/MyDrive/LibriSpeech/wav2vec2-fastapi"

# Save both processor and model
processor.save_pretrained(export_dir)
model.save_pretrained(export_dir)

print(f" Model and processor saved to: {export_dir}")

# Paths
LIBRISPEECH_ROOT = "/content"
TRAIN_SUBSET = "/content/LibriSpeech/train-clean-100"
TEST_SUBSET = "/content/LibriSpeech/test-clean"
model_name = "facebook/wav2vec2-base-960h"

# Load processor and fresh model for fine-tuning
processor = Wav2Vec2Processor.from_pretrained(model_name)
model_ft = Wav2Vec2ForCTC.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# === Prepare train-clean-100 Dataset ===
# Build DataFrame from torchaudio's loader
train_rows = []
loader_train = LIBRISPEECH(root=LIBRISPEECH_ROOT, url="train-clean-100", download=False)

for waveform, sr, transcript, speaker_id, chapter_id, utterance_id in loader_train:
    filename = f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac"
    flac_path = os.path.join(TRAIN_SUBSET, str(speaker_id), str(chapter_id), filename)
    train_rows.append({"audio_path": flac_path, "transcript": transcript})

train_df = pd.DataFrame(train_rows)
dataset_train = Dataset.from_pandas(train_df)
dataset_train = dataset_train.cast_column("audio_path", Audio(sampling_rate=16000))
dataset_train = dataset_train.rename_column("audio_path", "audio")
dataset_train = dataset_train.rename_column("transcript", "text")

# Same for test-clean (again, from scratch)
test_rows = []
loader_test = LIBRISPEECH(root=LIBRISPEECH_ROOT, url="test-clean", download=False)

for waveform, sr, transcript, speaker_id, chapter_id, utterance_id in loader_test:
    filename = f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac"
    flac_path = os.path.join(TEST_SUBSET, str(speaker_id), str(chapter_id), filename)
    test_rows.append({"audio_path": flac_path, "transcript": transcript})

test_df = pd.DataFrame(test_rows)
dataset_test = Dataset.from_pandas(test_df)
dataset_test = dataset_test.cast_column("audio_path", Audio(sampling_rate=16000))
dataset_test = dataset_test.rename_column("audio_path", "audio")
dataset_test = dataset_test.rename_column("transcript", "text")

# ===  RAM-Safe Preprocessing ===
# Only use first 2000 training samples for now

def prepare_batch(batch):
    audio = batch["audio"]["array"]
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

#  Select a small subset from train-clean-100
dataset_train = dataset_train.select(range(2000))   # safe size for Colab

#  Keep full test-clean (small enough to fit)
dataset_train = dataset_train.map(prepare_batch, remove_columns=["audio", "text"])
dataset_test  = dataset_test.map(prepare_batch, remove_columns=["audio", "text"])

print(" Preprocessing complete (subset of train, full test)")

from dataclasses import dataclass

# === Cell D: Data Collator (for padding) ===
from dataclasses import dataclass  #  Required for @dataclass

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )

        # Replace padding tokens with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch

#  Instantiate the collator
data_collator = DataCollatorCTCWithPadding(processor=processor)

# === Cell E: WER Evaluation Function ===
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

!pip install --upgrade transformers

from transformers import TrainingArguments

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test",
    evaluation_strategy="epoch"
)
print("TrainingArguments works!")

# === Cell F: TrainingArguments and Trainer ===
from transformers import TrainingArguments, Trainer  #  Make sure this is imported

output_dir = "/content/drive/MyDrive/LibriSpeech/finetuned-wav2vec2"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",  #  Requires correct transformers version
    num_train_epochs=3,
    gradient_accumulation_steps=4,
    fp16=torch.cuda.is_available(),
    learning_rate=1e-4,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    report_to="none",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model_ft,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# === Cell G: Start Fine-Tuning ===
trainer.train()
trainer.save_model(output_dir)
print(f" Fine-tuned model saved to: {output_dir}")

!pip install fastapi uvicorn transformers torchaudio soundfile python-multipart

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Assuming you already have 'model' and 'processor' loaded from training
export_dir = "/content/wav2vec2-fastapi"

#  Save processor and model (creates preprocessor_config.json too)
processor.save_pretrained(export_dir)
model.save_pretrained(export_dir)

#  Confirm files
import os
print("Saved files:", os.listdir(export_dir))
