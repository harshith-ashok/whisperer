# 🎯 Speech-Impairment → Clear Speech Converter
## Complete Engineering Implementation Guide

> **System Goal**: Convert impaired/slurred speech into clear, natural speech using open-source ML models, optimized for Apple Silicon M3 Pro with minimal Azure usage.

---

## 📋 Table of Contents

1. [System Architecture Overview](#architecture)
2. [Environment Setup](#environment-setup)
3. [Dataset Acquisition & Preparation](#dataset-prep)
4. [Whisper ASR Fine-Tuning](#whisper-training)
5. [TTS/Voice Cloning Setup](#tts-setup)
6. [Backend Development (FastAPI)](#backend)
7. [Frontend Development (React)](#frontend)
8. [Integration & Testing](#integration)
9. [Deployment & Optimization](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## 🏗️ System Architecture Overview {#architecture}

```
┌─────────────────┐
│  React Frontend │
│  (Audio I/O)    │
└────────┬────────┘
         │ WebSocket
         ▼
┌─────────────────────────┐
│   FastAPI Backend       │
│                         │
│  ┌──────────────────┐  │
│  │ Audio Preprocessor│  │
│  └────────┬─────────┘  │
│           ▼            │
│  ┌──────────────────┐  │
│  │ Whisper ASR      │  │
│  │ (Fine-tuned)     │  │
│  └────────┬─────────┘  │
│           ▼            │
│  ┌──────────────────┐  │
│  │ Text Processing  │  │
│  └────────┬─────────┘  │
│           ▼            │
│  ┌──────────────────┐  │
│  │ VITS/Coqui TTS   │  │
│  └────────┬─────────┘  │
│           ▼            │
│  ┌──────────────────┐  │
│  │ Audio Output     │  │
│  └──────────────────┘  │
└─────────────────────────┘
```

**Technology Stack:**
- **ASR**: Whisper (fine-tuned on impaired speech)
- **TTS**: Coqui TTS (VITS) or Bark
- **Backend**: FastAPI + WebSockets
- **Frontend**: React + Web Audio API
- **ML Framework**: PyTorch with Metal acceleration
- **Cloud**: Azure Blob Storage (dataset/models only)

---

## 🛠️ Environment Setup {#environment-setup}

### Step 1: Install System Dependencies

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11 (recommended for Metal support)
brew install python@3.11

# Install ffmpeg (required for audio processing)
brew install ffmpeg

# Install portaudio (for PyAudio)
brew install portaudio

# Install git-lfs (for large model files)
brew install git-lfs
git lfs install
```

### Step 2: Create Project Structure

```bash
# Create main project directory
mkdir -p ~/Projects/speech-impairment-converter
cd ~/Projects/speech-impairment-converter

# Create directory structure
mkdir -p {data/{raw,processed,manifests},models/{whisper,tts},backend,frontend,notebooks,scripts,logs}

# Initialize git repository
git init
echo "*.pyc\n__pycache__/\n.DS_Store\n*.ckpt\n*.pt\ndata/raw/\nlogs/\n.env\nnode_modules/" > .gitignore
```

**Final Directory Structure:**
```
speech-impairment-converter/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed audio
│   └── manifests/        # CSV/JSON metadata
├── models/
│   ├── whisper/          # Fine-tuned Whisper checkpoints
│   └── tts/              # TTS model weights
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models.py
│   │   └── websocket.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
├── notebooks/           # Jupyter notebooks for experimentation
├── scripts/            # Training/preprocessing scripts
└── logs/              # Training logs
```

### Step 3: Python Environment Setup

```bash
# Install uv (modern fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
cd ~/Projects/speech-impairment-converter
uv venv --python 3.11
source .venv/bin/activate

# Create requirements.txt for ML training
cat > requirements-train.txt << 'EOF'
torch==2.2.0
torchaudio==2.2.0
transformers==4.38.0
datasets==2.17.0
accelerate==0.27.0
librosa==0.10.1
soundfile==0.12.1
pydub==0.25.1
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
jupyter==1.0.0
tensorboard==2.16.2
wandb==0.16.3
evaluate==0.4.1
jiwer==3.0.3
onnx==1.15.0
onnxruntime==1.17.0
optimum==1.17.0
EOF

# Install training dependencies
uv pip install -r requirements-train.txt

# Verify PyTorch Metal support
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Expected Output:**
```
PyTorch version: 2.2.0
MPS available: True
```

### Step 4: Install Whisper and Fine-tuning Tools

```bash
# Install OpenAI Whisper
uv pip install openai-whisper

# Clone Whisper fine-tuning repository
cd ~/Projects
git clone https://github.com/huggingface/transformers.git
cd transformers
uv pip install -e .

# Test Whisper installation
python -c "import whisper; model = whisper.load_model('base'); print('Whisper installed successfully')"
```

### Step 5: Install TTS System (Coqui TTS)

```bash
# Install Coqui TTS (best open-source TTS for local use)
uv pip install TTS

# Test installation
tts --list_models

# Download a pre-trained model for testing
tts --text "Hello, this is a test." --model_name tts_models/en/ljspeech/vits --out_path test_output.wav
```

### Step 6: VSCode Setup

```bash
# Install VSCode extensions
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension esbenp.prettier-vscode
code --install-extension dbaeumer.vscode-eslint

# Create VSCode settings
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  }
}
EOF
```

---

## 📊 Dataset Acquisition & Preparation {#dataset-prep}

### Step 1: Download Open-Source Dysarthric Speech Datasets

**Primary Datasets:**

1. **UASpeech** (Universally Accessible Speech)
   - 16 speakers with cerebral palsy
   - ~300 hours of speech
   - Transcriptions included

2. **TORGO** (University of Toronto)
   - 8 speakers with cerebral palsy or ALS
   - ~30 hours of speech
   - High-quality recordings

3. **EasyCall Corpus**
   - Elderly speech with various impairments
   - ~50 hours

**Download Script:**

```bash
# Create download script
cat > scripts/download_datasets.sh << 'EOF'
#!/bin/bash

echo "Downloading speech impairment datasets..."

# Create directories
mkdir -p data/raw/{uaspeech,torgo,easycall}

# UASpeech (requires manual request from http://www.isle.illinois.edu/sst/data/UASpeech/)
echo "⚠️  UASpeech requires manual request. Visit: http://www.isle.illinois.edu/sst/data/UASpeech/"
echo "Download and extract to: data/raw/uaspeech/"

# TORGO (publicly available)
echo "Downloading TORGO dataset..."
cd data/raw/torgo
wget http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.tar.gz
tar -xzf torgo.tar.gz
rm torgo.tar.gz

# Alternative: Use Hugging Face datasets
echo "Alternative: Download from Hugging Face..."
python << PYTHON
from datasets import load_dataset
import os

# Download a smaller open dataset for initial testing
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train[:1000]")
dataset.save_to_disk("data/raw/common_voice_sample")
print("Sample dataset downloaded for testing")
PYTHON

echo "✅ Dataset download complete (except UASpeech - requires manual download)"
EOF

chmod +x scripts/download_datasets.sh
./scripts/download_datasets.sh
```

### Step 2: Dataset Preprocessing

**Create Preprocessing Script:**

```bash
cat > scripts/preprocess_audio.py << 'EOF'
"""
Audio preprocessing for speech impairment datasets
Standardizes audio format, sample rate, and creates manifests
"""

import os
import json
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Configuration
SAMPLE_RATE = 16000
TARGET_DIR = Path("data/processed")
MANIFEST_DIR = Path("data/manifests")
RAW_DIR = Path("data/raw")

def preprocess_audio(input_path, output_path):
    """Load, resample, and normalize audio"""
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        
        # Normalize to [-1, 1]
        audio = librosa.util.normalize(audio)
        
        # Remove silence (optional but recommended)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Save
        sf.write(output_path, audio, SAMPLE_RATE)
        return len(audio) / SAMPLE_RATE  # duration in seconds
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None

def create_manifest(dataset_name, data_dir, transcript_file=None):
    """Create training manifest (CSV) for dataset"""
    
    manifest_data = []
    audio_dir = data_dir / "audio"
    
    # Load transcripts if available
    transcripts = {}
    if transcript_file and transcript_file.exists():
        with open(transcript_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    transcripts[parts[0]] = parts[1]
    
    # Process each audio file
    audio_files = list(audio_dir.glob("**/*.wav"))
    for audio_path in tqdm(audio_files, desc=f"Processing {dataset_name}"):
        # Get file ID
        file_id = audio_path.stem
        
        # Process audio
        output_path = TARGET_DIR / dataset_name / f"{file_id}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        duration = preprocess_audio(audio_path, output_path)
        
        if duration:
            manifest_data.append({
                'audio_path': str(output_path),
                'duration': duration,
                'transcript': transcripts.get(file_id, ''),
                'speaker_id': audio_path.parent.name,
                'dataset': dataset_name
            })
    
    # Save manifest
    df = pd.DataFrame(manifest_data)
    manifest_path = MANIFEST_DIR / f"{dataset_name}_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    
    print(f"✅ Created manifest: {manifest_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Total duration: {df['duration'].sum() / 3600:.2f} hours")
    
    return df

def split_dataset(manifest_df, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train/val/test"""
    
    # Shuffle
    manifest_df = manifest_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n = len(manifest_df)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    # Split
    train_df = manifest_df[:train_idx]
    val_df = manifest_df[train_idx:val_idx]
    test_df = manifest_df[val_idx:]
    
    # Save splits
    dataset_name = manifest_df['dataset'].iloc[0]
    train_df.to_csv(MANIFEST_DIR / f"{dataset_name}_train.csv", index=False)
    val_df.to_csv(MANIFEST_DIR / f"{dataset_name}_val.csv", index=False)
    test_df.to_csv(MANIFEST_DIR / f"{dataset_name}_test.csv", index=False)
    
    print(f"✅ Split dataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    # Example: Process TORGO dataset
    torgo_dir = RAW_DIR / "torgo"
    if torgo_dir.exists():
        manifest = create_manifest("torgo", torgo_dir)
        split_dataset(manifest)
    
    print("✅ Preprocessing complete!")
EOF

python scripts/preprocess_audio.py
```

### Step 3: Data Quality Checks

```bash
cat > scripts/validate_data.py << 'EOF'
"""Validate preprocessed data quality"""

import pandas as pd
import librosa
from pathlib import Path
import matplotlib.pyplot as plt

def validate_manifest(manifest_path):
    """Check manifest for issues"""
    df = pd.read_csv(manifest_path)
    
    print(f"\n📊 Validating {manifest_path.name}")
    print(f"   Total samples: {len(df)}")
    print(f"   Duration stats:")
    print(f"     - Mean: {df['duration'].mean():.2f}s")
    print(f"     - Min: {df['duration'].min():.2f}s")
    print(f"     - Max: {df['duration'].max():.2f}s")
    print(f"   Missing transcripts: {df['transcript'].isna().sum()}")
    
    # Check for corrupted files
    corrupted = []
    for _, row in df.iterrows():
        try:
            librosa.load(row['audio_path'], sr=16000, duration=0.1)
        except:
            corrupted.append(row['audio_path'])
    
    if corrupted:
        print(f"   ⚠️  Corrupted files: {len(corrupted)}")
    else:
        print(f"   ✅ All files valid")
    
    # Duration distribution
    plt.figure(figsize=(10, 4))
    plt.hist(df['duration'], bins=50)
    plt.xlabel('Duration (s)')
    plt.ylabel('Count')
    plt.title(f'Duration Distribution - {manifest_path.stem}')
    plt.savefig(f'logs/{manifest_path.stem}_duration_dist.png')
    print(f"   📈 Saved duration plot to logs/")

if __name__ == "__main__":
    manifest_dir = Path("data/manifests")
    for manifest in manifest_dir.glob("*_train.csv"):
        validate_manifest(manifest)
EOF

python scripts/validate_data.py
```

**Expected Data Requirements:**
- **Minimum**: 10 hours of impaired speech for fine-tuning
- **Recommended**: 30-50 hours for production quality
- **Audio Format**: 16kHz, mono, WAV
- **Transcript Quality**: WER < 5% if using existing transcripts

---

## 🎯 Whisper ASR Fine-Tuning {#whisper-training}

### Step 1: Prepare Training Script

```bash
cat > scripts/train_whisper.py << 'EOF'
"""
Fine-tune Whisper on dysarthric speech
Uses HuggingFace transformers with Metal (MPS) acceleration
"""

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from datasets import Dataset, Audio
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List
import evaluate
import numpy as np

# Configuration
MODEL_NAME = "openai/whisper-small"  # or "whisper-base" for faster training
OUTPUT_DIR = "models/whisper/checkpoints"
MAX_AUDIO_LENGTH = 30  # seconds
SAMPLE_RATE = 16000

# Check for Metal support
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def load_dataset_from_manifest(manifest_path):
    """Load audio dataset from CSV manifest"""
    df = pd.read_csv(manifest_path)
    
    # Create HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=SAMPLE_RATE))
    dataset = dataset.rename_column("audio_path", "audio")
    
    return dataset

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Custom data collator for Whisper"""
    
    processor: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract audio and pad
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Extract labels and pad
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch

def prepare_dataset(batch, processor):
    """Preprocess audio and text"""
    # Load audio
    audio = batch["audio"]
    
    # Compute input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode text labels
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
    
    return batch

def compute_metrics(pred, processor, metric):
    """Calculate WER during training"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Calculate WER
    wer = metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

def train_whisper():
    """Main training function"""
    
    # Load processor and model
    print("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Freeze encoder (optional - faster training, may reduce quality slightly)
    # model.freeze_encoder()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset_from_manifest("data/manifests/torgo_train.csv")
    val_dataset = load_dataset_from_manifest("data/manifests/torgo_val.csv")
    
    # Preprocess
    print("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=val_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Metrics
    wer_metric = evaluate.load("wer")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,  # Adjust based on M3 Pro memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=10,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=False,  # Metal doesn't support fp16
        report_to=["tensorboard"],
        push_to_hub=False,
        generation_max_length=225,
        predict_with_generate=True,
        use_mps_device=True if device == "mps" else False
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, wer_metric),
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("🚀 Starting training...")
    trainer.train()
    
    # Save final model
    print("💾 Saving model...")
    trainer.save_model("models/whisper/final")
    processor.save_pretrained("models/whisper/final")
    
    print("✅ Training complete!")

if __name__ == "__main__":
    train_whisper()
EOF
```

### Step 2: Run Training

```bash
# Start training (this will take several hours)
python scripts/train_whisper.py

# Monitor training in TensorBoard
tensorboard --logdir models/whisper/checkpoints
```

**Training Hyperparameters Explained:**
- **Batch Size**: 4 per device (M3 Pro has ~18GB unified memory)
- **Gradient Accumulation**: 4 steps (effective batch size = 16)
- **Learning Rate**: 1e-5 (conservative for fine-tuning)
- **Epochs**: 10 (with early stopping)
- **Warmup Steps**: 500 (gradual learning rate increase)

**Expected Training Time:**
- **10 hours of data**: ~4-6 hours on M3 Pro
- **50 hours of data**: ~20-24 hours

### Step 3: Export to ONNX (Optional - for faster inference)

```bash
cat > scripts/export_whisper_onnx.py << 'EOF'
"""Export Whisper to ONNX for optimized inference"""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

# Load trained model
model_path = "models/whisper/final"
print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(model_path)
processor = WhisperProcessor.from_pretrained(model_path)

# Export to ONNX
print("Exporting to ONNX...")
onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
    model_path,
    export=True
)

# Save
onnx_path = "models/whisper/onnx"
onnx_model.save_pretrained(onnx_path)
processor.save_pretrained(onnx_path)

print(f"✅ ONNX model saved to {onnx_path}")
EOF

python scripts/export_whisper_onnx.py
```

### Step 4: Test Fine-tuned Model

```bash
cat > scripts/test_whisper.py << 'EOF'
"""Test fine-tuned Whisper on held-out test set"""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import pandas as pd
from jiwer import wer
from tqdm import tqdm

def test_model(model_path, test_manifest_path):
    # Load model
    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    
    # Load test data
    test_df = pd.read_csv(test_manifest_path)
    
    predictions = []
    references = []
    
    print("Running inference on test set...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Load audio
        audio, _ = librosa.load(row['audio_path'], sr=16000)
        
        # Process
        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        predictions.append(transcription)
        references.append(row['transcript'])
    
    # Calculate WER
    test_wer = wer(references, predictions)
    
    print(f"\n📊 Test Results:")
    print(f"   WER: {test_wer * 100:.2f}%")
    print(f"\n📝 Sample Predictions:")
    for i in range(min(5, len(predictions))):
        print(f"   Reference:  {references[i]}")
        print(f"   Predicted:  {predictions[i]}")
        print()
    
    return test_wer

if __name__ == "__main__":
    wer_score = test_model(
        "models/whisper/final",
        "data/manifests/torgo_test.csv"
    )
EOF

python scripts/test_whisper.py
```

---

## 🎤 TTS/Voice Cloning Setup {#tts-setup}

### Step 1: Choose TTS System

**Recommended: Coqui TTS (VITS)**
- Fast inference (~200ms on M3 Pro)
- High quality
- Easy to fine-tune
- Active community

**Alternative: Bark**
- More natural prosody
- Slower inference (~2-3s)
- No fine-tuning needed

### Step 2: Fine-tune VITS on Target Voice

```bash
cat > scripts/train_tts.py << 'EOF'
"""
Fine-tune VITS TTS model for natural speech generation
"""

import os
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.datasets import load_tts_samples
from TTS.trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor

# Configuration
OUTPUT_PATH = "models/tts/"
DATASET_PATH = "data/tts_training/"  # Clean speech samples for target voice

def prepare_tts_dataset():
    """
    Prepare clean speech samples for TTS training
    Use LJSpeech or LibriTTS for generic voice, or record custom samples
    """
    print("📥 Downloading LJSpeech dataset (if needed)...")
    os.system("bash scripts/download_ljspeech.sh")

def train_vits():
    """Train VITS TTS model"""
    
    # Audio processor config
    audio_config = {
        "sample_rate": 22050,
        "num_mels": 80,
        "mel_fmin": 0,
        "mel_fmax": 8000,
        "hop_length": 256,
        "win_length": 1024
    }
    
    # Model config
    config = VitsConfig(
        audio=audio_config,
        run_name="vits_speech_converter",
        batch_size=16,
        eval_batch_size=8,
        num_loader_workers=4,
        num_epochs=1000,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        compute_input_seq_cache=True,
        print_step=50,
        save_step=1000,
        mixed_precision=False,
        output_path=OUTPUT_PATH,
        datasets=[
            {
                "formatter": "ljspeech",
                "meta_file_train": "metadata.csv",
                "path": DATASET_PATH
            }
        ]
    )
    
    # Initialize audio processor
    ap = AudioProcessor(**config.audio.to_dict())
    
    # Load data samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_size=0.1
    )
    
    # Initialize model
    model = Vits(config, ap)
    
    # Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )
    
    # Train
    print("🚀 Starting TTS training...")
    trainer.fit()
    
    print("✅ TTS training complete!")

if __name__ == "__main__":
    prepare_tts_dataset()
    train_vits()
EOF
```

### Step 3: Quick Start with Pre-trained TTS

```bash
# For rapid prototyping, use pre-trained Coqui TTS
cat > scripts/setup_tts_pretrained.py << 'EOF'
"""Setup pre-trained TTS for immediate use"""

from TTS.api import TTS
import torch

# Initialize TTS with best English model
device = "mps" if torch.backends.mps.is_available() else "cpu"
tts = TTS("tts_models/en/ljspeech/vits", progress_bar=True).to(device)

# Test synthesis
text = "This is a test of the speech synthesis system."
output_path = "models/tts/test_output.wav"
tts.tts_to_file(text=text, file_path=output_path)

print(f"✅ TTS model ready! Test output saved to {output_path}")
print(f"   Device: {device}")
print(f"   Model: VITS (LJSpeech)")
EOF

python scripts/setup_tts_pretrained.py
```

### Step 4: Voice Cloning (Optional - for personalized output)

```bash
cat > scripts/voice_clone.py << 'EOF'
"""
Voice cloning with YourTTS or XTTS
Allows generating speech in a target speaker's voice
"""

from TTS.api import TTS

# Use XTTS v2 for zero-shot voice cloning
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Clone voice from reference audio (3-10 seconds of clean speech)
reference_audio = "data/reference_voices/target_speaker.wav"
output_path = "output/cloned_speech.wav"

text = "Hello, this is synthesized speech in a cloned voice."

tts.tts_to_file(
    text=text,
    speaker_wav=reference_audio,
    language="en",
    file_path=output_path
)

print(f"✅ Voice cloned! Output: {output_path}")
EOF
```

---

## ⚡ Backend Development (FastAPI) {#backend}

### Step 1: Backend Structure

```bash
cd backend

# Create backend requirements
cat > requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
websockets==12.0
pydantic==2.6.0
numpy==1.26.4
torch==2.2.0
torchaudio==2.2.0
transformers==4.38.0
TTS==0.22.0
librosa==0.10.1
soundfile==0.12.1
python-dotenv==1.0.0
aiofiles==23.2.1
EOF

uv pip install -r requirements.txt

# Create app structure
mkdir -p app/{api,core,models,services,utils}
touch app/{__init__.py,api/__init__.py,core/__init__.py,models/__init__.py,services/__init__.py,utils/__init__.py}
```

### Step 2: Main Application

```python
# app/main.py
cat > app/main.py << 'EOF'
"""
FastAPI backend for Speech Impairment Converter
Handles audio streaming, ASR inference, and TTS synthesis
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import logging
from pathlib import Path

from app.services.asr_service import ASRService
from app.services.tts_service import TTSService
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Speech Impairment Converter API",
    description="Convert impaired speech to clear speech using ML",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
asr_service = ASRService(
    model_path="models/whisper/final",
    device=settings.DEVICE
)
tts_service = TTSService(
    model_name=settings.TTS_MODEL,
    device=settings.DEVICE
)

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting Speech Converter API...")
    await asr_service.load_model()
    await tts_service.load_model()
    logger.info("✅ Models loaded successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Speech Impairment Converter API",
        "models": {
            "asr": "loaded" if asr_service.model else "not loaded",
            "tts": "loaded" if tts_service.model else "not loaded"
        }
    }

@app.post("/api/convert")
async def convert_audio(audio: UploadFile = File(...)):
    """
    Single audio file conversion
    Upload impaired speech, get clear speech back
    """
    try:
        # Read audio file
        audio_bytes = await audio.read()
        
        # ASR: Impaired speech → Text
        logger.info("Running ASR...")
        transcript = await asr_service.transcribe(audio_bytes)
        logger.info(f"Transcript: {transcript}")
        
        # TTS: Text → Clear speech
        logger.info("Running TTS...")
        clear_audio = await tts_service.synthesize(transcript)
        
        return {
            "transcript": transcript,
            "audio_duration": len(clear_audio) / 22050,  # seconds
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return {"status": "error", "message": str(e)}

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming
    Client sends audio chunks → Server responds with clear audio
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        audio_buffer = bytearray()
        
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            # Process when buffer reaches threshold (e.g., 3 seconds)
            if len(audio_buffer) >= 16000 * 3 * 2:  # 3 seconds at 16kHz, 16-bit
                # Transcribe
                transcript = await asr_service.transcribe(bytes(audio_buffer))
                
                if transcript.strip():
                    # Synthesize clear speech
                    clear_audio = await tts_service.synthesize(transcript)
                    
                    # Send back
                    await websocket.send_json({
                        "transcript": transcript,
                        "audio": clear_audio.tolist(),  # Send as list for JSON
                        "sample_rate": 22050
                    })
                
                # Clear buffer
                audio_buffer.clear()
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
EOF
```

### Step 3: ASR Service

```python
# app/services/asr_service.py
cat > app/services/asr_service.py << 'EOF'
"""ASR (Whisper) inference service"""

import torch
import numpy as np
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import logging

logger = logging.getLogger(__name__)

class ASRService:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
    
    async def load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper from {self.model_path}...")
            self.processor = WhisperProcessor.from_pretrained(self.model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Whisper model loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            Transcribed text
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample to 16kHz if needed
            audio_array = librosa.resample(audio_array, orig_sr=16000, target_sr=16000)
            
            # Process audio
            input_features = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=225,
                    num_beams=5,
                    temperature=0.0
                )
            
            # Decode
            transcript = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            return transcript.strip()
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
EOF
```

### Step 4: TTS Service

```python
# app/services/tts_service.py
cat > app/services/tts_service.py << 'EOF'
"""TTS synthesis service"""

import torch
import numpy as np
from TTS.api import TTS
import logging
import io
import soundfile as sf

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/vits", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
    
    async def load_model(self):
        """Load TTS model"""
        try:
            logger.info(f"Loading TTS model: {self.model_name}...")
            self.model = TTS(self.model_name, progress_bar=False).to(self.device)
            logger.info("✅ TTS model loaded")
        except Exception as e:
            logger.error(f"Failed to load TTS: {e}")
            raise
    
    async def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize clear speech from text
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Audio array (numpy)
        """
        try:
            # Generate audio
            wav = self.model.tts(text)
            
            # Convert to numpy array
            if isinstance(wav, list):
                wav = np.array(wav, dtype=np.float32)
            
            return wav
        
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return np.array([], dtype=np.float32)
EOF
```

### Step 5: Configuration

```python
# app/core/config.py
cat > app/core/config.py << 'EOF'
"""Application configuration"""

import torch
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Speech Impairment Converter"
    
    # Model paths
    WHISPER_MODEL_PATH: str = "models/whisper/final"
    TTS_MODEL: str = "tts_models/en/ljspeech/vits"
    
    # Device configuration
    DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Audio settings
    SAMPLE_RATE: int = 16000
    TTS_SAMPLE_RATE: int = 22050
    
    class Config:
        case_sensitive = True

settings = Settings()
EOF
```

### Step 6: Run Backend

```bash
# Start the backend server
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Test the API
curl http://localhost:8000/
```

---

## 🎨 Frontend Development (React) {#frontend}

### Step 1: Initialize React App

```bash
cd ~/Projects/speech-impairment-converter

# Create React app with Vite (faster than create-react-app)
npm create vite@latest frontend -- --template react
cd frontend
npm install

# Install dependencies
npm install axios recharts lucide-react
```

### Step 2: Audio Recorder Component

```javascript
// frontend/src/components/AudioRecorder.jsx
cat > src/components/AudioRecorder.jsx << 'EOF'
import { useState, useRef, useEffect } from 'react';
import { Mic, Square, Play, Download } from 'lucide-react';

export default function AudioRecorder({ onAudioCaptured }) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 16000,
        } 
      });
      
      streamRef.current = stream;
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(audioBlob);
        setAudioURL(url);
        
        // Send to backend
        if (onAudioCaptured) {
          setIsProcessing(true);
          await onAudioCaptured(audioBlob);
          setIsProcessing(false);
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Could not access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      streamRef.current?.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  return (
    <div className="audio-recorder">
      <div className="controls">
        {!isRecording ? (
          <button 
            onClick={startRecording}
            className="btn btn-primary"
            disabled={isProcessing}
          >
            <Mic size={24} />
            Start Recording
          </button>
        ) : (
          <button 
            onClick={stopRecording}
            className="btn btn-danger"
          >
            <Square size={24} />
            Stop Recording
          </button>
        )}
      </div>

      {isProcessing && (
        <div className="processing">
          <div className="spinner"></div>
          <p>Processing audio...</p>
        </div>
      )}

      {audioURL && !isProcessing && (
        <div className="playback">
          <audio src={audioURL} controls />
          <a href={audioURL} download="recorded_audio.wav" className="btn btn-secondary">
            <Download size={20} />
            Download
          </a>
        </div>
      )}
    </div>
  );
}
EOF
```

### Step 3: Main App Component

```javascript
// frontend/src/App.jsx
cat > src/App.jsx << 'EOF'
import { useState } from 'react';
import AudioRecorder from './components/AudioRecorder';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [transcript, setTranscript] = useState('');
  const [clearAudioURL, setClearAudioURL] = useState(null);
  const [error, setError] = useState(null);

  const handleAudioCaptured = async (audioBlob) => {
    try {
      setError(null);
      
      // Create form data
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recorded_audio.wav');

      // Send to backend
      const response = await axios.post(`${API_URL}/api/convert`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Update UI
      setTranscript(response.data.transcript);
      
      // For now, we'll use the original audio
      // In production, backend should return the synthesized audio
      console.log('Conversion successful:', response.data);
      
    } catch (err) {
      console.error('Error processing audio:', err);
      setError('Failed to process audio. Please try again.');
    }
  };

  return (
    <div className="App">
      <header>
        <h1>🎤 Speech Impairment Converter</h1>
        <p>Convert impaired speech to clear, natural speech</p>
      </header>

      <main>
        <section className="recorder-section">
          <h2>Step 1: Record Your Speech</h2>
          <AudioRecorder onAudioCaptured={handleAudioCaptured} />
        </section>

        {transcript && (
          <section className="transcript-section">
            <h2>Step 2: Transcription</h2>
            <div className="transcript-box">
              <p>{transcript}</p>
            </div>
          </section>
        )}

        {clearAudioURL && (
          <section className="output-section">
            <h2>Step 3: Clear Speech Output</h2>
            <audio src={clearAudioURL} controls />
          </section>
        )}

        {error && (
          <div className="error">
            <p>{error}</p>
          </div>
        )}
      </main>

      <footer>
        <p>Built with Whisper ASR + Coqui TTS</p>
      </footer>
    </div>
  );
}

export default App;
EOF
```

### Step 4: Styling

```css
// frontend/src/App.css
cat > src/App.css << 'EOF'
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

.App {
  max-width: 900px;
  margin: 0 auto;
  padding: 2rem;
}

header {
  text-align: center;
  color: white;
  margin-bottom: 3rem;
}

header h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

main {
  background: white;
  border-radius: 20px;
  padding: 2rem;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

section {
  margin-bottom: 2rem;
}

section h2 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.5rem;
}

.audio-recorder {
  text-align: center;
}

.controls {
  margin: 2rem 0;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem 2rem;
  font-size: 1.1rem;
  border: none;
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 600;
}

.btn-primary {
  background: #667eea;
  color: white;
}

.btn-primary:hover {
  background: #5568d3;
  transform: translateY(-2px);
  box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
}

.btn-danger {
  background: #e74c3c;
  color: white;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}

.btn-secondary {
  background: #95a5a6;
  color: white;
  padding: 0.5rem 1rem;
  font-size: 0.9rem;
  text-decoration: none;
  margin-top: 1rem;
}

.processing {
  margin: 2rem 0;
}

.spinner {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.transcript-box {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 10px;
  border-left: 4px solid #667eea;
}

.transcript-box p {
  font-size: 1.1rem;
  line-height: 1.6;
  color: #333;
}

audio {
  width: 100%;
  margin-top: 1rem;
}

.error {
  background: #fee;
  border: 2px solid #e74c3c;
  border-radius: 10px;
  padding: 1rem;
  color: #c0392b;
  margin-top: 1rem;
}

footer {
  text-align: center;
  color: white;
  margin-top: 2rem;
  opacity: 0.8;
}

.playback {
  margin-top: 2rem;
}

@media (max-width: 768px) {
  .App {
    padding: 1rem;
  }
  
  header h1 {
    font-size: 2rem;
  }
  
  main {
    padding: 1.5rem;
  }
}
EOF
```

### Step 5: Run Frontend

```bash
cd frontend
npm run dev

# Open browser to http://localhost:5173
```

---

## 🔗 Integration & Testing {#integration}

### Step 1: End-to-End Test Script

```bash
cat > scripts/test_pipeline.py << 'EOF'
"""
End-to-end pipeline test
Tests the complete flow: Audio → ASR → TTS → Audio
"""

import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000"

def test_conversion_api():
    """Test the /api/convert endpoint"""
    
    print("🧪 Testing Speech Conversion Pipeline...")
    
    # Load test audio
    test_audio_path = "data/test/sample_impaired_speech.wav"
    
    if not Path(test_audio_path).exists():
        print(f"❌ Test audio not found: {test_audio_path}")
        return
    
    # Send request
    with open(test_audio_path, 'rb') as f:
        files = {'audio': ('test.wav', f, 'audio/wav')}
        
        print("📤 Sending audio to API...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_URL}/api/convert",
            files=files
        )
        
        elapsed = time.time() - start_time
    
    # Check response
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Conversion successful!")
        print(f"   Transcript: {data['transcript']}")
        print(f"   Duration: {data.get('audio_duration', 'N/A')}s")
        print(f"   Latency: {elapsed:.2f}s")
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(f"   {response.text}")

def test_health_check():
    """Test API health check"""
    response = requests.get(f"{API_URL}/")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API is healthy")
        print(f"   ASR Model: {data['models']['asr']}")
        print(f"   TTS Model: {data['models']['tts']}")
    else:
        print(f"❌ API health check failed")

if __name__ == "__main__":
    test_health_check()
    print()
    test_conversion_api()
EOF

python scripts/test_pipeline.py
```

### Step 2: Performance Benchmarking

```bash
cat > scripts/benchmark.py << 'EOF'
"""Benchmark inference speed and quality"""

import time
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from TTS.api import TTS
import librosa

def benchmark_whisper(model_path, test_audio_path, n_iterations=10):
    """Benchmark Whisper inference speed"""
    
    print("⏱️  Benchmarking Whisper ASR...")
    
    # Load model
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load audio
    audio, _ = librosa.load(test_audio_path, sr=16000)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(input_features)
    
    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(input_features)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"   Average inference time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"   Real-time factor: {(len(audio)/16000)/avg_time:.2f}x")
    
    return avg_time

def benchmark_tts(text="This is a test of the text to speech system.", n_iterations=10):
    """Benchmark TTS inference speed"""
    
    print("\n⏱️  Benchmarking TTS...")
    
    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tts = TTS("tts_models/en/ljspeech/vits").to(device)
    
    # Warmup
    _ = tts.tts(text)
    
    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.time()
        _ = tts.tts(text)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"   Average synthesis time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    
    return avg_time

if __name__ == "__main__":
    whisper_time = benchmark_whisper(
        "models/whisper/final",
        "data/test/sample_audio.wav"
    )
    
    tts_time = benchmark_tts()
    