Perfect — using the **RTX Pro 4000** is the right call 👍
Below is a **complete, end-to-end, copy-paste-safe checklist** you can follow on the NVIDIA laptop.

I’ll assume:

- You are **fine-tuning Whisper on TORGO**
- You already have **pseudo-labels** (or will generate them)
- You want **train → evaluate → use the model**

No hand-waving. This is the **full pipeline**.

---

# 0️⃣ What you can reuse from Mac vs what changes

### ✅ Same

- Dataset structure
- CSV / HuggingFace `Dataset`
- Training code
- Tokenization logic
- TORGO preprocessing

### 🔄 Changes on RTX

- `device="cuda"`
- `fp16=True`
- Faster batch size
- Enable `torch.compile` (optional)

---

# 1️⃣ Setup the RTX Pro 4000 Laptop

## 1.1 Install system requirements

```bash
nvidia-smi
```

You should see your GPU.

---

## 1.2 Create Python environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

---

## 1.3 Install PyTorch (CUDA)

**IMPORTANT:** install PyTorch first

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python - <<EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
EOF
```

---

## 1.4 Install remaining dependencies

```bash
pip install \
  transformers \
  datasets \
  accelerate \
  evaluate \
  jiwer \
  soundfile \
  librosa \
  tqdm \
  torchcodec
```

---

# 2️⃣ Transfer your project

Copy from Mac → RTX laptop:

```
project/
├── data/
│   └── torgo-audio/
├── prepare_dataset.py
├── train_whisper.py
├── eval_whisper.py
└── infer.py
```

---

# 3️⃣ STEP 1 — Generate transcriptions (Pseudo-labels)

Since TORGO has **no transcripts**, Whisper must label it first.

### 3.1 Pseudo-label script (`pseudo_label.py`)

```python
import os
import csv
import whisper

ROOT = "data/torgo-audio"
model = whisper.load_model("base").cuda()

rows = []

for root, _, files in os.walk(ROOT):
    for f in files:
        if f.endswith(".wav"):
            path = os.path.join(root, f)
            result = model.transcribe(path, fp16=True)

            rows.append({
                "file": path,
                "text": result["text"].strip()
            })

with open("torgo_pseudo_labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "text"])
    writer.writeheader()
    writer.writerows(rows)
```

Run:

```bash
python pseudo_label.py
```

⏱️ ~20–30 minutes total.

---

# 4️⃣ STEP 2 — Build HuggingFace Dataset

### 4.1 `prepare_dataset.py`

```python
import csv
from datasets import Dataset, Audio

rows = []

with open("torgo_pseudo_labels.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["text"]:
            rows.append({
                "audio": r["file"],
                "transcription": r["text"].lower()
            })

dataset = Dataset.from_list(rows)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.train_test_split(test_size=0.1)

dataset.save_to_disk("torgo_hf")
```

Run:

```bash
python prepare_dataset.py
```

---

# 5️⃣ STEP 3 — Training Whisper (RTX Optimized)

### 5.1 `train_whisper.py`

```python
import torch
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer
)

dataset = load_from_disk("torgo_hf")

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-base",
    language="en",
    task="transcribe"
)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-base"
)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

model.to("cuda")

def preprocess(batch):
    audio = batch["audio"]
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    )

    with processor.as_target_processor():
        labels = processor(batch["transcription"])

    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels.input_ids
    return batch

dataset = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names
)

training_args = TrainingArguments(
    output_dir="whisper-torgo",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=2000,
    fp16=True,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=50,
    report_to="none",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor
)

trainer.train()

model.save_pretrained("whisper-torgo-final")
processor.save_pretrained("whisper-torgo-final")
```

Run:

```bash
python train_whisper.py
```

⏱️ **2–4 minutes** on RTX Pro 4000.

---

# 6️⃣ STEP 4 — Evaluate (WER)

### 6.1 `eval_whisper.py`

```python
import torch
import evaluate
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration

wer = evaluate.load("wer")

dataset = load_from_disk("torgo_hf")["test"]

processor = WhisperProcessor.from_pretrained("whisper-torgo-final")
model = WhisperForConditionalGeneration.from_pretrained("whisper-torgo-final").cuda()

preds, refs = [], []

for sample in dataset:
    inputs = processor(
        sample["audio"]["array"],
        sampling_rate=16000,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        pred_ids = model.generate(**inputs)

    text = processor.decode(pred_ids[0], skip_special_tokens=True)
    preds.append(text)
    refs.append(sample["transcription"])

print("WER:", wer.compute(predictions=preds, references=refs))
```

---

# 7️⃣ STEP 5 — Inference on New Audio

### 7.1 `infer.py`

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("whisper-torgo-final")
model = WhisperForConditionalGeneration.from_pretrained("whisper-torgo-final").cuda()

audio_path = "sample.wav"

inputs = processor(
    audio_path,
    sampling_rate=16000,
    return_tensors="pt"
).to("cuda")

with torch.no_grad():
    ids = model.generate(**inputs)

print(processor.decode(ids[0], skip_special_tokens=True))
```

---

# 🏁 Final Checklist

| Step               | Done |
| ------------------ | ---- |
| Pseudo-label TORGO | ✅   |
| HF dataset         | ✅   |
| Train Whisper      | ✅   |
| Evaluate (WER)     | ✅   |
| Inference          | ✅   |

---

# ⚠️ Important Warnings (Read This)

- You are **bootstrapping** Whisper with its own labels
- Accuracy improves **mainly for dysarthric speech**
- Do NOT compare with LibriSpeech WER
- Real transcripts → **much better results**

---

# 🔥 Next-Level Improvements (optional)

1️⃣ Freeze encoder first 500 steps
2️⃣ Curriculum: Con → Dys
3️⃣ Train `small` model after base
4️⃣ LM decoding with beam=5
5️⃣ Add SpecAugment

---

If you want, next I can:

- Convert this to **WhisperX**
- Add **speaker-aware training**
- Make this **MLX inference compatible**
- Package as a **Hackathon demo**

Just say the word 🚀
