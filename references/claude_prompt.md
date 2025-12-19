Here is a **copy-paste ready, high-quality prompt for Claude** that will force it to produce a *full, step-by-step development environment setup + full project completion guide*, tailored for your constraints (Azure student plan, local M3 Pro GPU, open-source models/datasets, full walkthrough).

---

# ✅ **Claude Prompt (Copy & Paste)**

**You are an expert ML engineer and full-stack architect. Your task is to give me an extremely detailed, opinionated, step-by-step guide to set up the full development environment and implement the following project end-to-end:**

> **A Speech-Impairment → Clear Speech Converter**
> A system that takes impaired/slurred speech and converts it into clear, natural speech — built with *open-source models, datasets, and tools*.
> The final MVP must include:
> • A fine-tuned open-source ASR model (Whisper) for impaired speech
> • A voice-cloning / TTS model (open source, NOT Azure Custom Neural Voice)
> • A streaming backend (FastAPI + WebSockets)
> • A Web client (React) that records audio and plays corrected speech
> • Real-time or near-real-time inference
> • A simple inference pipeline: audio → ASR → text → TTS → clear audio

---

## **My Constraints (follow exactly):**

### **💻 Hardware**

* I have an **Apple Silicon MacBook Pro (M3 Pro)** that I can use for training/inference.
* I prefer to run **Whisper fine-tuning locally** using Metal acceleration unless Azure is necessary.

### **☁ Cloud Access**

* I only have **Azure Student Plan credits** (limited).
* So:

  * Use Azure **only where absolutely necessary** (storage, basic VM).
  * Avoid expensive GPU compute on Azure unless unavoidable.

### **📦 Models & Datasets**

Give me EXACT instructions using only **open-source** components, including:

* ASR dataset for impaired speech (UASpeech, TORGO, ALS datasets, etc.)
* Whisper fine-tuning
* An open-source TTS system (VITS, SoVITS, RVC, Bark, etc.)
* No proprietary datasets or paid models.

### **🧩 Architecture Requirements**

Your answer **must** walk me through:

1. Full environment setup

   * Python
   * Conda or uv
   * CUDA/Metal configuration for Whisper training
   * Brew dependencies
   * GitHub repo structure
   * VSCode setup

2. Local development workflow

   * Installing PyTorch for Apple Silicon
   * Installing Whisper + fine-tuning tools
   * Installing an open-source TTS (recommend the best one for my hardware)

3. Dataset preparation

   * Where to download datasets
   * How to preprocess them
   * How to create CSV manifests
   * How to split train/val/test
   * How much data I need

4. Whisper fine-tuning

   * Full command-line instructions
   * Hyperparameters
   * Training loop explanation
   * How to monitor loss
   * How to export to ONNX for low-latency inference

5. TTS/Voice-Cloning training

   * How to train a small open-source TTS/voice model locally
   * Exact commands
   * Dataset preparation
   * How to generate samples

6. Backend creation

   * FastAPI structure
   * REST endpoints
   * WebSocket endpoint for streaming audio
   * How to load ONNX Whisper and the TTS model
   * How to run inference in real time

7. Frontend implementation

   * React setup
   * Audio capture
   * WebSocket streaming
   * Audio playback buffer
   * Clean UI

8. Putting everything together

   * Local end-to-end pipeline test
   * Latency optimisation
   * Deployment options (local vs Azure)
   * How to package the demo video for competition submission

---

## **Style Guide**

**Write the instructions like a full professional engineering guide.**
Use:

* step-by-step bullet points
* terminal commands in code blocks
* folder structures
* architecture diagrams
* mini checklists
* troubleshooting tips
* example outputs
* recommended settings

Do not leave things vague.
Assume I will follow your instructions exactly.

---

# **End of Prompt**

(Do NOT output a short answer. Output a full, complete, extremely detailed, comprehensive guide.)


