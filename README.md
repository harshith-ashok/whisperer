# Whisperer - MLX Edition

Small collection of scripts for transcription, language detection, TTS and simple translation but for Apple silicon chips.

Quick start

- Install system dependency: `ffmpeg` (macOS: `brew install ffmpeg`).
- Install Python dependencies:

```
pip install -r requirements.txt
```

Note: For `torch` follow the official install guide if you need CUDA builds: https://pytorch.org/get-started/locally/

Common scripts

- `audio_converter/transcribe.py` — transcribe `output.wav` with Whisper, writes `output.txt`.
- `audio_converter/language.py` — detect language and decode audio with Whisper.
- `audio_converter/sampler.py` — create a TTS sample using `gTTS` and save `output.wav`.
- `audio_converter/translator.py` — translate first line of `output.txt` to English using `googletrans`.
- `audio_converter/torgo/whisperer.py` — batch-transcribe TORGO dataset WAVs and write `torgo_whisper.csv`.

Usage example

```
python audio_converter/transcribe.py
python audio_converter/sampler.py
python audio_converter/translator.py
```

## Torgo Dataset

from `https://www.kaggle.com/datasets/pranaykoppula/torgo-audio?resource=download`

Downloaded using

```bash
kaggle datasets download pranaykoppula/torgo-audio
```

```
.
├── F_Con
│   ├── wav_arrayMic_FC01S01
│   ├── wav_arrayMic_FC02S02
│   ├── wav_arrayMic_FC02S03
│   ├── wav_arrayMic_FC03S01
│   ├── wav_arrayMic_FC03S02
│   ├── wav_arrayMic_FC03S03
│   ├── wav_headMic_FC01S01
│   ├── wav_headMic_FC02S03
│   ├── wav_headMic_FC03S01
│   ├── wav_headMic_FC03S02
│   └── wav_headMic_FC03S03
├── F_Dys
│   ├── wav_arrayMic_F01
│   ├── wav_arrayMic_F03S01
│   ├── wav_arrayMic_F03S02
│   ├── wav_arrayMic_F03S03
│   ├── wav_arrayMic_F04S01
│   ├── wav_arrayMic_F04S02
│   ├── wav_headMic_F01
│   ├── wav_headMic_F03S01
│   ├── wav_headMic_F03S02
│   ├── wav_headMic_F03S03
│   └── wav_headMic_F04S02
├── M_Con
│   ├── wav_arrayMic_MC01S01
│   ├── wav_arrayMic_MC01S02
│   ├── wav_arrayMic_MC01S03
│   ├── wav_arrayMic_MC02S01
│   ├── wav_arrayMic_MC02S02
│   ├── wav_arrayMic_MC03S01
│   ├── wav_arrayMic_MC03S02
│   ├── wav_arrayMic_MC04S01
│   ├── wav_arrayMic_MC04S02
│   ├── wav_headMic_MC01S01
│   ├── wav_headMic_MC01S02
│   ├── wav_headMic_MC01S03
│   ├── wav_headMic_MC02S01
│   ├── wav_headMic_MC02S02
│   ├── wav_headMic_MC03S01
│   ├── wav_headMic_MC03S02
│   └── wav_headMic_MC04S01
└── M_Dys
    ├── wav_arrayMic_M01S01
    ├── wav_arrayMic_M01S02
    ├── wav_arrayMic_M02S01
    ├── wav_arrayMic_M02S02
    ├── wav_arrayMic_M03S02
    ├── wav_arrayMic_M04S01
    ├── wav_arrayMic_M04S02
    ├── wav_arrayMic_M05S01
    ├── wav_headMic_M01S01
    ├── wav_headMic_M01S02
    ├── wav_headMic_M02S01
    ├── wav_headMic_M02S02
    ├── wav_headMic_M03S02
    ├── wav_headMic_M04S02
    ├── wav_headMic_M05S01
    └── wav_headMic_M05S02
```
