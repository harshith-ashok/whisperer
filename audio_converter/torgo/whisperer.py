import os
import csv
import whisper
from collections import defaultdict
import random


ROOT = "../../data/torgo-audio"
OUTPUT_CSV = "torgo_whisper.csv"
MODEL_NAME = "base"
DEVICE = "cpu"
FILES_PER_SESSION = 5


def collect_wav_files(root):
    wavs = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(r, f))
    return wavs


def parse_metadata(path):
    parts = path.split(os.sep)

    if len(parts) < 3:
        raise ValueError(f"Invalid TORGO path: {path}")

    group = parts[-3]
    folder = parts[-2]
    filename = parts[-1]

    folder_parts = folder.split("_")
    if len(folder_parts) < 3:
        raise ValueError(f"Unexpected folder format: {folder}")

    speaker_session = folder_parts[2]   # FC01S01
    speaker = speaker_session[:-3]      # FC01 / F04 / MC02
    session = speaker_session[-3:]      # S01 / S02

    utterance = filename.split("_")[-1].replace(".wav", "")

    return group, speaker, session, utterance


def main():
    print("Collecting WAV files...")
    wav_files = collect_wav_files(ROOT)
    print(f"Found {len(wav_files)} wav files")

    if not wav_files:
        raise RuntimeError("No WAV files found")

    print("Grouping files by session...")
    session_map = defaultdict(list)

    for wav in wav_files:
        try:
            _, speaker, session, _ = parse_metadata(wav)
            session_key = f"{speaker}_{session}"
            session_map[session_key].append(wav)
        except Exception:
            continue

    print(f"Found {len(session_map)} sessions")

    print(f"Selecting {FILES_PER_SESSION} files per session...")
    selected_wavs = []

    for session_key in sorted(session_map.keys()):
        files = session_map[session_key]
        random.shuffle(files)
        selected_wavs.extend(files[:FILES_PER_SESSION])

    print(f"Total selected files: {len(selected_wavs)}")

    print("Loading Whisper model...")
    model = whisper.load_model(MODEL_NAME, device=DEVICE)

    print("Starting transcription...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "group",
                "speaker",
                "session",
                "utterance",
                "language",
                "text"
            ]
        )
        writer.writeheader()

        for i, wav in enumerate(selected_wavs, start=1):
            try:
                group, speaker, session, utt = parse_metadata(wav)
                result = model.transcribe(wav, fp16=False)

                writer.writerow({
                    "file": wav,
                    "group": group,
                    "speaker": speaker,
                    "session": session,
                    "utterance": utt,
                    "language": result["language"],
                    "text": result["text"]
                })

                print(f"[{i}/{len(selected_wavs)}] ✓ {speaker}-{session}-{utt}")

            except Exception as e:
                print(f"[{i}] ✗ Failed: {wav}")
                print("   ", e)

    print("Done. Results saved to", OUTPUT_CSV)


if __name__ == "__main__":
    main()
