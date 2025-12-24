import os
import csv
from collections import defaultdict
import mlx_whisper

ROOT = "../../data/torgo-audio"
OUTPUT_CSV = "torgo_whisper_mlx.csv"
FILES_PER_SESSION = 5
MODEL_NAME = "mlx-community/whisper-turbo"


def parse_metadata(path):
    parts = path.split(os.sep)

    if len(parts) < 3:
        raise ValueError(f"Invalid TORGO path: {path}")

    group = parts[-3]       # F_Con / F_Dys / M_Con / M_Dys
    folder = parts[-2]      # wav_arrayMic_FC01S01
    filename = parts[-1]    # wav_arrayMic_FC01S01_0003.wav

    folder_parts = folder.split("_")
    if len(folder_parts) < 3:
        raise ValueError(f"Unexpected folder format: {folder}")

    speaker_session = folder_parts[2]   # FC01S01
    speaker = speaker_session[:4]       # FC01
    session = speaker_session[4:]       # S01

    utterance = filename.split("_")[-1].replace(".wav", "")

    return group, speaker, session, utterance


# --------------------------------------------------
# Collect WAV files grouped by session
# --------------------------------------------------
sessions = defaultdict(list)

for root, _, files in os.walk(ROOT):
    for f in files:
        if f.lower().endswith(".wav"):
            full_path = os.path.join(root, f)
            try:
                group, speaker, session, utt = parse_metadata(full_path)
                key = (group, speaker, session)
                sessions[key].append(full_path)
            except Exception as e:
                print("Skipping:", full_path, e)

print(f"Found {len(sessions)} sessions")


# --------------------------------------------------
# Transcribe (5 files per session)
# --------------------------------------------------
rows = []

for (group, speaker, session), wavs in sessions.items():
    wavs = sorted(wavs)[:FILES_PER_SESSION]

    for wav in wavs:
        try:
            result = mlx_whisper.transcribe(
                wav,
                path_or_hf_repo=MODEL_NAME
            )

            rows.append({
                "file": wav,
                "group": group,
                "speaker": speaker,
                "session": session,
                "utterance": os.path.basename(wav).split("_")[-1].replace(".wav", ""),
                "language": result.get("language", "en"),
                "text": result["text"].strip()
            })

            print(f"[{speaker}-{session}] {result['text']}")

        except Exception as e:
            print("Failed:", wav, e)


# --------------------------------------------------
# Write CSV
# --------------------------------------------------
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
    writer.writerows(rows)

print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")
