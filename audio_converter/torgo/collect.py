import os

ROOT = "../../data/torgo-audio"
wav_files = []

for root, _, files in os.walk(ROOT):
    for f in files:
        if f.lower().endswith(".wav"):
            wav_files.append(os.path.join(root, f))

# print(f"Found {len(wav_files)} wav files")


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


print(parse_metadata(
    '../../data/torgo-audio/F_Con/wav_arrayMic_FC01S01/wav_arrayMic_FC01S01_0003.wav'))
