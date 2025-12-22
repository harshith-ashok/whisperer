import csv

with open("torgo_whisper.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file", "group", "speaker", "session",
            "utterance", "language", "text"
        ]
    )
    writer.writeheader()

    for wav in wav_files[:10]:
        try:
            group, speaker, session, utt = parse_metadata(wav)
            result = model.transcribe(wav)

            writer.writerow({
                "file": wav,
                "group": group,
                "speaker": speaker,
                "session": session,
                "utterance": utt,
                "language": result["language"],
                "text": result["text"]
            })

        except Exception as e:
            print("Failed:", wav, e)
