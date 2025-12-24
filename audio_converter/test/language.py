import whisper

model = whisper.load_model("turbo")

audio = whisper.load_audio("output.wav")
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(
    audio, n_mels=model.dims.n_mels).to(model.device)

_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

print(result.text)

with open("output.txt", "w") as f:
    f.write(result.text)
