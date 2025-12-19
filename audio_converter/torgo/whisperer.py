import whisper

model = whisper.load_model("base", device='cpu')

wav = "../../data/torgo-audio/F_Con/wav_arrayMic_FC01S01/wav_arrayMic_FC01S01_0197.wav"

result = model.transcribe(wav)

print("Language:", result["language"])
print("Text:", result["text"])
