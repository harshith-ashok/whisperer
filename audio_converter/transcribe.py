import whisper

model = whisper.load_model("base")
result = model.transcribe("output.wav")

print(result["text"])

with open("output.txt", "w") as f:
    f.write(result["text"])
