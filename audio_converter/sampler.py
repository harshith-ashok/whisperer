from gtts import gTTS
import os

text = "வணக்கம்"

language = 'ta'
speech = gTTS(text=text, lang=language, slow=False)


filename = "output.wav"
speech.save(filename)

print(f"Audio saved to {filename}")
