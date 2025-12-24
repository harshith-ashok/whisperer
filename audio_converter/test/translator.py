import asyncio
from googletrans import Translator

str = ''
with open("output.txt", "r") as f:
    str = f.readline()


async def translate_text():
    async with Translator() as translator:
        result = await translator.translate(str, dest='en')
    print(result.text)

asyncio.run(translate_text())
