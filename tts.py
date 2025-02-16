import os
import asyncio
import edge_tts
import playsound

async def async_speak_text(text: str, voice="en-GB-RyanNeural"):
    tts_file = "temp_tts.mp3"
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(tts_file)
    playsound.playsound(tts_file, True)
    os.remove(tts_file)

def speak_text(text: str, voice="en-GB-RyanNeural"):
    asyncio.run(async_speak_text(text, voice=voice))
