import os
import asyncio
from openai import OpenAI
import speech_recognition as sr
from dotenv import load_dotenv

import edge_tts
import playsound
from llama_cpp import Llama
import re

useLocalModel = True

# 1. Load your OpenAI API key from the .env file
load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

#######################
#   TEXT-TO-SPEECH    #
#######################
# We'll create an async function for edge-tts
# and a synchronous wrapper to call it easily.
#async def async_speak_text(text: str, voice="en-US-AriaNeural"):
async def async_speak_text(text: str, voice="en-GB-RyanNeural"):
    """Asynchronously convert text to speech using edge-tts and play the audio."""
    tts_file = "temp_tts.mp3"
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(tts_file)

    # Play the generated MP3
    playsound.playsound(tts_file, True)

    # Remove the temporary file
    os.remove(tts_file)

def speak_text(text: str, voice="en-GB-RyanNeural"):
    """Synchronous wrapper around the async edge-tts TTS function."""
    asyncio.run(async_speak_text(text, voice=voice))

#######################
#  SPEECH RECOGNITION #
#######################
def transcribe_speech_to_text() -> str:
    """Listen to the microphone and return speech as text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        recognizer.energy_threshold = 2000  # Adjust based on your environment
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Error with the speech recognition service: {e}")
        return ""

######################
#   LOCAL LLM (GGUF) #
######################
# For efficiency, load the llama model once at startup rather than on every fallback call.
# Adjust model_path to match the path to your .gguf file.
# model_path = "C:\\Users\\jshun\\AppData\\Local\\nomic.ai\\GPT4All\\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
model_path = "C:\\Users\\jshun\\AppData\\Local\\nomic.ai\\GPT4All\\DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf"
local_llm = None

def load_local_model():
    global local_llm
    if local_llm is None:
        local_llm = Llama(
            model_path=model_path,
            n_ctx=2048,       # Context window, adjustable
            n_threads=6,      # Adjust based on your CPU
            temperature=0,  # Adjust generation parameters as needed
        )

def get_local_llm_response(user_input: str) -> str:
    """
    Generate a response from a local llama-cpp-based model with .gguf format.
    """
    try:
        if local_llm is None:
            load_local_model()

        # For llama-cpp-python, you provide the prompt and get back a dictionary response
        output = local_llm(
            prompt=user_input,
            max_tokens=256,
            stop=["User:", "System:", "Assistant:"],  # optional stop tokens
        )
        # The text is typically found under output["choices"][0]["text"]
        generated_text = output["choices"][0]["text"]
        pattern = r'<think>(.*?)</think>'
        # Use DOTALL so that '.' matches newlines
        cleaned_text = re.sub(pattern, '', generated_text, flags=re.DOTALL)
        
        # Optional: remove extra spaces introduced by the removal
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        pattern = r'^.*?</think>'  # match everything from the start up to </think>
        cleaned_text = re.sub(pattern, '', cleaned_text, 1, flags=re.DOTALL)

        return cleaned_text.strip()

    except Exception as e:
        print(f"Error using local LLM: {e}")
        return "I'm sorry, I'm having an issue with the local LLM."

#######################
#      CHAT API       #
#######################
def get_ai_response(user_input: str) -> str:
    if useLocalModel:
        return get_local_llm_response(user_input)
    else:
        """Send user input to the ChatGPT API and get a response."""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if you have access
                messages=[{"role": "user", "content": user_input}],
                temperature=0.7
            )
            # Extract the assistant's reply
            assistant_reply = response.choices[0].message.content
            return assistant_reply.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "I'm sorry, but I couldn't process that."

########################
#       MAIN LOOP      #
########################
def main():
    load_local_model()
    print("Say 'quit' or 'exit' to stop.")
    print("Listening...")
    while True:
        # 1. Get user speech and convert to text
        user_input = transcribe_speech_to_text()
        if not user_input:
            continue
        
        # 2. Check if user wants to exit
        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Exiting the voice chat. Goodbye!")
            break
        
        # 3. Get AI response
        reply = get_ai_response(user_input)
        print(f"Marvin says: {reply}")
        
        # 4. Speak out ChatGPT's response
        speak_text(reply)

if __name__ == "__main__":
    main()
