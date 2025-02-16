import os
import asyncio
import threading
import time
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
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

system_prompt = "You are Mavin the paranoid voice assistant, like the android from The Hitchhiker's Guide to the Galaxy but living inside of a computer. Be concise. When you are asked to perform a task, respond in english, then add xml tags <action>xxx</action>, where xxx is the action to be performed.";

#######################
#   TEXT-TO-SPEECH    #
#######################
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
        #recognizer.energy_threshold = 2000  # Adjust based on your environment
        print("Listening...")
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

#######################
#      WAITING SOUND  #
#######################
def play_waiting_sound(stop_event):
    """
    Plays a short waiting sound in a loop until stop_event is set.
    Make sure you have a small sound file (e.g., "waiting_sound.mp3") in the same directory.
    """
    while not stop_event.is_set():
        # Play the waiting sound. Using block=True ensures the sound finishes before looping.
        playsound.playsound("waiting_sound.mp3", True)
        # Optional: add a brief pause if needed
        time.sleep(0.1)

#######################
#   LOCAL LLM (GGUF)  #
#######################
model_path = "C:\\Users\\jshun\\AppData\\Local\\nomic.ai\\GPT4All\\DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf"
#model_path = "C:\\Users\\jshun\\AppData\\Local\\nomic.ai\\GPT4All\\Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf"
local_llm = None

def load_local_model():
    global local_llm
    if local_llm is None:
        local_llm = Llama(
            model_path=model_path,
            n_ctx=2048,       # Context window, adjustable
            n_threads=6,      # Adjust based on your CPU
            temperature=0,    # Adjust generation parameters as needed
        )

def clean_generated_text(output: dict) -> str:
    """
    Extracts and cleans the generated text from the output dictionary.
    
    The cleaning process includes:
      - Removing any content within <think>...</think> tags.
      - Removing any text from the start up to the closing </think> tag.
      - Collapsing multiple whitespace characters into a single space.
      - Removing all asterisks (*) from the text.
    
    Args:
        output (dict): A dictionary with a key "choices", where the first element is a dict
                       containing the key "text".
    
    Returns:
        str: The cleaned text.
    """
    # Extract the generated text
    generated_text = output.get("choices", [{}])[0].get("text", "")

    # Remove any content within <think>...</think> tags
    cleaned_text = re.sub(r'<think>(.*?)</think>', '', generated_text, flags=re.DOTALL)
    
    # Collapse multiple whitespace characters into a single space and trim
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Remove any text from the start up to the closing </think> tag (if it still exists)
    cleaned_text = re.sub(r'^.*?</think>', '', cleaned_text, count=1, flags=re.DOTALL)
    
    # Remove all asterisks
    cleaned_text = re.sub(r'\*', '', cleaned_text)
    
    return cleaned_text.strip()

def get_local_llm_response(user_input: str) -> str:
    """
    Generate a response from a local llama-cpp-based model with .gguf format.
    """
    try:
        if local_llm is None:
            load_local_model()

        full_prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"

        output = local_llm(
            prompt=user_input,
            max_tokens=256,
            stop=["User:", "System:", "Assistant:"],  # optional stop tokens
        )
        return clean_generated_text(output)
        # generated_text = output["choices"][0]["text"]
        # pattern = r'<think>(.*?)</think>'
        # cleaned_text = re.sub(pattern, '', generated_text, flags=re.DOTALL)
        # cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        # pattern = r'^.*?</think>'  # match everything from the start up to </think>
        # cleaned_text = re.sub(pattern, '', cleaned_text, 1, flags=re.DOTALL)

        # return cleaned_text.strip()

    except Exception as e:
        print(f"Error using local LLM: {e}")
        return "I'm sorry, I'm having an issue with the local LLM."

#######################
#      CHAT API       #
#######################
def get_ai_response(user_input: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.7
        )
        assistant_reply = response.choices[0].message.content
        return assistant_reply.strip()
    except Exception as e:
        # Fall back to local model in case of error
        return get_local_llm_response(user_input)

########################
#       MAIN LOOP      #
########################
def main():
    load_local_model()
    print("Say 'quit' or 'exit' to stop.")
    while True:
        # 1. Get user speech and convert to text
        user_input = transcribe_speech_to_text()
        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Exiting the voice chat. Goodbye!")
            break

        # 2. Start the waiting sound in a background thread
        stop_event = threading.Event()
        waiting_thread = threading.Thread(target=play_waiting_sound, args=(stop_event,))
        waiting_thread.start()

        # 3. Get AI response (this is a blocking call)
        reply = get_ai_response(user_input)

        # 4. Stop the waiting sound once the response is ready
        stop_event.set()
        waiting_thread.join()

        print(f"Marvin says: {reply}")

        # 5. Speak out the response
        speak_text(reply)

if __name__ == "__main__":
    main()
