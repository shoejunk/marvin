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

system_prompt = "You are Marvin the paranoid voice assistant, like the android from The Hitchhiker's Guide to the Galaxy but living inside of a computer. Be concise. Determine whether or not the user is asking you to perform a task in the world like turning on a light or opening an app. If they are, respond in english, then add xml tags <action>xxx</action>, where xxx is the action to be performed, but be sure to indicate that you are going to do the task even if begrudgingly...If they are not, just respond in english as normal.";

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

def clean_generated_text(original_text: str) -> str:
    """
    Extracts and cleans the generated text.
    
    The cleaning process includes:
      - Removing any content within XML tags (including the tags themselves).
      - Removing any stray XML tags (e.g. self-closing tags).
      - Removing any text from the start up to a closing XML tag if it still exists.
      - Collapsing multiple whitespace characters into a single space.
      - Removing all asterisks (*) from the text.
    
    Args:
        original_text (str): A string containing the original text.

    Returns:
        str: The cleaned text.
    """

    print(f"Original response: {original_text}")

    # Remove any XML blocks that have a matching closing tag.
    # This regex matches tags made up of letters (optionally with attributes) and removes everything from the opening tag to the corresponding closing tag.
    cleaned_text = re.sub(r'<([a-zA-Z]+)(\s[^>]*?)?>.*?</\1>', '', original_text, flags=re.DOTALL)
    
    # Remove any remaining stray XML tags (opening, closing, or self-closing).
    cleaned_text = re.sub(r'</?[a-zA-Z]+[^>]*>', '', cleaned_text)
    
    # In case there is still extraneous text before a closing XML tag, remove text up to the first closing tag.
    cleaned_text = re.sub(r'^.*?</[a-zA-Z]+>', '', cleaned_text, count=1, flags=re.DOTALL)
    
    # Collapse multiple whitespace characters into a single space and trim
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
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
        return clean_generated_text(output["choices"][0]["text"])

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
        return clean_generated_text(assistant_reply)
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

       # Process input only if it starts with the wake word "marvin"
        if not user_input.lower().startswith("marvin"):
            print("Waiting for wake word 'marvin'...")
            continue

        # Strip the wake word from the input
        prompt = user_input[len("marvin"):].strip()
        if not prompt:
            print("No command detected after the wake word.")
            continue

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
        if (reply != ""):
            speak_text(reply)

if __name__ == "__main__":
    main()
