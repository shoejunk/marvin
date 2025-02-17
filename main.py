#!/usr/bin/env python3
"""
main.py - Main entry point for the voice assistant.
Processes voice input, obtains AI responses, strips out <action> tags before speaking,
and triggers actions (e.g., turning lights on/off) via the MerossController.
"""

import asyncio
import re
import threading
from speech import transcribe_speech_to_text
from tts import speak_text
from llm import get_ai_response, load_local_model
from waiting_sound import play_waiting_sound
from meross_control import MerossController

# Define valid action strings (all lowercase, with underscores)
action_strings = ['turn_on_light', 'turn_off_light']

async def async_main():
    # load_local_model()
    print("Initializing Meross Controller...")
    meross_controller = await MerossController.init()
    
    print("Say 'quit' or 'exit' to stop.")
    
    while True:
        # Call the blocking transcribe function in a thread
        user_input = await asyncio.to_thread(transcribe_speech_to_text)
        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Exiting the voice chat. Goodbye!")
            await meross_controller.shutdown()
            break

        # Process commands only if the wake word "marvin" is detected.
        if not user_input.lower().startswith("marvin"):
            print("Waiting for wake word 'marvin'...")
            continue

        # Remove the wake word.
        command = user_input[len("marvin"):].strip()

        # Start waiting sound in a separate thread.
        stop_event = threading.Event()
        waiting_thread = threading.Thread(target=play_waiting_sound, args=(stop_event,))
        waiting_thread.start()

        # Get AI response using a thread since it may block.
        reply = await asyncio.to_thread(get_ai_response, user_input)
        stop_event.set()
        waiting_thread.join()


        # Remove any <action> tags from the text before speaking.
        text_to_speak = re.sub(r'<action>.*?</action>', '', reply, flags=re.IGNORECASE)
        text_to_speak = re.sub(r'<[^>]+>', '', text_to_speak).strip()

        if text_to_speak:
            print(f"Marvin says: {text_to_speak}")
            await asyncio.to_thread(speak_text, text_to_speak)

        # Parse the AI reply for <action> tags to trigger actions.
        action_tags = re.findall(r'<action>(.*?)</action>', reply, flags=re.IGNORECASE)
        for action in action_tags:
            normalized_action = action.lower().replace(" ", "_")
            if normalized_action in action_strings:
                print(f"Detected action: {normalized_action}")
                if normalized_action == "turn_on_light":
                    await meross_controller.turn_on_light()
                elif normalized_action == "turn_off_light":
                    await meross_controller.turn_off_light()
            else:
                print(f"Action '{normalized_action}' not recognized in the action list.")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
