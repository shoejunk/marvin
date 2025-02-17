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
from llm import get_ai_response
from waiting_sound import play_waiting_sound
from meross_control import MerossController
from config import action_strings  # Import shared valid actions list
from conversation_history import update_history

async def async_main():
    print("Initializing Meross Controller...")
    meross_controller = await MerossController.init()
    
    while True:
        # Get user input from speech transcription.
        user_input = await asyncio.to_thread(transcribe_speech_to_text)
        if not user_input:
            continue

        # Process commands only if a valid wake word ("marvin", "hey marvin", or "ok marvin") is detected.
        wake_words = ["marvin", "hey marvin", "ok marvin", "okay marvin", "hi marvin"]
        wake_words += ["martin", "hey martin", "ok martin", "okay martin", "hi martin"]
        user_input_lower = user_input.lower()
        matched_wake_word = None
        for wake_word in wake_words:
            if user_input_lower.startswith(wake_word):
                matched_wake_word = wake_word
                break

        if not matched_wake_word:
            print("Waiting for wake word 'marvin'...")
            continue

        # Remove the detected wake word from the beginning of the input.
        command = user_input[len(matched_wake_word):].strip()

        # Start waiting sound in a separate thread.
        stop_event = threading.Event()
        waiting_thread = threading.Thread(target=play_waiting_sound, args=(stop_event,))
        waiting_thread.start()

        # Get AI response using a thread since it may block.
        reply = await asyncio.to_thread(get_ai_response, user_input)
        stop_event.set()
        waiting_thread.join()

        # Update conversation history with the current turn.
        update_history(user_input, reply)

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
