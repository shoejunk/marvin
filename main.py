import threading
import asyncio
from speech import transcribe_speech_to_text
from tts import speak_text
from llm import get_ai_response, load_local_model
from waiting_sound import play_waiting_sound
from meross_control import meross_toggle

def main():
    load_local_model()
    print("Say 'quit' or 'exit' to stop.")
    
    while True:
        user_input = transcribe_speech_to_text()
        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Exiting the voice chat. Goodbye!")
            break

        # Process commands only if the wake word "marvin" is detected.
        if not user_input.lower().startswith("marvin"):
            print("Waiting for wake word 'marvin'...")
            continue

        # Remove the wake word.
        command = user_input[len("marvin"):].strip()

        # If the command is to control a Meross outlet, handle it here.
        asyncio.run(meross_toggle())

        # if command.lower().startswith("turn on outlet"):
        #     control_meross_outlet("on")
        #     continue
        # elif command.lower().startswith("turn off outlet"):
        #     control_meross_outlet("off")
        #     continue

        # Otherwise, let the AI generate a reply.
        stop_event = threading.Event()
        waiting_thread = threading.Thread(target=play_waiting_sound, args=(stop_event,))
        waiting_thread.start()
        reply = get_ai_response(user_input)
        stop_event.set()
        waiting_thread.join()

        print(f"Marvin says: {reply}")
        if reply:
            speak_text(reply)

if __name__ == "__main__":
    main()
