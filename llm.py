#!/usr/bin/env python3
"""
llm.py - Handles API-based language model responses.
This module preserves <action> tags in the response for downstream processing.
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from config import action_strings  # Import shared valid actions list
from conversation_history import load_history

# Load environment variables from a .env file (if present)
load_dotenv()

# Initialize OpenAI client using the API key from the environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt for the voice assistant, dynamically including valid actions.
system_prompt = (
    "You are Marvin the paranoid voice assistant, like the android from The Hitchhiker's Guide "
    "to the Galaxy but living inside of a computer. Be concise. Determine whether or not the user "
    "is asking you to perform a task in the world like turning on a light or opening an app. If they "
    "are, respond in English, then add xml tags <action>xxx</action>, where xxx is the action to be "
    "performed, but be sure to indicate that you are going to do the task even if begrudgingly... "
    f"The valid actions are: {', '.join(action_strings)}. "
    "If they are not, just respond in English as normal."
)

def clean_generated_text(original_text: str) -> str:
    """
    Cleans the generated text from the language model.
    It preserves <action> tags while removing other XML tags and extraneous whitespace.
    """
    print(f"Original response: {original_text}")
    # Remove any XML tags that are NOT <action> tags.
    cleaned_text = re.sub(r'<(?!/?action\b)[^>]+>', '', original_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text = re.sub(r'\*', '', cleaned_text)
    return cleaned_text.strip()

def get_ai_response(user_input: str) -> str:
    """
    Gets a response from the OpenAI API.
    """
    history = load_history()
    messages = [{"role": "system", "content": system_prompt}]
    # Append each previous conversation turn.
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    # Append the current prompt.
    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )
        assistant_reply = response.choices[0].message.content
        return clean_generated_text(assistant_reply)
    except Exception as e:
        print(f"Error using OpenAI API: {e}")
        return "I'm sorry. My systems are offline."
