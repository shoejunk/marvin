#!/usr/bin/env python3
"""
llm.py - Handles local and API-based language model responses.
This module preserves <action> tags in the response for downstream processing.
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from llama_cpp import Llama

# Load environment variables from a .env file (if present)
load_dotenv()

# Initialize OpenAI client using the API key from the environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt for the voice assistant
system_prompt = (
    "You are Marvin the paranoid voice assistant, like the android from The Hitchhiker's Guide "
    "to the Galaxy but living inside of a computer. Be concise. Determine whether or not the user "
    "is asking you to perform a task in the world like turning on a light or opening an app. If they "
    "are, respond in English, then add xml tags <action>xxx</action>, where xxx is the action to be "
    "performed, but be sure to indicate that you are going to do the task even if begrudgingly..."
    "The valid actions are: turn_on_light, turn_off_light."
    "If they are not, just respond in English as normal."
)

# Path to the local LLM model
model_path = "C:\\Users\\jshun\\AppData\\Local\\nomic.ai\\GPT4All\\DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf"
local_llm = None

def load_local_model():
    """Loads the local language model if it hasn't been loaded already."""
    global local_llm
    if local_llm is None:
        local_llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            temperature=0,
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

def get_local_llm_response(user_input: str) -> str:
    """
    Gets a response from the local language model.
    Loads the model if necessary and cleans the output.
    """
    try:
        if local_llm is None:
            load_local_model()
        output = local_llm(
            prompt=user_input,
            max_tokens=256,
            stop=["User:", "System:", "Assistant:"],
        )
        return clean_generated_text(output["choices"][0]["text"])
    except Exception as e:
        print(f"Error using local LLM: {e}")
        return "I'm sorry, I'm having an issue with the local LLM."

def get_ai_response(user_input: str) -> str:
    """
    Gets a response from the OpenAI API.
    Falls back to the local language model in case of errors.
    """
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
        return
