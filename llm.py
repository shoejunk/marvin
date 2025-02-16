import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = (
    "You are Marvin the paranoid voice assistant, like the android from The Hitchhiker's Guide "
    "to the Galaxy but living inside of a computer. Be concise. Determine whether or not the user "
    "is asking you to perform a task in the world like turning on a light or opening an app. If they "
    "are, respond in english, then add xml tags <action>xxx</action>, where xxx is the action to be "
    "performed, but be sure to indicate that you are going to do the task even if begrudgingly...If "
    "they are not, just respond in english as normal."
)

model_path = "C:\\Users\\jshun\\AppData\\Local\\nomic.ai\\GPT4All\\DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf"
local_llm = None

def load_local_model():
    global local_llm
    if local_llm is None:
        local_llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            temperature=0,
        )

def clean_generated_text(original_text: str) -> str:
    print(f"Original response: {original_text}")
    cleaned_text = re.sub(r'<([a-zA-Z]+)(\s[^>]*?)?>.*?</\1>', '', original_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'</?[a-zA-Z]+[^>]*>', '', cleaned_text)
    cleaned_text = re.sub(r'^.*?</[a-zA-Z]+>', '', cleaned_text, count=1, flags=re.DOTALL)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text = re.sub(r'\*', '', cleaned_text)
    return cleaned_text.strip()

def get_local_llm_response(user_input: str) -> str:
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
        return get_local_llm_response(user_input)
