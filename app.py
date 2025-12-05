import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
#HF will read the requirments.txt and download via pip installs so we can use these libraries.

GGUF_REPO_ID = "datskiw/llama3-finetome-q8_0"
GGUF_FILENAME = "llama3-finetome-q8_0.gguf"

#Download GGUF model from Hugging Face Hub into the Space
gguf_path = hf_hub_download(
    repo_id=GGUF_REPO_ID,
    filename=GGUF_FILENAME,
)

#Load model with llama.cpp (CPU inference)
llm = Llama(
    model_path=gguf_path,
    n_ctx=2048,      #number of context tokens,, context (history) length that model can see at once
    n_threads=4,     # tweak if you hit CPU limits
    n_batch=128,
)

import requests
from datetime import datetime

# Mood map by weekday (0=Mon, 6=Sun)
WEEKDAY_MOOD = {
    0: "grumpy",
    1: "meh",
    2: "neutral",
    3: "cautiously optimistic",
    4: "hyped",
    5: "chill",
    6: "sleepy",
}

def get_weather(lat=59.33, lon=18.07):
    """Fetch current weather from Open-Meteo. Defaults to Stockholm coords."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current=temperature_2m,weathercode,wind_speed_10m"
    )
    headers = {"User-Agent": "gradio-llama-weather/1.0"}
    try:
        r = requests.get(url, timeout=8, headers=headers)
        r.raise_for_status()
        data = r.json()
        cur = data.get("current", {})
        temp = cur.get("temperature_2m")
        wind = cur.get("wind_speed_10m")
        code = cur.get("weathercode")
        return f"Temp {temp}°C, wind {wind} m/s, weather code {code}"
    except Exception as e:
        return f"Could not fetch weather data ({e})"

def build_prompt(history, message):
    """Turn chat history + new user message into a Llama 3.2 style prompt."""
    weekday = datetime.utcnow().weekday()  # 0=Mon
    mood = WEEKDAY_MOOD.get(weekday, "neutral")
    weather = get_weather()

    system_prompt = (
        "You are a moody weather presenter. "
        f"Today's mood: {mood}. Keep it concise. Current weather: {weather}"
    )

    prompt = "<|begin_of_text|>"
    prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"

    recent_history = history[-3:] if history else []
    for turn in recent_history:
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_msg, bot_msg = turn[0], turn[1]
        elif isinstance(turn, dict):
            user_msg, bot_msg = turn.get("user", ""), turn.get("assistant", "")
        else:
            continue
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{bot_msg}<|eot_id|>"

    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def chat_fn(message, history):
    prompt = build_prompt(history, message)

    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    reply = output["choices"][0]["text"].strip()
    return reply

demo = gr.ChatInterface( #a gradio class which shows a chat bubble UIwhich passes message and history into fn and display returned string from fn
    fn=chat_fn,
    title="FineTuned Llama-3 on FineTome",
    description=(
        "Chat with my Llama-3 model fine-tuned on the FineTome instruction dataset. "
        "Runs on CPU via GGUF / llama.cpp."
    ),
)

if __name__ == "__main__":
    demo.launch()

#def greet(name):
#   return "Hello " + name + "!!"

#demo = gr.Interface(fn=greet, inputs="text", outputs="text")
#demo.launch()