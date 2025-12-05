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
import time
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

# Weather code descriptions (Open-Meteo / WMO)
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

def get_weather(lat=59.33, lon=18.07):
    """Fetch current weather from Open-Meteo. Defaults to Stockholm coords."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current=temperature_2m,weathercode,wind_speed_10m"
    )
    headers = {"User-Agent": "gradio-llama-weather/1.0"}
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=8, headers=headers)
            if r.status_code == 429:
                # backoff and retry on rate limit
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
            data = r.json()
            cur = data.get("current", {})
            temp = cur.get("temperature_2m")
            # API returns km/h; convert to m/s for clarity
            wind_kmh = cur.get("wind_speed_10m")
            wind_ms = None
            if wind_kmh is not None:
                try:
                    wind_ms = round(float(wind_kmh) / 3.6, 1)
                except Exception:
                    wind_ms = wind_kmh
            code = cur.get("weathercode")
            code_desc = WEATHER_CODES.get(code, "unknown")
            return f"Temp {temp}°C, wind {wind_ms} m/s, {code_desc} (code {code})"
        except Exception as e:
            last_err = e
            print(f"[weather] attempt {attempt+1} failed: {e}")
            time.sleep(0.5 * (attempt + 1))
    return f"Could not fetch weather data ({last_err})"

def build_prompt(history, message, weather):
    """Turn chat history + new user message into a Llama 3.2 style prompt."""
    weekday = datetime.utcnow().weekday()  # 0=Mon
    mood = WEEKDAY_MOOD.get(weekday, "neutral")

    system_prompt = (
        "You are a moody weather presenter. "
        f"Today's mood: {mood}. Be concise, factual, and avoid making up forecasts or highs/lows.\n"
        "Always include a single line starting with 'Weather:' that echoes the provided data "
        "with temperature, wind (m/s), and description. Do not invent values.\n"
        "If weather data is unavailable, say 'Weather: unavailable <error>'.\n"
        f"Current weather data (must reuse, no changes): {weather}"
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
    weather = get_weather()
    prompt = build_prompt(history, message, weather)

    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.1,
        top_p=0.8,
        repeat_penalty=1.1,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    reply = output["choices"][0]["text"].strip()
    # Prepend deterministic weather line to prevent the model from omitting it
    return f"Weather: {weather}\n{reply}"

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