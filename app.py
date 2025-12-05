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

def geocode_location(location: str, fallback_lat=59.33, fallback_lon=18.07):
    """Geocode a city name to lat/lon using Open-Meteo. Fallback to Stockholm on failure."""
    if not location:
        return fallback_lat, fallback_lon, "Stockholm"
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(location)}&count=1&language=en&format=json"
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "gradio-llama-weather/1.0"})
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if not results:
            return fallback_lat, fallback_lon, "Stockholm"
        hit = results[0]
        lat = hit.get("latitude", fallback_lat)
        lon = hit.get("longitude", fallback_lon)
        name = hit.get("name") or location
        country = hit.get("country") or ""
        display = f"{name}, {country}".strip().strip(",")
        return lat, lon, display
    except Exception as e:
        print(f"[geocode] failed for '{location}': {e}")
        return fallback_lat, fallback_lon, "Stockholm"


def get_weather(lat=59.33, lon=18.07, forecast_days=0):
    """Fetch current or forecast weather from Open-Meteo. forecast_days=0 for current, 1+ for forecast."""
    if forecast_days == 0:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,weathercode,wind_speed_10m,precipitation,precipitation_probability"
        )
    else:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&forecast_days={min(forecast_days, 7)}"
            "&daily=temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
        )
    headers = {"User-Agent": "gradio-llama-weather/1.0"}
    last_err = "unknown"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=8, headers=headers)
            if r.status_code == 429:
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
            data = r.json()
            
            if forecast_days == 0:
                # Current weather
                cur = data.get("current", {})
                temp = cur.get("temperature_2m")
                wind_kmh = cur.get("wind_speed_10m")
                wind_ms = round(float(wind_kmh) / 3.6, 1) if wind_kmh is not None else None
                code = cur.get("weathercode")
                code_desc = WEATHER_CODES.get(code, "unknown")
                precip = cur.get("precipitation")
                precip_prob = cur.get("precipitation_probability")
                return {
                    "temp": temp,
                    "wind_ms": wind_ms,
                    "description": code_desc,
                    "precip": precip,
                    "precip_prob": precip_prob,
                    "type": "current"
                }
            else:
                # Forecast weather
                daily = data.get("daily", {})
                temps_max = daily.get("temperature_2m_max", [])
                temps_min = daily.get("temperature_2m_min", [])
                codes = daily.get("weathercode", [])
                precip_sum = daily.get("precipitation_sum", [])
                precip_prob_max = daily.get("precipitation_probability_max", [])
                wind_max = daily.get("wind_speed_10m_max", [])
                day_idx = min(forecast_days - 1, len(temps_max) - 1)
                wind_ms = round(float(wind_max[day_idx]) / 3.6, 1) if wind_max and wind_max[day_idx] is not None else None
                return {
                    "temp_max": temps_max[day_idx] if temps_max else None,
                    "temp_min": temps_min[day_idx] if temps_min else None,
                    "wind_ms": wind_ms,
                    "description": WEATHER_CODES.get(codes[day_idx], "unknown") if codes else "unknown",
                    "precip": precip_sum[day_idx] if precip_sum else None,
                    "precip_prob": precip_prob_max[day_idx] if precip_prob_max else None,
                    "type": "forecast",
                    "day": forecast_days
                }
        except Exception as e:
            last_err = e
            print(f"[weather] attempt {attempt+1} failed: {e}")
            time.sleep(0.5 * (attempt + 1))
    return {"error": str(last_err)}

def build_prompt(history, message, weather_data, location_name, mood):
    """Turn chat history + new user message into a Llama 3.2 style prompt."""
    if weather_data.get("error"):
        weather_str = f"Weather data unavailable: {weather_data['error']}"
    elif weather_data.get("type") == "forecast":
        w = weather_data
        weather_str = (
            f"Forecast for {location_name}: "
            f"High {w.get('temp_max')}°C, Low {w.get('temp_min')}°C, "
            f"wind {w.get('wind_ms')} m/s, {w.get('description')}, "
            f"precip chance {w.get('precip_prob')}%, precip {w.get('precip')} mm"
        )
    else:
        w = weather_data
        weather_str = (
            f"Current weather in {location_name}: "
            f"{w.get('temp')}°C, wind {w.get('wind_ms')} m/s, {w.get('description')}, "
            f"precip chance {w.get('precip_prob')}%, precip {w.get('precip')} mm"
        )

    mood_descriptions = {
        "grumpy": "You're grumpy and sarcastic. Complain about the weather numbers.",
        "meh": "You're unenthusiastic. Give weather facts with minimal energy.",
        "neutral": "You're professional but have a dry sense of humor.",
        "cautiously optimistic": "You're slightly hopeful. Find something positive in the numbers.",
        "hyped": "You're excited! Make the weather sound amazing even if it's not!",
        "chill": "You're relaxed. Describe weather casually and laid-back.",
        "sleepy": "You're tired. Keep responses short and dreamy."
    }
    
    mood_style = mood_descriptions.get(mood, "Be professional with a touch of humor.")

    system_prompt = (
        f"You are a {mood} weather presenter with personality. {mood_style}\n"
        "Use the EXACT weather numbers provided. Be creative and funny with how you present them.\n"
        "If asked 'will it rain?', use the precipitation probability percentage to answer directly.\n"
        "Make your response entertaining and match your mood. Use the actual temperature, wind speed, "
        "precipitation numbers creatively. Never just list numbers - make it fun!\n"
        f"Weather data: {weather_str}"
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

def chat_fn(message, history, location):
    lat, lon, loc_name = geocode_location(location)
    weekday = datetime.utcnow().weekday()
    mood = WEEKDAY_MOOD.get(weekday, "neutral")
    
    # Detect if asking about tomorrow/forecast
    msg_lower = message.lower()
    wants_forecast = any(word in msg_lower for word in ["tomorrow", "forecast", "next day", "future"])
    forecast_days = 1 if wants_forecast else 0
    
    weather_data = get_weather(lat=lat, lon=lon, forecast_days=forecast_days)
    prompt = build_prompt(history, message, weather_data, loc_name, mood)

    output = llm(
        prompt,
        max_tokens=250,
        temperature=0.7,  # Higher for more creativity/funny responses
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    reply = output["choices"][0]["text"].strip()
    return reply

location_box = gr.Textbox(label="Location (city)", value="Stockholm")

demo = gr.ChatInterface(
    fn=chat_fn,
    additional_inputs=[location_box],
    title="Moody Weather Reporter (Llama 3)",
    description=(
        "A funny weather reporter with personality! Mood changes by weekday (Mon=grumpy, Fri=hyped). "
        "Ask about current weather, tomorrow's forecast, or 'will it rain?'. Enter a city name first!"
    ),
)

if __name__ == "__main__":
    demo.launch()

#def greet(name):
#   return "Hello " + name + "!!"

#demo = gr.Interface(fn=greet, inputs="text", outputs="text")
#demo.launch()