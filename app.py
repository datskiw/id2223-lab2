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
    """Fetch current or forecast weather from Open-Meteo. forecast_days=0 for today, 1=tomorrow, etc."""
    if forecast_days == 0:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,weathercode,wind_speed_10m,precipitation,precipitation_probability"
        )
    else:
        # Always fetch 7 days of forecast data
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&forecast_days=7"
            "&daily=temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
        )
    headers = {"User-Agent": "gradio-llama-weather/1.0"}
    last_err = "unknown error"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=8, headers=headers)
            if r.status_code == 429:
                last_err = "429 Too Many Requests"
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
                # Forecast weather - fetch up to 7 days
                daily = data.get("daily", {})
                temps_max = daily.get("temperature_2m_max", [])
                temps_min = daily.get("temperature_2m_min", [])
                codes = daily.get("weathercode", [])
                precip_sum = daily.get("precipitation_sum", [])
                precip_prob_max = daily.get("precipitation_probability_max", [])
                wind_max = daily.get("wind_speed_10m_max", [])
                # forecast_days=1 means tomorrow (index 0), forecast_days=2 means day after (index 1), etc.
                day_idx = min(forecast_days - 1, len(temps_max) - 1) if forecast_days > 0 else 0
                wind_ms = round(float(wind_max[day_idx]) / 3.6, 1) if wind_max and day_idx < len(wind_max) and wind_max[day_idx] is not None else None
                return {
                    "temp_max": temps_max[day_idx] if temps_max and day_idx < len(temps_max) else None,
                    "temp_min": temps_min[day_idx] if temps_min and day_idx < len(temps_min) else None,
                    "wind_ms": wind_ms,
                    "description": WEATHER_CODES.get(codes[day_idx], "unknown") if codes and day_idx < len(codes) else "unknown",
                    "precip": precip_sum[day_idx] if precip_sum and day_idx < len(precip_sum) else None,
                    "precip_prob": precip_prob_max[day_idx] if precip_prob_max and day_idx < len(precip_prob_max) else None,
                    "type": "forecast",
                    "day": forecast_days
                }
        except Exception as e:
            last_err = e
            print(f"[weather] attempt {attempt+1} failed: {e}")
            time.sleep(0.5 * (attempt + 1))
    return {"error": str(last_err)}

def build_prompt(history, message, weather_data, location_name, mood, forecast_days=0):
    """Turn chat history + new user message into a Llama 3.2 style prompt."""
    if weather_data.get("type") == "forecast":
        w = weather_data
        weather_str = (
            f"High {w.get('temp_max')}°C, Low {w.get('temp_min')}°C, "
            f"wind {w.get('wind_ms')} m/s, {w.get('description')}, "
            f"precip chance {w.get('precip_prob')}%, precip {w.get('precip')} mm"
        )
    else:
        w = weather_data
        weather_str = (
            f"{w.get('temp')}°C, wind {w.get('wind_ms')} m/s, {w.get('description')}, "
            f"precip chance {w.get('precip_prob')}%, precip {w.get('precip')} mm"
        )

    mood_descriptions = {
        "grumpy": "You're GRUMPY and SARCASTIC. Complain dramatically about the weather. Use phrases like 'Ugh', 'Of course', 'Typical'. Make it funny by being overly dramatic.",
        "meh": "You're UNENTHUSIASTIC. Give weather facts with minimal energy but add dry humor. Use 'meh', 'whatever', 'I guess'.",
        "neutral": "You're professional but have a DRY SENSE OF HUMOR. Make subtle jokes about the weather numbers.",
        "cautiously optimistic": "You're SLIGHTLY HOPEFUL. Find something positive in the numbers but be realistic. Use 'maybe', 'could be worse', 'at least'.",
        "hyped": "You're EXCITED! Make EVERYTHING sound amazing! Use exclamation marks! Be enthusiastic! Even boring weather is 'incredible'!",
        "chill": "You're RELAXED. Describe weather casually and laid-back. Use 'dude', 'pretty', 'kinda'. Be cool about it.",
        "sleepy": "You're TIRED. Keep responses short and dreamy. Use 'zzz', 'yawn', 'so sleepy'. Make it cute."
    }
    
    mood_style = mood_descriptions.get(mood, "Be professional with humor.")

    day_label = "today" if weather_data.get("type") == "current" else f"in {forecast_days} day{'s' if forecast_days > 1 else ''}"
    
    system_prompt = (
        f"You are a {mood} weather presenter. {mood_style}\n\n"
        "IMPORTANT:\n"
        "- Use the EXACT weather numbers provided - temperature, wind, precipitation probability, precipitation amount.\n"
        "- Present the weather in an entertaining way that matches your mood.\n"
        "- If asked 'will it rain?', use the precipitation probability percentage to answer directly.\n"
        "- Be creative and funny, but always use the actual data provided.\n\n"
        f"Weather data for {day_label}: {weather_str}\n\n"
        f"Give a {mood} weather report using these exact numbers!"
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

def parse_day_request(message):
    """Parse message to determine which day's weather is requested (0=today, 1=tomorrow, etc.)"""
    msg_lower = message.lower()
    
    # Check for specific day numbers
    for i in range(8):
        if f"day {i}" in msg_lower or f"{i} days" in msg_lower:
            return i
    
    # Check for day names
    day_keywords = {
        "today": 0,
        "tomorrow": 1,
        "day after tomorrow": 2,
        "in 2 days": 2,
        "in 3 days": 3,
        "in 4 days": 4,
        "in 5 days": 5,
        "in 6 days": 6,
        "in 7 days": 7,
    }
    for keyword, days in day_keywords.items():
        if keyword in msg_lower:
            return days
    
    # Default: check if asking about forecast/future
    if any(word in msg_lower for word in ["tomorrow", "forecast", "future", "next"]):
        return 1  # Default to tomorrow if asking about future
    
    return 0  # Default to today

def chat_fn(message, history, location):
    lat, lon, loc_name = geocode_location(location)
    weekday = datetime.utcnow().weekday()
    mood = WEEKDAY_MOOD.get(weekday, "neutral")
    
    # Determine which day's weather to fetch
    forecast_days = parse_day_request(message)
    
    weather_data = get_weather(lat=lat, lon=lon, forecast_days=forecast_days)

    # Handle errors gracefully
    if weather_data.get("error"):
        err = weather_data.get("error")
        mood_lines = {
            "grumpy": "Ugh, even the weather feed bailed. Try again later.",
            "meh": "Meh, no data right now. Maybe later.",
            "neutral": "Weather data unavailable at the moment.",
            "cautiously optimistic": "No data now, but maybe it'll show up soon.",
            "hyped": "Whoa, the weather API took a coffee break!",
            "chill": "No data, dude. Let's try again later.",
            "sleepy": "Zzz… no weather data. Maybe after a nap."
        }
        return f"Weather ({loc_name}): unavailable ({err}). {mood_lines.get(mood, 'Try again later.')}"
    
    prompt = build_prompt(history, message, weather_data, loc_name, mood, forecast_days)

    output = llm(
        prompt,
        max_tokens=300,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    return output["choices"][0]["text"].strip()

location_box = gr.Textbox(label="Location (city)", value="Stockholm")

demo = gr.ChatInterface(
    fn=chat_fn,
    additional_inputs=[location_box],
    title="Moody Weather Reporter (Llama 3)",
    description=(
        "A moody weather reporter! Ask about weather today, tomorrow, or any of the next 7 days. "
        "Mood changes by weekday (Mon=grumpy, Fri=hyped). Enter a city name, then ask your question!"
    ),
)

if __name__ == "__main__":
    demo.launch()

#def greet(name):
#   return "Hello " + name + "!!"

#demo = gr.Interface(fn=greet, inputs="text", outputs="text")
#demo.launch()