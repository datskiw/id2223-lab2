import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import requests

GGUF_REPO_ID = "datskiw/llama3-finetome-q8_0"
GGUF_FILENAME = "llama3-finetome-q8_0.gguf"

# Download and load model
gguf_path = hf_hub_download(repo_id=GGUF_REPO_ID, filename=GGUF_FILENAME)
llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=4, n_batch=128)

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

def geocode_location(location: str):
    """Get lat/lon for a city name."""
    if not location:
        return 59.33, 18.07, "Stockholm"
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(location)}&count=1&format=json"
        r = requests.get(url, timeout=5)
        data = r.json()
        if data.get("results"):
            hit = data["results"][0]
            return hit["latitude"], hit["longitude"], hit.get("name", location)
    except:
        pass
    return 59.33, 18.07, "Stockholm"


def get_weather(lat, lon, day=0):
    """Get weather for today (day=0) or forecast (day=1-7)."""
    if day == 0:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weathercode,wind_speed_10m,precipitation,precipitation_probability"
    else:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&forecast_days=7&daily=temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
    
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        
        if day == 0:
            cur = data["current"]
            return {
                "temp": cur["temperature_2m"],
                "wind": round(cur["wind_speed_10m"] / 3.6, 1),
                "description": WEATHER_CODES.get(cur["weathercode"], "unknown"),
                "precip": cur["precipitation"],
                "precip_prob": cur["precipitation_probability"],
            }
        else:
            daily = data["daily"]
            idx = min(day - 1, 6)
            return {
                "temp_max": daily["temperature_2m_max"][idx],
                "temp_min": daily["temperature_2m_min"][idx],
                "wind": round(daily["wind_speed_10m_max"][idx] / 3.6, 1),
                "description": WEATHER_CODES.get(daily["weathercode"][idx], "unknown"),
                "precip": daily["precipitation_sum"][idx],
                "precip_prob": daily["precipitation_probability_max"][idx],
            }
    except Exception as e:
        return {"error": str(e)}

def parse_day(message):
    """Determine which day: 0=today, 1=tomorrow, 2=day after, etc."""
    msg = message.lower()
    if "today" in msg:
        return 0
    if "day after tomorrow" in msg:
        return 2
    if "tomorrow" in msg:
        return 1
    for i in range(2, 8):
        if f"in {i} days" in msg or f"day {i}" in msg:
            return i
    return 0  # default to today

def build_prompt(history, message, weather_data, location):
    """Build prompt with weather data."""
    if "temp_max" in weather_data:
        w = weather_data
        weather_info = f"High {w['temp_max']}°C, Low {w['temp_min']}°C, wind {w['wind']} m/s, {w['description']}, precipitation chance {w['precip_prob']}%, precipitation {w['precip']} mm"
    else:
        w = weather_data
        weather_info = f"Temperature {w['temp']}°C, wind {w['wind']} m/s, {w['description']}, precipitation chance {w['precip_prob']}%, precipitation {w['precip']} mm"
    
    system_prompt = (
        f"You are a helpful weather assistant for {location}. "
        f"CRITICAL: You MUST use the EXACT numbers provided. Do NOT invent or change any values.\n\n"
        f"Weather data: {weather_info}\n\n"
        "Answer naturally in plain language using these EXACT numbers. "
        "If asked 'will it rain?' use the exact precipitation probability percentage. "
        "If asked about sunny/cloudy, use the exact description provided. "
        "DO NOT make up temperatures, wind speeds, or any other values - use only what is provided above."
    )
    
    prompt = "<|begin_of_text|>"
    prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    
    if history:
        for turn in history[-3:]:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{turn[0]}<|eot_id|>"
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{turn[1]}<|eot_id|>"
    
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def chat_fn(message, history, location):
    lat, lon, loc_name = geocode_location(location)
    day = parse_day(message)
    weather_data = get_weather(lat, lon, day)
    
    if weather_data.get("error"):
        return f"Sorry, couldn't fetch weather data: {weather_data['error']}"
    
    prompt = build_prompt(history, message, weather_data, loc_name)
    output = llm(prompt, max_tokens=200, temperature=0.7, stop=["<|eot_id|>", "<|end_of_text|>"])
    reply = output["choices"][0]["text"].strip()
    
    # Append raw data for verification (from Open-Meteo API)
    if "temp_max" in weather_data:
        raw_data = f"\n\n[Raw data from Open-Meteo (day {day}): High {weather_data['temp_max']}°C, Low {weather_data['temp_min']}°C, wind {weather_data['wind']} m/s, {weather_data['description']}, precip chance {weather_data['precip_prob']}%, precip {weather_data['precip']} mm]"
    else:
        raw_data = f"\n\n[Raw data from Open-Meteo (day {day}): Temp {weather_data['temp']}°C, wind {weather_data['wind']} m/s, {weather_data['description']}, precip chance {weather_data['precip_prob']}%, precip {weather_data['precip']} mm]"
    
    return reply + raw_data

demo = gr.ChatInterface(
    fn=chat_fn,
    additional_inputs=[gr.Textbox(label="Location (city)", value="Stockholm")],
    title="Weather Assistant",
    description="Ask about weather today, tomorrow, or any of the next 7 days. Enter a city name first!",
)

if __name__ == "__main__":
    demo.launch()