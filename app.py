import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import requests
import time
import re

# ===== MODEL CONFIGURATION =====
# Chat format: "llama3" or "chatml" (for Qwen)
CHAT_FORMAT = "chatml"  # Change to "llama3" for Llama models

# Model selection - uncomment the one you want to use:

# Llama 3.2 1B
# GGUF_REPO_ID = "datskiw/llama3-1B-finetome-q8_0"
# GGUF_FILENAME = "llama3-1B-finetome-q8_0.gguf"

# Llama 3.2 3B
# GGUF_REPO_ID = "datskiw/llama3-3B-finetome-q8_0"
# GGUF_FILENAME = "llama3-3B-finetome-q8_0.gguf"

# Qwen 2 0.5B
GGUF_REPO_ID = "datskiw/qwen2-0.5b-finetome-q8_0"
GGUF_FILENAME = "qwen2-0.5b-finetome-q8_0.gguf"

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


def extract_location_from_message(message: str) -> str:
    """Try to pull a location from the user message, e.g., 'weather in Barcelona'."""
    if not message:
        return ""
    match = re.search(r"\bweather\s+(?:in|at|for)\s+([A-Za-z\s,'-]{2,60})", message, re.IGNORECASE)
    if match:
        loc = match.group(1).strip(" ,")
        if len(loc) >= 2:
            return loc
    return ""


def get_weather(lat, lon, day=0):
    """Get weather for today (day=0) or forecast (day=1-7)."""
    if day == 0:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weathercode,wind_speed_10m,precipitation,precipitation_probability"
    else:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&forecast_days=7&daily=temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
    
    # Retry logic for rate limits
    max_retries = 5  # allow a couple more retries
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=5)
            
            # Handle rate limiting with exponential backoff
            if r.status_code == 429:
                if attempt < max_retries - 1:
                    # exponential backoff with slight jitter
                    wait_time = (2 ** attempt) + 1 + (0.2 * attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    return {"error": "Rate limit exceeded. Please try again in a minute or two."}
            
            r.raise_for_status()  # Check for other HTTP errors
            data = r.json()
            
            # Check if API returned an error
            if "error" in data:
                return {"error": f"API error: {data['error']}"}
            
            if day == 0:
                if "current" not in data:
                    return {"error": f"API response missing 'current' key. Available keys: {list(data.keys())}"}
                cur = data["current"]
                return {
                    "temp": cur["temperature_2m"],
                    "wind": round(cur["wind_speed_10m"] / 3.6, 1),
                    "description": WEATHER_CODES.get(cur["weathercode"], "unknown"),
                    "precip": cur["precipitation"],
                    "precip_prob": cur["precipitation_probability"],
                    "api_url": url,
                    "time": cur.get("time", "N/A"),
                }
            else:
                if "daily" not in data:
                    return {"error": f"API response missing 'daily' key. Available keys: {list(data.keys())}"}
                daily = data["daily"]
                idx = min(day - 1, 6)
                # Get the date for this forecast day
                dates = daily.get("time", [])
                forecast_date = dates[idx] if dates and idx < len(dates) else "N/A"
                return {
                    "temp_max": daily["temperature_2m_max"][idx],
                    "temp_min": daily["temperature_2m_min"][idx],
                    "wind": round(daily["wind_speed_10m_max"][idx] / 3.6, 1),
                    "description": WEATHER_CODES.get(daily["weathercode"][idx], "unknown"),
                    "precip": daily["precipitation_sum"][idx],
                    "precip_prob": daily["precipitation_probability_max"][idx],
                    "api_url": url,
                    "date": forecast_date,
                    "day_index": idx,
                }
            # Success - break out of retry loop
            break
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return {"error": f"Request failed after {max_retries} attempts: {str(e)}"}
            time.sleep(1 * (attempt + 1))  # Wait before retry
            continue
        except KeyError as e:
            return {"error": f"Missing key in API response: {e}"}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": f"Unexpected error after {max_retries} attempts: {str(e)}"}
            time.sleep(1 * (attempt + 1))
            continue
    
    # This shouldn't be reached, but just in case
    return {"error": "Failed to fetch weather data after retries"}

def parse_day(message):
    """Determine which day: 0=today, 1=tomorrow, 2=day after, etc."""
    msg = message.lower()
    # Word-to-number mapping for 0-7
    word_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7
    }
    for word, num in word_map.items():
        if f"in {word} days" in msg or f"day {word}" in msg:
            return num
    # Numeric patterns
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

def get_condition_emoji(description: str) -> str:
    """Map weather description to an emoji for quick flair."""
    desc = (description or "").lower()
    if any(word in desc for word in ["thunder", "storm"]):
        return "⛈️"
    if any(word in desc for word in ["snow", "sleet"]):
        return "❄️"
    if any(word in desc for word in ["rain", "drizzle", "shower"]):
        return "🌧️"
    if any(word in desc for word in ["fog", "mist", "rime"]):
        return "🌫️"
    if "overcast" in desc or "cloud" in desc:
        return "☁️"
    if "clear" in desc or "sun" in desc:
        return "☀️"
    return "🌍"

def is_general_weather_question(message):
    """Detect if this is a general weather question (not specific like 'will it rain?')."""
    msg_lower = message.lower()
    general_keywords = ["what is the weather", "what's the weather", "how's the weather", 
                       "how is the weather", "weather today", "weather tomorrow",
                       "tell me about the weather", "describe the weather"]
    specific_keywords = ["will it rain", "rain", "sunny", "cloudy", "wind", "temperature", "temp"]

    # Check if it's a general question
    is_general = any(keyword in msg_lower for keyword in general_keywords)
    
    # But not if it's asking something specific
    is_specific = any(keyword in msg_lower for keyword in specific_keywords)
    
    # If it's general and not asking something specific, it's a general weather question
    return is_general and not is_specific

def build_prompt(history, message, weather_data, location):
    """Build prompt with weather data. Supports both Llama 3.2 and Qwen (ChatML) formats."""
    # Use configured chat format
    is_chatml = (CHAT_FORMAT.lower() == "chatml")
    
    if "temp_max" in weather_data:
        w = weather_data
        weather_info = f"High {w['temp_max']}°C, Low {w['temp_min']}°C, wind {w['wind']} m/s, {w['description']}, precipitation chance {w['precip_prob']}%, precipitation {w['precip']} mm"
    else:
        w = weather_data
        weather_info = f"Temperature {w['temp']}°C, wind {w['wind']} m/s, {w['description']}, precipitation chance {w['precip_prob']}%, precipitation {w['precip']} mm"
    
    system_prompt = (
        """
        You are a professional weather reporter. Only use the weather data given here: {weather_info}.
        Give short answers (1-3 sentences) summarizing the weather data that the use asks about.
        Dont make up any weather data and only use the given weather data.
        If you dont have the required weather data, then say that you are unsure.
        """

    )


    
    
    if is_chatml:
        # ChatML format for Qwen
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        if history:
            for turn in history[-3:]:
                if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    prompt += f"<|im_start|>user\n{turn[0]}<|im_end|>\n"
                    prompt += f"<|im_start|>assistant\n{turn[1]}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
    else:
        # Llama 3.2 format
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

def chat_fn(message, history, location, show_raw_data):
    # Prefer explicit input; if not provided or is default Stockholm, try to parse from message
    lat, lon, loc_name = geocode_location(location)
    extracted_loc = extract_location_from_message(message)
    if extracted_loc and (not location or location.strip().lower() == "stockholm"):
        lat, lon, loc_name = geocode_location(extracted_loc)
    day = parse_day(message)
    weather_data = get_weather(lat, lon, day)
    
    if weather_data.get("error"):
        # In streaming mode we must yield, not return a plain string
        yield f"Sorry, couldn't fetch weather data: {weather_data['error']}"
        return
    
    prompt = build_prompt(history, message, weather_data, loc_name)
    # Use appropriate stop tokens based on chat format
    is_chatml = (CHAT_FORMAT.lower() == "chatml")
    stop_tokens = ["<|im_end|>", "<|end_of_text|>"] if is_chatml else ["<|eot_id|>", "<|end_of_text|>"]
    
    # Stream tokens for real-time display (similar to TextStreamer)
    reply = ""
    stream = llm(prompt, max_tokens=200, temperature=0.7, stop=stop_tokens, stream=True)
    
    for output in stream:
        if "choices" in output and len(output["choices"]) > 0:
            token = output["choices"][0].get("text", "")
            reply += token
            yield reply  # Yield partial response for streaming
    
    # Final processing on complete reply
    original_reply = reply.strip()
    reply = original_reply
    
    if show_raw_data:
        verify_url = f"https://open-meteo.com/en/docs#latitude={lat}&longitude={lon}"
        
        if "temp_max" in weather_data:
            date_info = f", date: {weather_data.get('date', 'N/A')}" if weather_data.get('date') else ""
            raw_data = (
                f"\n\n[Raw data from Open-Meteo API (day {day}, index {weather_data.get('day_index', 'N/A')}{date_info}): "
                f"High {weather_data['temp_max']}°C, Low {weather_data['temp_min']}°C, wind {weather_data['wind']} m/s, "
                f"{weather_data['description']}, precip chance {weather_data['precip_prob']}%, precip {weather_data['precip']} mm]\n"
                f"[API URL: {weather_data.get('api_url', 'N/A')}]\n"
                f"[Verify on Open-Meteo: {verify_url}]"
            )
        else:
            time_info = f", time: {weather_data.get('time', 'N/A')}" if weather_data.get('time') else ""
            raw_data = (
                f"\n\n[Raw data from Open-Meteo API (day {day}{time_info}): "
                f"Temp {weather_data['temp']}°C, wind {weather_data['wind']} m/s, {weather_data['description']}, "
                f"precip chance {weather_data['precip_prob']}%, precip {weather_data['precip']} mm]\n"
                f"[API URL: {weather_data.get('api_url', 'N/A')}]\n"
                f"[Verify on Open-Meteo: {verify_url}]"
            )
        # Yield final result with raw data
        yield reply + raw_data
        return
    
    # Yield final processed result (in case post-processing changed it)
    yield reply

demo = gr.ChatInterface(
    fn=chat_fn,
    additional_inputs=[
        gr.Textbox(label="Location (city)", value="Stockholm"),
        gr.Checkbox(label="Show raw data", value=False, info="Display raw API data and verification links")
    ],
    title="Weather Assistant",
    description="Ask about weather today, tomorrow, or any of the next 7 days. Enter a city name first!",
)

if __name__ == "__main__":
    demo.launch()