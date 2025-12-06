---
title: Id2223atw
emoji: 🌤️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.0.1"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Fine-tuned Llama/Qwen weather assistant (GGUF via llama.cpp)
---

# Weather Assistant (HF Space)

This README is for the Hugging Face Space only and includes the required YAML front matter. The app runs `app.py` with `llama-cpp-python` using the GGUF model downloaded from the Hub.

## How to run locally
```
pip install -r requirements.txt
python app.py
```

## Models
- Default: `datskiw/qwen2-0.5b-finetome-q8_0`
- Switch in `app.py` to Llama 1B/3B if desired.


