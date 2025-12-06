<!-- YAML for Hugging Face - uncomment when pushing to HF, comment when pushing to GitHub -->
<!--
---
title: Id2223atw
emoji: 🌤️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Fine-tuned Llama 3.2 Weather Assistant
---
-->

# Lab 2: Fine-Tuned Llama 3.2 Weather Assistant

A weather assistant built with a fine-tuned Llama 3.2 1B model that answers questions about current weather and forecasts using real-time data from Open-Meteo API.

**Live Demo:** https://huggingface.co/spaces/datskiw/id2223atw

## Task 1: Fine-Tuning and UI

### Fine-Tuning

**Notebook Source:** The fine-tuning notebook is based on the [Unsloth](https://github.com/unslothai/unsloth) framework by [unslothai](https://github.com/unslothai), which provides memory-efficient fine-tuning that is 2x faster than standard methods.

- **Model**: Llama 3.2 1B Instruct
- **Method**: LoRA (QLoRA with 4-bit quantization)
- **Dataset**: [FineTome 100K](https://huggingface.co/datasets/mlabonne/FineTome-100k) by Maxime Labonne
- **Training**: Google Colab with T4 GPU
- **LoRA Parameters**: r=16, alpha=16

### GGUF Conversion - Manual Method

Due to RAM limitations, the standard `push_to_hub_gguf()` method crashed. We implemented an **alternative manual conversion method** at the bottom of the notebook:

1. Save merged model to 16-bit: `model.save_pretrained_merged(merged_model_dir, tokenizer, save_method="merged_16bit")`
2. Clone llama.cpp: `git clone https://github.com/ggerganov/llama.cpp.git`
3. Convert to GGUF: `python llama.cpp/convert_hf_to_gguf.py merged_model --outfile llama3-finetome-q8_0.gguf --outtype q8_0`
4. Download from Colab and manually upload to Hugging Face

This produced a ~1.3GB GGUF file (q8_0 quantization) for CPU inference.

### Creative Application

The weather assistant combines:
- Fine-tuned LLM for natural language understanding
- Real-time weather data integration (Open-Meteo API)
- Multi-city support via geocoding
- Current weather and 7-day forecasts
- Verification features (raw data display, API URLs)
- Post-processing to ensure factual accuracy

**Key Features:**
- Natural language queries: "What's the weather in Paris?", "Will it rain tomorrow?"
- Automatic location detection from messages
- Weather condition emojis
- Error handling and retry logic

## Task 2: Model and Data-Centric Improvements

### Model Comparison

We compared three fine-tuned models to select the best base architecture for further optimization:

- **Llama 3.2 1B**: 1.22s inference speed, 67% quality score
- **Llama 3.2 3B**: 2.57s inference speed, 100% quality score  
- **Qwen 2 0.5B**: 0.54s inference speed, 100% quality score

**Test Methodology**: We measured two metrics using a simple comparison script (`compare_models.py`):
1. **Inference Speed**: Average time to generate responses (3 runs per model)
2. **Response Quality**: Percentage of responses that correctly mention weather data (temperature, conditions, precipitation)

**Results**: Qwen 2 0.5B achieved the fastest inference speed (1.21s) while maintaining 100% quality, making it our choice for CPU deployment on Hugging Face Spaces.

**Decision**: We selected **Qwen 2 0.5B** for hyperparameter grid search due to its speed-to-quality ratio.

### Model-Centric Approaches

1. **Hyperparameter Tuning**: Test different learning rates, LoRA ranks (8, 16, 32, 64), and training epochs
2. **Model Architecture**: Compare different base models (Llama 3.2 3B, Mistral 7B), quantization levels (q4_k_m, q5_k_m, q8_0)
3. **Fine-Tuning Strategy**: Task-specific fine-tuning on weather-related instructions, multi-task learning

### Data-Centric Approaches

1. **Weather-Specific Training Data**: Create instruction dataset with weather Q&A examples, synthetic data generation
2. **Data Augmentation**: Combine multiple weather APIs, include historical patterns, diverse locations and temporal variations
3. **Data Quality**: Filter low-quality examples, ensure balanced representation of weather conditions

### Implementation Priority

- **Easiest**: Test different quantization levels, add weather-specific examples to prompts
- **Medium**: Create small weather-specific dataset (100-500 examples), test different LoRA ranks
- **Complex**: Full hyperparameter sweep, large-scale weather dataset creation

### Current Limitations

- Model sometimes hallucinates (mitigated with post-processing)
- Rate limiting from Open-Meteo API (handled with retries)
- Limited to 7-day forecasts


## Usage

1. Visit [Hugging Face Space](https://huggingface.co/spaces/datskiw/id2223atw)
2. Enter a city name or mention it in your message
3. Ask questions like:
   - "What's the weather today?"
   - "Will it rain tomorrow?"
   - "What's the weather in Paris in 3 days?"

## Acknowledgments

- **Unsloth**: Fast fine-tuning framework by [unslothai](https://github.com/unslothai/unsloth)
- **FineTome Dataset**: By [Maxime Labonne](https://huggingface.co/datasets/mlabonne/FineTome-100k)
- **llama.cpp**: GGUF conversion by [ggerganov](https://github.com/ggerganov/llama.cpp)
- **Open-Meteo**: Free weather API

## License

Apache 2.0
