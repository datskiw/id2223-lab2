# Lab 2: Fine-Tuned Llama 3.2 Weather Assistant

A weather assistant built with a fine-tuned Qwen 2 0.5B that answers questions about current weather and forecasts using real-time data from Open-Meteo API.

**Live Demo:** https://huggingface.co/spaces/datskiw/id2223atw

## Task 1: Fine-Tuning and UI

### Fine-Tuning

**Notebook Source:** The fine-tuning notebook is based on the [Unsloth](https://github.com/unslothai/unsloth) framework by [unslothai](https://github.com/unslothai), which provides memory-efficient fine-tuning that is 2x faster than standard methods.


### GGUF Conversion - Manual Method

Due to RAM limitations, the standard `push_to_hub_gguf()` method crashed. We implemented an **alternative manual conversion method** at the bottom of the notebook:

1. Save merged model to 16-bit: `model.save_pretrained_merged(merged_model_dir, tokenizer, save_method="merged_16bit")`
2. Clone llama.cpp: `git clone https://github.com/ggerganov/llama.cpp.git`
3. Convert to GGUF: `python llama.cpp/convert_hf_to_gguf.py merged_model --outfile llama3-finetome-q8_0.gguf --outtype q8_0`
4. Download from Colab and manually upload to Hugging Face

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

### Hyperparameter Grid Search (Qwen 2 0.5B)

- Data: 98,000 train / 2,000 validation samples
- Search: 3 learning rates × 3 weight decays (9 configs, 1 epoch, total batch size 8)
- Best config: lr=5e-4, weight_decay=0.01 → best val loss **1.0551** (≈275s)
- Full sweep (best val losses):

```
LR      Weight Decay   Val Loss   Time (s)
1e-4    0.00           1.0717     277.1
1e-4    0.01           1.0718     275.5
1e-4    0.10           1.0718     275.0
2e-4    0.00           1.0626     274.9
2e-4    0.01           1.0626     275.9
2e-4    0.10           1.0627     274.9
5e-4    0.00           1.0552     274.7
5e-4    0.01           1.0551     276.8
5e-4    0.10           1.0552     275.9
```

### Final Production Model (on Hugging Face Space)

- Base: Qwen 2 0.5B, LoRA, quantized to q8_0
- Data: 100,000 training examples; **1,000 steps**
- Optimizer settings: lr=5e-4, weight_decay=0.1 (best from grid search)
- Observed training loss: started ~1.0 and reached **0.69** at step 1,000 (steady downward trend)
- Deployed model: the 1,000-step checkpoint is the version running on the Hugging Face Space

### Model-Centric Approaches

Building on the model comparison and hyperparameter search above, our main model-centric improvements were:

- **Architecture selection:** We started with Llama 3.2 (1B and 3B) and empirically selected **Qwen 2 0.5B** as the best speed–quality trade-off for CPU inference.
- **Targeted hyperparameter tuning:** A small grid search over learning rate and weight decay (see table above) was used to choose the final training setup, which we then scaled to 100k examples and 1,000 steps.
- **Quantization for deployment:** We quantized the Qwen 2 0.5B + LoRA model to **q8_0**, which kept latency low on CPU while preserving the quality measured in our comparison.

Potential future model-centric work includes varying the LoRA rank and target layers, experimenting with other compact base models, and tuning decoding parameters specifically for factual weather answers.



### Model-Centric Approaches

Building on the model comparison and hyperparameter search above, our main model-centric work was:

- **Choosing the backbone:** We empirically compared Llama 3.2 (1B, 3B) and Qwen 2 0.5B as instruction-tuned chat models and selected **Qwen 2 0.5B** as the best speed–quality trade-off for CPU deployment.
- **Tuning training dynamics:** We ran a small grid search over learning rate and weight decay (see table above) and then trained the final LoRA model for 1,000 steps with the best-performing configuration.
- **Making it deployable:** The chosen model is **quantized to q8_0**, which keeps latency acceptable on the Hugging Face CPU Space while preserving response quality.

If we had more time, additional model-centric directions we would try:

- Varying **LoRA rank** and which layers are adapted, to see how much capacity is actually needed for this dialogue task.
- Experimenting with longer context windows and different decoding settings (temperature, top-p, repetition penalties) tuned to produce factual, concise weather answers.

---

### Data-Centric Approaches

Our current model is fine-tuned on a generic instruction dataset (FineTome) and then combined with a real-time weather API at inference. To make the assistant better at weather-specific dialogue, we identified several data-centric improvements:

1. **Weather-focused instruction data**

   - Create a small instruction dataset where inputs are natural language weather questions and outputs are *ideal* assistant answers (in plain text), e.g.  
     “What’s the weather like in Stockholm today?” → answer that references temperature, conditions, and precipitation.
   - Generate such pairs by querying real weather APIs and turning their JSON responses into chat-style explanations. The LLM would still be trained on text, but grounded in real weather data.

2. **Richer conversational coverage**

   - Add multi-turn examples where the user asks follow-up questions (“What about tomorrow?”, “What if I travel to Paris instead?”) so the model sees realistic dialogue flows rather than only single-turn instructions.
   - Include edge-case questions (extreme heat, storms, snow, wind warnings, comparing two cities, planning a trip) so the model learns to talk sensibly about less common situations.

3. **Data quality and anti-hallucination signals**

   - Filter out examples where the target answer is inconsistent with the underlying API data (e.g. wrong temperature or conditions).
   - Add **negative / correction examples**, where a hallucinated answer is followed by a corrected version, training the model to be conservative when it is unsure.
   - Keep a small held-out set of manually written weather questions and regularly check that fine-tuning does not degrade factual accuracy compared to directly querying the API.


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
