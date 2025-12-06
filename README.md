### Model-Centric Approaches

Building on the model comparison and hyperparameter search above, our main model-centric work was:

- **Choosing the backbone:** We empirically compared Llama 3.2 (1B, 3B) and Qwen 2 0.5B as instruction-tuned chat models and selected **Qwen 2 0.5B** as the best speed–quality trade-off for CPU deployment.
- **Tuning training dynamics:** We ran a small grid search over learning rate and weight decay (see table above) and then trained the final LoRA model for 1,000 steps with the best-performing configuration.
- **Making it deployable:** The chosen model is **quantized to q8_0**, which keeps latency acceptable on the Hugging Face CPU Space while preserving response quality.

If we had more time, additional model-centric directions we would try:

- Varying **LoRA rank** and which layers are adapted, to see how much capacity is actually needed for this dialogue task.
- Experimenting with longer context windows and different decoding settings (temperature, top-p, repetition penalties) tuned to produce factual, concise weather answers.
- Comparing against other small instruction-tuned LLMs (e.g. newer 1B-scale chat models) to see if we can further reduce latency without hurting quality.

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

These data-centric ideas are all about improving the **instruction-style text the LLM sees during fine-tuning**, while still relying on the external weather API at inference time for up-to-date facts.
