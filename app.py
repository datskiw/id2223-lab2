import gradio as gr
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

MODEL_ID = "datskiw/llama3-finetome-lora"

# Load tokenizer and LoRA model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",            # Space is CPU Basic
    torch_dtype=torch.float32,   # safer on CPU
)

# (Optional) limit CPU usage a bit
torch.set_num_threads(4)

def chat_fn(message, history):
    """
    Gradio ChatInterface calls:
      fn(message: str, history: list[list[str, str]])
    and expects:
      return reply: str, history: list[list[str, str]]
    """
    if history is None:
        history = []

    # Build a compact conversation prompt.
    # To keep it fast, we only use the last few turns.
    recent_history = history[-3:]

    conversation = ""
    for user, bot in recent_history:
        conversation += f"<|user|>\n{user}\n<|assistant|>\n{bot}\n"
    conversation += f"<|user|>\n{message}\n<|assistant|>\n"

    inputs = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=True,
        max_length=512,   # shorter context -> faster
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,    # keep this small for speed
            do_sample=False,      # greedy is faster & more stable on CPU
            num_beams=1,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Simple way: take everything after the last assistant tag
    if "<|assistant|>" in text:
        reply = text.split("<|assistant|>")[-1].strip()
    else:
        reply = text.strip()

    history.append((message, reply))
    return reply, history

demo = gr.ChatInterface(
    fn=chat_fn,
    title="FineTome Llama Chat",
    description="Chat with my FineTome-finetuned Llama model (CPU).",
)

if __name__ == "__main__":
    demo.launch()
