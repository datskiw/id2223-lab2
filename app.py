import gradio as gr
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

MODEL_ID = "datskiw/llama3-finetome-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = model.merge_and_unload() 

def chat_fn(message, history):
    # Gradio calls this with: message (str), history (list of [user, bot])
    if history is None:
        history = []

    # Build conversation text
    conversation = ""
    for user, bot in history:
        conversation += f"<|user|>\n{user}\n<|assistant|>\n{bot}\n"
    conversation += f"<|user|>\n{message}\n<|assistant|>\n"

    inputs = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    reply = text.split("<|assistant|>")[-1].strip()

    history.append((message, reply))
    return reply, history

demo = gr.ChatInterface(
    fn=chat_fn,
    title="FineTome Llama Chat",
    description="Chat with my FineTome-finetuned Llama model.",
)

if __name__ == "__main__":
    demo.launch()
