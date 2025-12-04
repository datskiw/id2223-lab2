import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

MODEL_ID = "datskiw/llama3-finetome-q8_0"
MODEL_FILE = "llama3-finetome-q8_0.gguf"

# Download GGUF model from Hugging Face if not present
if not os.path.exists(MODEL_FILE):
    print(f"Downloading {MODEL_FILE} from {MODEL_ID}...")
    MODEL_FILE = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILE,
        local_dir=".",
    )
    print(f"Downloaded to: {MODEL_FILE}")

# Load GGUF model with llama-cpp-python
print("Loading GGUF model...")
llm = Llama(
    model_path=MODEL_FILE,
    n_ctx=2048,          # Context window
    n_threads=4,         # CPU threads
    verbose=False,       # Set to True for debugging
)
print("Model loaded!")

def chat_fn(message, history):
    """
    Gradio ChatInterface calls:
      fn(message: str, history: list[list[str, str]])
    and expects:
      return reply: str, history: list[list[str, str]]
    """
    if history is None:
        history = []

    # Build conversation prompt using Llama 3.2 format
    # Llama 3.2 uses: <|begin_of_text|><|start_header_id|>user<|end_header_id|>...<|eot_id|>
    recent_history = history[-3:]  # Keep last 3 turns for context
    
    # Start with begin_of_text token
    prompt = "<|begin_of_text|>"
    
    # Add conversation history
    for user_msg, bot_msg in recent_history:
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{bot_msg}<|eot_id|>"
    
    # Add current user message
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    # Generate response using llama-cpp-python
    response = llm(
        prompt,
        max_tokens=128,        # Maximum tokens to generate
        temperature=0.7,       # Sampling temperature
        top_p=0.9,            # Nucleus sampling
        repeat_penalty=1.1,    # Reduce repetition
        stop=["<|eot_id|>", "<|end_of_text|>"],  # Stop tokens
        echo=False,            # Don't echo the prompt
    )

    # Extract the reply
    reply = response["choices"][0]["text"].strip()
    
    history.append((message, reply))
    return reply, history

demo = gr.ChatInterface(
    fn=chat_fn,
    title="FineTome Llama Chat",
    description="Chat with my FineTome-finetuned Llama model (CPU).",
)

if __name__ == "__main__":
    demo.launch()
