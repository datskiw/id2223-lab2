import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Your model repository
GGUF_REPO_ID = "datskiw/llama3-finetome-q8_0"
GGUF_FILENAME = "llama3-finetome-q8_0.gguf"

# Download GGUF model from Hugging Face Hub into the Space
print(f"Downloading {GGUF_FILENAME} from {GGUF_REPO_ID}...")
gguf_path = hf_hub_download(
    repo_id=GGUF_REPO_ID,
    filename=GGUF_FILENAME,
)
print(f"Model downloaded to: {gguf_path}")

# Load model with llama.cpp (CPU inference)
print("Loading model...")
llm = Llama(
    model_path=gguf_path,
    n_ctx=2048,      # Number of context tokens (history length)
    n_threads=4,     # CPU threads - tweak if you hit CPU limits
    n_batch=128,     # Batch size for processing
)
print("Model loaded!")

# Optional: Customize the system prompt to define the assistant's personality
SYSTEM_PROMPT = (
    "You are a helpful assistant fine-tuned on the FineTome instruction dataset. "
    "Answer clearly and concisely."
)

def build_prompt(history, message):
    """Turn chat history + new user message into a prompt using Llama 3.2 format."""
    # Llama 3.2 uses special tokens for conversation format
    prompt = "<|begin_of_text|>"
    
    # Add system message (optional, can be removed if not needed)
    # prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
    
    # Add conversation history (keep last few turns to stay within context)
    recent_history = history[-3:] if history else []
    for user_msg, bot_msg in recent_history:
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{bot_msg}<|eot_id|>"
    
    # Add current user message
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt

def chat_fn(message, history):
    """Handle chat messages - Gradio ChatInterface compatible."""
    if history is None:
        history = []
    
    # Build prompt with conversation history
    prompt = build_prompt(history, message)
    
    # Generate response
    output = llm(
        prompt,
        max_tokens=256,        # Maximum tokens to generate
        temperature=0.7,       # Sampling temperature (0.0 = deterministic, higher = more creative)
        top_p=0.9,            # Nucleus sampling
        repeat_penalty=1.1,    # Reduce repetition
        stop=["<|eot_id|>", "<|end_of_text|>", "User:", "Assistant:"],  # Stop tokens
    )
    
    # Extract the reply
    reply = output["choices"][0]["text"].strip()
    return reply

demo = gr.ChatInterface(
    fn=chat_fn,
    title="FineTuned Llama-3 on FineTome",
    description=(
        "Chat with my Llama-3 model fine-tuned on the FineTome instruction dataset. "
        "Runs on CPU via GGUF / llama.cpp."
    ),
)

if __name__ == "__main__":
    demo.launch()
