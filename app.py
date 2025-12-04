import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
#HF will read the requirments.txt and download via pip installs so we can use these libraries.

GGUF_REPO_ID = "datskiw/llama3-finetome-q8_0"
GGUF_FILENAME = "llama3-finetome-q8_0.gguf"

#Download GGUF model from Hugging Face Hub into the Space
gguf_path = hf_hub_download(
    repo_id=GGUF_REPO_ID,
    filename=GGUF_FILENAME,
)

#Load model with llama.cpp (CPU inference)
llm = Llama(
    model_path=gguf_path,
    n_ctx=2048,      #number of context tokens,, context (history) length that model can see at once
    n_threads=4,     # tweak if you hit CPU limits
    n_batch=128,
)

#global instruction. Helps align responses in a certain style.
#its here we can define the purpose of our app and add creative ways of how ppl will use our LLM.
SYSTEM_PROMPT = (
    ""
)

#when building the prompt for the LLM, we include the history of prev prompts
#this way we get context. History length is decided by n_ctx variable and if history overwrite this threshold older lines are ignored (I hope).

def build_promptOG(history, message):
    """Turn chat history + new user message into a plain-text prompt."""
    lines = [f"System: {SYSTEM_PROMPT}"]
    for user_msg, bot_msg in history:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {bot_msg}")
    lines.append(f"User: {message}")
    lines.append("Assistant:") #empty because this will be the output for model to gen
    return "\n".join(lines)

def build_prompt(history, message):
    """Turn chat history + new user message into a plain-text prompt."""
    lines = [f"System: {SYSTEM_PROMPT}"]

    for turn in history:
        # turn might be (user, bot) or (user, bot, extra_stuff, ...)
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_msg, bot_msg = turn[0], turn[1]
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {bot_msg}")
        # if it's some other weird shape, just ignore it

    lines.append(f"User: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)

def chat_fn(message, history):
    prompt = build_prompt(history, message)

    output = llm( 
        prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        stop=["User:", "Assistant:", "System:"], #stop when any of these substrings are generated (so we dont re-loop)
        #n=1, #num of possible ouptut entries to choose from. ! bcus CPU load and netw latence already gives long answ time.
        #^^Aparently this llama-cpp ver is a HIGH LEVEL wrapper not handling more than 1 possible outputs compared to OpenAI-style completion APIs. More low level APIS do exists though.
    )
    
    reply = output["choices"][0]["text"].strip() #out of a list of possible completions we take the first and best completion
    return reply

demo = gr.ChatInterface( #a gradio class which shows a chat bubble UIwhich passes message and history into fn and display returned string from fn
    fn=chat_fn,
    title="FineTuned Llama-3 on FineTome",
    description=(
        "Chat with my Llama-3 model fine-tuned on the FineTome instruction dataset. "
        "Runs on CPU via GGUF / llama.cpp."
    ),
)

if __name__ == "__main__":
    demo.launch()

#def greet(name):
#   return "Hello " + name + "!!"

#demo = gr.Interface(fn=greet, inputs="text", outputs="text")
#demo.launch()