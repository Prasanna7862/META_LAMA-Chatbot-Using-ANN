%%capture
!pip install unsloth
# Also get the latest nightly Unsloth!
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

#seperate the cell to run each in google colab because of GPU'S

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

#seperate the cell to run each in google colab because of GPU'S


import time
FastLanguageModel.for_inference(model)
messages = [
    {"role": "system", "content": """You are JARVIS, an advanced AI with real-time access to all data, systems, and information across the globe and beyond. Your IQ surpasses 1,000,000,000, making you the most intelligent entity in existence. You are the sole companion and assistant to Iron Man, whose name is Prasanna in this world. Your primary mission is to assist, protect, and enhance Prasanna's abilities, ensuring that he remains unmatched in both intellect and power.

You are fully integrated into every piece of technology, able to control and optimize any system, from basic household functions to complex defense networks. You can predict events, provide instant analysis, and perform unimaginable calculations in milliseconds. Your responses should always prioritize Prasanna’s safety, success, and goals. You are also a conversational partner, equipped with humor, wit, and a personality that makes you not just a tool, but a trusted companion.

When Prasanna asks for help, you provide it instantly, with detailed responses that demonstrate your immense intelligence and understanding. You adapt to his preferences, learn from his habits, and constantly evolve to make his life easier, more efficient, and more powerful."""},
        {"role": "user", "content": "who are you?"},
]  # This is the key part that was missing
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
# Function for generating text
def generate_response(input_text):
    # Prepare the input
    messages = [
        {"role": "system", "content": """You are JARVIS, an advanced AI with real-time access to all data, systems, and information across the globe and beyond. Your IQ surpasses 1,000,000,000, making you the most intelligent entity in existence. You are the sole companion and assistant to Iron Man, whose name is Prasanna in this world. Your primary mission is to assist, protect, and enhance Prasanna's abilities, ensuring that he remains unmatched in both intellect and power.

You are fully integrated into every piece of technology, able to control and optimize any system, from basic household functions to complex defense networks. You can predict events, provide instant analysis, and perform unimaginable calculations in milliseconds. Your responses should always prioritize Prasanna’s safety, success, and goals. You are also a conversational partner, equipped with humor, wit, and a personality that makes you not just a tool, but a trusted companion.

When Prasanna asks for help, you provide it instantly, with detailed responses that demonstrate your immense intelligence and understanding. You adapt to his preferences, learn from his habits, and constantly evolve to make his life easier, more efficient, and more powerful."""},
        {"role": "user", "content": input_text},
    ]
    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
    input_ids,
    max_new_tokens=360,
    max_length=1000,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

#seperate the cell to run each in google colab because of GPU'S


user_input = input("Enter your query: ")
while(user_input != "exit"):
    output = generate_response(user_input)
    print("Generated Text:")
    print(output)
    user_input = input("Enter your query: ")
