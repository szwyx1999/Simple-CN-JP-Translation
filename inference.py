from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="Hunyuan-MT-7B")
parser.add_argument("--language", type=str, default="Chinese")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model from {args.model_name_or_path} on {device}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)

print("Model loaded")

input_text = ""
curr_text = input("Enter the text to translate (type 'eos' to end): ")
while curr_text != "eos":
    input_text += f"\n{curr_text}"
    curr_text = input("Enter the text to translate (type 'eos' to end): ")

print(f"Input text: {input_text}")

messages = [
    {"role": "user", "content": f"Translate the following segment into {args.language}, without additional explanation.\n\n{input_text}"},
]

print("Tokenizing input")
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)

print("Generating output")

tokenized_chat = tokenized_chat.to(device)

outputs = model.generate(tokenized_chat, max_new_tokens=2048)
output_text = tokenizer.decode(outputs[0])

print("Output:")
# output only the translated text
# output_text = output_text.split("<|extra_0|>")[1].split("<|eos|>")[0]
print(output_text)