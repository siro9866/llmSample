import torch
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from config import OUTPUT_DIR

def generate(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)
    model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    result = generator(prompt, max_length=100, do_sample=True)
    print(result[0]["generated_text"])

def generate_mac(prompt):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)
    model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR).to(device)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output = model.generate(input_ids, max_length=100, do_sample=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text)