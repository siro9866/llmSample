from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from config import OUTPUT_DIR

def generate(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)
    model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    result = generator(prompt, max_length=100, do_sample=True)
    print(result[0]["generated_text"])
