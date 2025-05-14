from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from config import MODEL_NAME, OUTPUT_DIR

def tokenize_data(dataset, tokenizer):
    def tokenize_fn(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    return dataset.map(tokenize_fn, batched=True)

def train_model(dataset):
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    dataset = tokenize_data(dataset, tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=2,
        logging_steps=20,
        logging_dir="./logs",
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)