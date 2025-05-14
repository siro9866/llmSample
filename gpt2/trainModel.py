import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from config import MODEL_NAME, OUTPUT_DIR, LOG_DIR
import os

def tokenize_data(dataset, tokenizer):
    def tokenize_fn(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    return dataset.map(tokenize_fn, batched=True)


def train_model(dataset):
    # 🔸 1. MPS or CPU 디바이스 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"훈련 디바이스: {device}")

    # 🔸 2. Tokenizer 및 Model 설정
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2는 pad_token 없음

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.to(device)

    # 🔸 3. 데이터 전처리
    dataset = tokenize_data(dataset, tokenizer)

    # 🔸 4. 저장 경로 생성 (Trainer보다 먼저)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 🔸 5. 학습 인자 설정
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        num_train_epochs=5,
        save_steps=100,
        save_total_limit=2,
        logging_steps=20,
        logging_dir=LOG_DIR,
        fp16=False,  # MPS는 FP16 미지원
        evaluation_strategy="no",  # 검증 데이터가 있으면 'steps' or 'epoch'
        report_to="none",  # wandb/logging 방지
    )

    # 🔸 6. Trainer 설정
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 🔸 7. 학습 수행
    trainer.train()

    # 🔸 8. 최종 저장
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)