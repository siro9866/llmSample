# lora를 활용해 파인튜닝
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# 모델 및 토크나이저 로드
model_id = "skt/kogpt2-base-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
# tokenizer = AutoTokenizer.from_pretrained(model_id, force_download=True)
# tokenizer.pad_token = tokenizer.eos_token  # padding 설정
model = AutoModelForCausalLM.from_pretrained(model_id)

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # GPT-2 구조에서 key module
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# 데이터셋 로드
dataset = load_dataset("BCCard/BCAI-Finance-Kor", split="train[:2000]")  # 빠른 실험용

# 프롬프트 구성 함수
def format_prompt(example):
    return {
        "text": f"### 질문: {example['instruction']}\n\n### 답변: {example['output']}"
    }

dataset = dataset.map(format_prompt)

# 훈련 설정
training_args = TrainingArguments(
    output_dir="./kogpt2-lora-finance",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    fp16=True,
    save_total_limit=1,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    # train_dataset=dataset,
    # dataset_text_field="text",
    # args=training_args,
    # max_seq_length=512,
    args=training_args,
    train_dataset=dataset,  # tokenizer 적용된 Dataset
    max_seq_length=512,
)

trainer.train()
model.save_pretrained("./kogpt2-lora-finance")

