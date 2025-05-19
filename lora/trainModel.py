# lora를 활용해 파인튜닝
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# 모델과 토크나이저
model_path = "/Users/Shared/app/llm/model/origin/kogpt2-base-v2"
data_path = "/Users/Shared/app/llm/datasets/preprocess/AI_HUB_legal_QA_data"
finetune_model_path = "/Users/Shared/app/llm/model/finetune"

# 2. 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # 💡 핵심 설정
model = AutoModelForCausalLM.from_pretrained(model_path)

# 3. 데이터셋 로드 및 구조 확인
dataset = load_from_disk(data_path)
print("✅ 데이터 샘플 구조:", dataset[0].keys())  # 'input_ids', 'attention_mask', 'labels' 확인

# 4. 모델 구조 출력 (LoRA 타깃 모듈 확인용)
print("✅ 첫 번째 attention 모듈 구조:")
print(model.transformer.h[0].attn)

# 5. LoRA 구성
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # 또는 ['c_attn'] → 구조 보고 맞춰 조정  "q_proj", "v_proj"
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir=finetune_model_path,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
    fp16=False  # ⚠️ MPS 사용시 반드시 False
)

# 7. 데이터 콜레이터
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 8. 트레이너 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)


# 학습 시작
trainer.train()

# 모델 저장
model.save_pretrained(finetune_model_path)
tokenizer.save_pretrained(finetune_model_path)