# kogpt2-base-v2 모델로 성공
import os
# lora를 활용해 파인튜닝
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from multiprocessing import freeze_support

def main():
    # 경로 설정
    model_path = "/Users/Shared/app/llm/model/origin/kogpt2-base-v2"
    data_path = "/Users/Shared/app/llm/datasets/preprocess/AI_HUB_legal_QA_data"
    finetune_model_path = "/Users/Shared/app/llm/model/finetune/kogpt2-base-v2"
    
    # 토크나이저 및 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 맥에서 GPU 없이 학습하기 위한 설정
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto"  # CPU에서는 자동으로 적절한 매핑
    )
    
    # 데이터셋 로드
    dataset = load_from_disk(data_path)
    dataset = dataset.with_format("torch")
    
    # LoRA 설정 - 수정된 설정
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],  # 모델 구조에 맞게 조정 필요
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    # 학습 설정 - CPU에 최적화
    training_args = TrainingArguments(
        output_dir=finetune_model_path,
        per_device_train_batch_size=2,  # CPU에서는 작은 배치 크기 사용
        gradient_accumulation_steps=4,  # 누적으로 효과적인 배치 크기 늘리기
        num_train_epochs=3,
        logging_dir=os.path.join(finetune_model_path, "logs"),
        save_steps=100,
        save_total_limit=2,
        fp16=False,  # CPU에서는 fp16 비활성화
        bf16=False,  # CPU에서는 bf16 비활성화
        optim="adamw_torch",  # 일반 AdamW 사용
        learning_rate=2e-4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        dataloader_num_workers=0,  # CPU에서 안전하게 0으로 설정
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 트레이너 구성 (향후 버전 대비 processing_class 추가)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        # tokenizer=tokenizer,  # 곧 deprecated 될 예정
    )
    
    # 학습 시작
    trainer.train()
    
    # 모델 저장
    model.save_pretrained(finetune_model_path)
    tokenizer.save_pretrained(finetune_model_path)
    print(f"모델과 토크나이저가 {finetune_model_path}에 저장되었습니다.")

if __name__ == "__main__":
    freeze_support()  # Windows/Mac 멀티프로세싱 지원
    main()