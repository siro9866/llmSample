# Lora 파인튜닝
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import torch

# 데이터 불러오기 (경로는 사용할 데이터 사용법을 보면 친절하게 알려줌)
# train, dev 데이터로 나눠준다. BCCard/BCAI-Finance-Kor 는 train만 있음
dataset = DatasetDict({
    "train": load_dataset("BCCard/BCAI-Finance-Kor", split="train")
})

# 모델 로드 및 LoRA 설정
# model 에서 gemma-2-2b-it 을 사용하는 경우 attn_implementation='eagert' 속성으로 사용해야함.
# 만약 target_modules 가 모델마다 다르기 때문에 꼭 확인 필요. 에러가 생긴다면
model_path = "/home/llama/fine_tunning/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation='eager')

# 모델 구조 확인 - target_modules 를 모를시 사용
# for name, module in model.named_modules():
#    print(name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)


# 데이타 전처리
# 모델에 학습 시키기 전 데이터 구조에 맞게 전처리를 해준다.
# Data Collator는 데이터 로딩 중 배치를 생성할 때 필요한 전처리 작업을 해줌. 사용하는 이유는 정확성을 올려줌. 자세한 내용은 gpt에게 물어보는게 좋다.
def preprocess_function(examples):
    inputs = examples['startphrase']

    # 정답 레이블에 해당하는 ending을 선택합니다.
    labels = [examples[f'ending{label}'][i] for i, label in enumerate(examples['label'])]

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

    # labels를 토큰화합니다.
    label_tokens = tokenizer(labels, padding="max_length", truncation=True, max_length=512)

    # labels를 input_ids로 설정합니다.
    model_inputs["labels"] = label_tokens["input_ids"]

    # 패딩 토큰을 -100으로 대체합니다.
    for i, label in enumerate(model_inputs["labels"]):
        model_inputs["labels"][i] = [l if l != tokenizer.pad_token_id else -100 for l in label]

    return model_inputs


tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Data Collator 설정
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100
)

# 학습설정
# 학습된 데이터는 지정해준 폴더에 생성이됨.
# 데이터 학습 시키는데 대략 40분 정도 걸린듯함. (gpu - Nvidia 3070ti 8gb)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./fine_tuned_model_medical")
tokenizer.save_pretrained("./fine_tuned_model_medical")