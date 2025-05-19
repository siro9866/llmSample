# 허깅페이스에서 데이타를 다운로드 받는다
import datasets
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import os
from functools import partial

# 허깅페이스에서 데이타셋 다운로드
def save_dataset(data_model_name, data_origin_path):
    print("datasets version:", datasets.__version__)
    print("Trying to load dataset...")

    dataset = load_dataset(data_model_name, split="train")

    dataPath = data_origin_path
    # 데이타셋파일 저장
    # 폴더 없으면 생성 (중첩 경로 포함)
    print(f"데이타 저장될경로]{dataPath}")
    os.makedirs(dataPath, exist_ok=True)
    dataset.save_to_disk(dataPath)

    print(dataset[0])

# 로컬의 데이타셋 불러오기
def getDataset(data_origin_path):
    print(data_origin_path)
    return load_from_disk(data_origin_path)

# 텍스트 전처리 함수 정의
def preprocess_text(example):
    text = example["text"].strip().replace("\n", " ").replace("  ", " ")
    if not text.endswith("."):
        text += "."
    return {"text": text}

# 전처리 함수
# 여기 계속 변경해야함
def preprocess(example, tokenizer):
    prompt = f"질문: {example['anchor']}\n답변: {example['positive']}\n<|endoftext|>"
    tokenized = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 전처리 데이타로 변환
# save_dataMap 내부에서 tokenizer를 한 번만 로드하고 map에 전달
def save_dataMap(data_origin_path, data_path):
    from datasets import load_from_disk  # 혹은 getDataset 정의에 따라 다르게

    # 1. tokenizer 한번만 로드
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. dataset 불러오기
    dataset = getDataset(data_origin_path)
    print(dataset.column_names)

    # 3. partial을 사용해서 tokenizer를 preprocess에 고정 인자로 전달
    preprocess_fn = partial(preprocess, tokenizer=tokenizer)

    # 4. 병렬 map 적용
    processed_dataset = dataset.map(
        preprocess_fn,
        remove_columns=dataset.column_names,
        num_proc=4,  # 코어 수에 따라 조절
        desc="Tokenizing dataset"
    )

    # 5. 저장
    print(f"데이타 저장될경로] {data_path}")
    os.makedirs(data_path, exist_ok=True)
    processed_dataset.save_to_disk(data_path)

def getDataMap(data_path):
    print(data_path)
    return load_from_disk(data_path)

if __name__ == "__main__":
    # 판례
    data_model_name = "kakao1513/AI_HUB_legal_QA_data"
    data_origin_path = "/Users/Shared/app/llm/datasets/origin/AI_HUB_legal_QA_data"
    data_path = "/Users/Shared/app/llm/datasets/preprocess/AI_HUB_legal_QA_data"

    #save_dataset(data_model_name, data_origin_path)
    save_dataMap(data_origin_path, data_path)

