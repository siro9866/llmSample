# 허깅페이스에서 데이타를 다운로드 받는다
import datasets
from datasets import load_dataset, load_from_disk
from config import DATASET_PATH, DATASET_MODEL_PATH, DATASET_MODEL_NAME
import os


dataPath = os.path.join(DATASET_PATH, DATASET_MODEL_PATH, DATASET_MODEL_NAME)
# 허깅페이스에서 데이타셋 다운로드
def save_dataset():
    print("datasets version:", datasets.__version__)
    print("Trying to load dataset...")
    dataset = load_dataset(f"{DATASET_MODEL_PATH}/{DATASET_MODEL_NAME}", split="train")

    # 데이타셋파일 저장
    # 폴더 없으면 생성 (중첩 경로 포함)
    print(f"데이타 저장될경로]{dataPath}")
    os.makedirs(dataPath, exist_ok=True)
    dataset.save_to_disk(dataPath)

    print(dataset[0])

def load_dataset():
    print(dataPath)
    return load_from_disk(dataPath)

# 텍스트 전처리 함수 정의
def preprocess_text(example):
    text = example["text"].strip().replace("\n", " ").replace("  ", " ")
    if not text.endswith("."):
        text += "."
    return {"text": text}

print(dataPath)
save_dataset()

