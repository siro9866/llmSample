from dataset import load_dataset, preprocess_text
from train_model import train_model

if __name__ == "__main__":
    dataset = load_dataset()    # 데이타로드
    processed_dataset = dataset.map(preprocess_text)
    print(dataset[0])
    print("로데이타0번째 ====================================================")
    print("전처리후 ====================================================")
    print(processed_dataset[0]["text"])
    #train_model(dataset)