from datasetMake import getDataset, getDataMap
from trainModel import train_model
from datetime import datetime

if __name__ == "__main__":
    now = datetime.now()
    print(f"학습시작: {now}")
    dataset = getDataMap()
    print(dataset[0])
    train_model(dataset)
    now = datetime.now()
    print(f"학습종료: {now}")