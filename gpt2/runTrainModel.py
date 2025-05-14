from datasetMake import getDataset, getDataMap
from trainModel import train_model

if __name__ == "__main__":
    dataset = getDataMap()
    print("로데이타0번째 ====================================================")
    print(dataset[0])
    train_model(dataset)