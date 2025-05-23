from transformers import AutoModelForCausalLM, AutoTokenizer
from com.env import MODEL_ORIGIN_PATH
import os

# 허깅 페이스 모델을 로컬에 저장
def saveHuggingfaceModel(model_name, model_path):
    os.makedirs(model_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

# 저장된 모델 불러오기
def loadModel(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

# 자동 실행 구문odelForCausalLM.from_pretrained(model_path)
#     return tokenizer, model
if __name__ == "__main__":

    model_name = "skt/kogpt2-base-v2"
    #model_path = "/Users/Shared/app/llm/model/origin/kogpt2-base-v2"
    model_path = os.path.join(MODEL_ORIGIN_PATH, model_name)
    print(model_path)

    #saveHuggingfaceModel(model_name, model_path)
    #loadModel(model_path)