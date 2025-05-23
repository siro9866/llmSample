from transformers import AutoModelForCausalLM, AutoTokenizer
from com.env import MODEL_ORIGIN_PATH
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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

# gpt2 모델 저장하기
def saveGpt2Model(model_name, model_path):
    os.makedirs(model_path, exist_ok=True)

    # 모델 및 토크나이저 로드
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

# 자동 실행 구문odelForCausalLM.from_pretrained(model_path)
#     return tokenizer, model
if __name__ == "__main__":

    # gpt2 모델 저장
    # gpt_model_name = "gpt2"  # gpt2-medium, gpt2-large, gpt2-xl
    # gpt_model_path = os.path.join(MODEL_ORIGIN_PATH, gpt_model_name)
    # saveGpt2Model(gpt_model_name, gpt_model_path)


    # model_name = "skt/kogpt2-base-v2"
    model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    model_path = os.path.join(MODEL_ORIGIN_PATH, model_name)
    print(model_path)

    saveHuggingfaceModel(model_name, model_path)
    #loadModel(model_path)