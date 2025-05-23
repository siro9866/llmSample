class ModelManager:
    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def load_model(self, model_path):
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModelForCausalLM, LlamaTokenizer

            print("모델 로딩 중...")
            self._tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
            self._model = AutoModelForCausalLM.from_pretrained(model_path)
            print("모델 로딩 완료")

        return self._model, self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer


# 모델 매니저 인스턴스 생성
model_manager = ModelManager()

from com.env import MODEL_ORIGIN_PATH, MODEL_FINETUNE_PATH
import os
# 모델 경로 설정
model_name = "NeuralNovel/Gecko-7B-v0.1-DPO"
model_origin_path = os.path.join(MODEL_ORIGIN_PATH, model_name)
# model_finetune_path = os.path.join(MODEL_FINETUNE_PATH, model_name)

model_path = model_origin_path
# 처음 한 번만 모델 로드
model_manager.load_model(model_path)

# 이후 필요할 때마다 모델과 토크나이저 사용
model = model_manager.model
tokenizer = model_manager.tokenizer

# 질문 처리 예시
def process_query(query):
    # 이미 로드된 모델과 토크나이저 사용
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 테스트 함수
def test_model():
    test_queries = [
        "기업의 경영권을 헌법상의 권리로 볼 수 있는가?",
        "직권남용죄에서 말하는 '의무'에 심리적 의무감 또는 도덕적 의무가 해당되는가?",
        "형사소송법 제212조 제2항이 뭐지",
        "음주운전으로 사고를 내면 어떤 법의 근거로 처벌을 받나"
    ]

    print("===== 모델 테스트 시작 =====\n")
    for i, query in enumerate(test_queries):
        print(f"[테스트 {i + 1}] 질문: {query}")
        response = process_query(query)
        print(f"응답: {response}\n")

    print("===== 모델 테스트 완료 =====")
if __name__ == "__main__":
    # 테스트 실행
    test_model()

    # 대화형 테스트 (원하는 경우 주석 해제)
    # interactive_test()