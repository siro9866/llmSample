# kogpt2-base-v2 모델로 성공
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


def get_advice(query):
    """
    파인튜닝된 모델을 사용하여 질의에 대한 응답 생성

    Args:
        query (str): 사용자 질의

    Returns:
        str: 모델이 생성한 응답
    """
    # 모델 경로 설정
    # model_path = "/Users/Shared/app/llm/model/origin/kogpt2-base-v2" # 튜닝전
    model_path = "/Users/Shared/app/llm/model/finetune/kogpt2-base-v2" # 튜닝후

    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # PEFT/LoRA 설정 확인
        try:
            # LoRA 모델인 경우
            config = PeftConfig.from_pretrained(model_path)
            print(f"LoRA 모델 감지됨. 기본 모델: {config.base_model_name_or_path}")

            # 기본 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                device_map="cpu",  # 명시적으로 CPU 사용
                torch_dtype=torch.float32
            )

            # LoRA 어댑터 적용
            model = PeftModel.from_pretrained(base_model, model_path)

        except Exception as lora_error:
            print(f"LoRA 모델 로드 실패, 일반 모델로 시도: {lora_error}")
            # 일반 모델인 경우 직접 로드
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu",  # 명시적으로 CPU 사용
                torch_dtype=torch.float32
            )

        model.eval()  # 평가 모드로 설정
        print("모델 로드 완료")

        # 프롬프트 설정
        prompt = f"{query}\n응답:"
        print(f"프롬프트: {prompt}")

        # 입력 토큰화
        inputs = tokenizer(prompt, return_tensors="pt")

        # 모든 텐서를 CPU에 배치
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].to("cpu")

        print("텐서 준비 완료")

        # 텍스트 생성
        with torch.no_grad():  # 그래디언트 계산 비활성화
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=200,  # 적절한 길이로 조정
                num_beams=2,  # 빔 검색 크기 조정
                temperature=0.7,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 출력 디코딩
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 응답 부분만 추출 (필요시)
        if "응답:" in result:
            response = result.split("응답:")[-1].strip()
        else:
            response = result

        return response

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return f"모델 실행 중 오류가 발생했습니다: {str(e)}"


# 테스트 함수
def test_model():
    test_queries = [
        "Since when has Seoul, the capital of Korea, been the capital?",
        "기업의 경영권을 헌법상의 권리로 볼 수 있는가?",
        "직권남용죄에서 말하는 '의무'에 심리적 의무감 또는 도덕적 의무가 해당되는가?",
        "형사소송법 제212조 제2항이 뭐지",
        "음주운전으로 사고를 내면 어떤 법의 근거로 처벌을 받나"
    ]

    print("===== 모델 테스트 시작 =====\n")
    for i, query in enumerate(test_queries):
        print(f"[테스트 {i + 1}] 질문: {query}")
        response = get_advice(query)
        print(f"응답: {response}\n")

    print("===== 모델 테스트 완료 =====")


# 대화형 테스트 함수
def interactive_test():
    print("\n===== 대화형 테스트 시작 =====")
    print("종료하려면 'exit' 또는 'quit'을 입력하세요.")

    while True:
        user_input = input("\n질문을 입력하세요: ")
        if user_input.lower() in ['exit', 'quit']:
            print("대화형 테스트를 종료합니다.")
            break

        response = get_advice(user_input)
        print(f"응답: {response}")


if __name__ == "__main__":
    # 테스트 실행
    test_model()

    # 대화형 테스트 (원하는 경우 주석 해제)
    # interactive_test()