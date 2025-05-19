from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델과 토크나이저 로드
model_path = "/Users/Shared/app/llm/lora/model"  # 실제 모델 파일 경로로 대체
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path)

def get_medical_advice(patient_query):
    # 프롬프트 설정
    prompt = f"{patient_query} 응답:"

    # 입력 토큰 생성
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    model1 = model.to("cpu")
    input_ids = input_ids.to("cpu")


    # 모델을 통해 입력 토큰 전달하여 출력 생성
    outputs = model1.generate(
        input_ids,
        attention_mask=inputs["attention_mask"],
        max_length=1024,  # 출력 길이를 늘림
        num_beams=5,
        temperature=0.7,  # 다양성을 위해 temperature 추가
        no_repeat_ngram_size=3,  # n-gram 반복 방지
        repetition_penalty=1.5,
        early_stopping=True
    )

    # 출력 토큰을 문자열로 변환
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 프롬프트 제거 후 의사의 응답만 반환
    #doctor_response = result.split("응답:")[-1].strip()
    doctor_response=result
    return doctor_response

# 사용 예시
patient_query = "신용카드를 잃어버리면 어떻게 하나요?"
response = get_medical_advice(patient_query)
print("응답:")
print(response)