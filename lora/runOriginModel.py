from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# 모델과 토크나이저 불러오기
model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# 질문 프롬프트 구성
# prompt = "신용카드를 분실했을 때 어떻게 해야 하나요?\n\n### 답변:"
prompt = "신용카드를 분실했을 때 어떻게 해야 하나요?"

# 토큰화
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

# GPU 사용 가능 시
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
input_ids = input_ids.to(device)

# 텍스트 생성
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
    )

# 결과 디코딩
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("🔍 질문:\n", prompt)
print("🧠 응답:\n", result)

# text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
# match = re.search(r"### 답변:(.*?)(###|\Z)", text, re.DOTALL)
# if match:
#     answer = match.group(1).strip()
#     print("🔍 질문:\n", prompt)
#     print("🧠 답변:\n", answer)
# else:
#     print("답변을 찾지 못했어요.")