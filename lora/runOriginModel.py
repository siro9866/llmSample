from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re


def runModel(modelPath, prompt):
    tokenizer = AutoTokenizer.from_pretrained(modelPath, device_map="auto", quantization_config=None)
    model = AutoModelForCausalLM.from_pretrained(modelPath, device_map="auto", quantization_config=None)
    model.eval()

    # 토큰화
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # GPU 사용 가능 시
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    input_ids = input_ids.to(device)

    # 텍스트 생성
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
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


from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
def modelTest(modelPath):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    model = AutoModelForCausalLM.from_pretrained(modelPath, torch_dtype=torch.float16)

    input = tokenizer("비씨카드는 어떤 은행이 있어?", return_tensors="pt")
    output = model.generate(**input, max_length=100, do_sample=True, top_k=40, top_p=0.9, temperature=0.7)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    # 모델과 토크나이저 불러오기
    # modelPath = "/Users/Shared/app/llm/huggingface/kogpt2-base-v2"
    modelPath = "/Users/Shared/app/llm/lora/model"

    # 질문 프롬프트 구성
    # prompt = "신용카드를 분실했을 때 어떻게 해야 하나요?\n\n### 답변:"
    prompt = """
    너는 비씨카드 관련 모든 정보를 알고있어서 문맥에 맞게 친절하게 설명해주는 상담사야.
    
    질문: 비씨카드중 대출 이율이 가장 높은 상품은 뭐야?
     
    응답:
    """

    runModel(modelPath, prompt)
    # modelTest(modelPath)