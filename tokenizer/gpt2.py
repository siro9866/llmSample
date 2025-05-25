import os
import json
import string
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# 파일 경로 설정
training_data_path = r'C:\Users\sil\Downloads\tokenizer\TS_1'
output_path = r'C:\Users\sil\Downloads\tokenizer\gpt2Token01'
base_model_name = "gpt2"  # 베이스 모델로 gpt2 사용

# 데이터 로드 함수
def load_corpus_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:  # encoding을 'utf-8-sig'로 변경
        data = json.load(f)

    # "corpus" 필드에서 텍스트 추출
    corpus_texts = []
    for item in data.get('data', []):
        if 'corpus' in item:
            corpus_texts.append(item['corpus'])

    return corpus_texts

# 모든 JSON 파일에서 코퍼스 텍스트 수집
def collect_all_corpus_texts(directory):
    all_texts = []

    # 디렉토리 내의 모든 JSON 파일 처리
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            texts = load_corpus_from_json(file_path)
            all_texts.extend(texts)

    return all_texts

# 텍스트 파일로 저장 (토크나이저 학습용)
def save_texts_to_file(texts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

# 한글에 최적화된 GPT-2 토크나이저 학습
def train_korean_tokenizer(texts, base_model=base_model_name, vocab_size=50000):
    # 기본 GPT-2 토크나이저 가져오기
    print(f"베이스 모델 '{base_model}'에서 토크나이저 로드 중...")

    # 기존 GPT-2 토크나이저의 토크나이저 객체 가져오기
    base_tokenizer = GPT2TokenizerFast.from_pretrained(base_model)
    tokenizer_json = base_tokenizer.backend_tokenizer.to_str()

    # 토크나이저 객체 생성
    tokenizer = Tokenizer.from_str(tokenizer_json)

    # 한글에 적합한 pre-tokenizer 설정 (기존 설정 유지)
    # tokenizer.pre_tokenizer는 이미 ByteLevel로 설정되어 있음

    # 영어 + 숫자 + 한글 일부를 포함하는 기본 알파벳 예시
    alphabet = list(string.ascii_letters + string.digits) + list("가나다라마바사아자차카타파하")

    # 트레이너 설정 - 기존 어휘 유지하면서 한글 추가
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        initial_alphabet=alphabet,
        show_progress=True
    )

    # 임시 파일에 텍스트 저장
    temp_file = os.path.join(os.path.dirname(output_path), "temp_training_data.txt")
    save_texts_to_file(texts, temp_file)

    # 토크나이저 학습 (기존 어휘 기반으로 한글 추가)
    tokenizer.train([temp_file], trainer)

    # 임시 파일 삭제
    os.remove(temp_file)

    return tokenizer

# Hugging Face 형식의 토크나이저로 변환
def convert_to_huggingface_tokenizer(tokenizer, base_model=base_model_name):
    # Hugging Face 토크나이저로 변환
    hf_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)

    # 기본 GPT-2 토크나이저에서 설정 복사
    base_tokenizer = GPT2TokenizerFast.from_pretrained(base_model)

    # 필요한 특수 토큰 설정 (기본 GPT-2와 동일하게)
    hf_tokenizer.bos_token = base_tokenizer.bos_token
    hf_tokenizer.eos_token = base_tokenizer.eos_token
    hf_tokenizer.unk_token = base_tokenizer.unk_token
    hf_tokenizer.pad_token = base_tokenizer.eos_token  # GPT-2는 기본적으로 pad_token이 없으므로 eos_token 사용

    return hf_tokenizer

def main():
    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)

    print(f"학습 데이터 로드 중... ({training_data_path})")
    corpus_texts = collect_all_corpus_texts(training_data_path)

    if not corpus_texts:
        print("경고: 코퍼스 텍스트를 찾을 수 없습니다. JSON 파일 구조를 확인하세요.")
        return

    print(f"총 {len(corpus_texts)}개의 텍스트 항목을 로드했습니다.")
    print(f"첫 번째 텍스트 샘플 (일부): {corpus_texts[0][:100]}...")

    print(f"\n베이스 모델 '{base_model_name}'을 기반으로 한글에 최적화된 토크나이저 학습 중...")
    tokenizer = train_korean_tokenizer(corpus_texts, base_model=base_model_name)

    print("Hugging Face 토크나이저로 변환 중...")
    hf_tokenizer = convert_to_huggingface_tokenizer(tokenizer, base_model=base_model_name)

    print(f"토크나이저를 {output_path}에 저장 중...")
    hf_tokenizer.save_pretrained(output_path)

    print("토크나이저 학습 및 저장 완료!")

    # 토크나이저 테스트
    print("\n===== 토크나이저 테스트 =====")

    # 기본 GPT-2 토크나이저와 비교
    base_tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)

    test_texts = [
        "안녕하세요, 한글 토크나이저 테스트입니다.",
        "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다.",
        "GPT-2 모델을 한국어에 최적화했습니다.",
        "이 토크나이저는 한글과 영어를 모두 처리할 수 있습니다.",
        "Hello, this is a test of the Korean-optimized tokenizer."
    ]

    for i, test_text in enumerate(test_texts):
        print(f"\n[테스트 {i+1}] 텍스트: {test_text}")

        # 기본 GPT-2 토크나이저로 토큰화
        base_tokens = base_tokenizer.tokenize(test_text)
        print(f"기본 GPT-2 토큰화 결과: {base_tokens}")
        print(f"기본 GPT-2 토큰 수: {len(base_tokens)}")

        # 한글 최적화 토크나이저로 토큰화
        korean_tokens = hf_tokenizer.tokenize(test_text)
        print(f"한글 최적화 토큰화 결과: {korean_tokens}")
        print(f"한글 최적화 토큰 수: {len(korean_tokens)}")

    print("\n토크나이저가 성공적으로 생성되었습니다!")
    print(f"저장 경로: {output_path}")

if __name__ == "__main__":
    main()
