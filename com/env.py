from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

MODEL_ORIGIN_PATH = os.environ.get('MODEL_ORIGIN_PATH')
MODEL_FINETUNE_PATH = os.environ.get('MODEL_FINETUNE_PATH')
DATA_ORIGIN_PATH = os.environ.get('DATA_ORIGIN_PATH')
DATA_PREPROCESS_PATH = os.environ.get('DATA_PREPROCESS_PATH')
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH')

