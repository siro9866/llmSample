# 실행방법
1. 데이타파일 로컬에 저장
    - dataset.py save_dataset() 실행 하면 env 참고하여 해당 경로에 허깅페이스에서 다운받은 자료 생성
2. 모델 훈련
    - run_train.py 실행
3. 질의 실행
    - run_model.py 실행

> 훈련 조건
>- 추가 학습 가능
>- gpu 가속 가능 하면 gpu로 학습
>- 학습데이타는 로컬에 있는 json 파일이다.
>- 

> pip install
>- pip install python-dotenv   환경변수 사용
>- pip install datasets        허깅페이스 데이타셋
>- pip install transformers
>- pip install torch