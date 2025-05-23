# lora 사용해서 파인튜닝
모델: skt/kogpt2-base-v2
데이타셋: BCCard/BCAI-Finance-Kor

> pip install
>- pip install transformers
>- pip install torch
>- pip install datasets
>- pip install peft
>- pip install accelerate
>- pip install bitsandbytes
>- pip install trl
>- pip install protobuf



epoch:3
100%|██████████| 715392/715392 [7:38:51<00:00, 25.98it/s]
{'train_runtime': 27531.7144, 'train_samples_per_second': 103.937, 'train_steps_per_second': 25.984, 'train_loss': 5.369365833082488, 'epoch': 3.0}


# 성공
방식: 로라 파인튜닝
기본 모델: /Users/Shared/app/llm/model/origin/kogpt2-base-v2: 125M
데이타: AI_HUB_legal_QA_data: 각종현안 50MB, 70,765
완료모델: /Users/Shared/app/llm/model/finetune/kogpt2-base-v2
