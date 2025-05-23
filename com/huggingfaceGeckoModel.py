from com.env import MODEL_ORIGIN_PATH
import os
from huggingface_hub import snapshot_download

model_name = "NeuralNovel/Gecko-7B-v0.1-DPO"
model_path = os.path.join(MODEL_ORIGIN_PATH, model_name)
os.makedirs(model_path, exist_ok=True)
snapshot_download(
    repo_id=model_name,
    local_dir=model_path,
    local_dir_use_symlinks=False
)