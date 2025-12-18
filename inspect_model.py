from huggingface_hub import snapshot_download
import os

try:
    print("Downloading model to inspect...")
    path = snapshot_download(repo_id="tencent/HunyuanOCR", allow_patterns=["*.json", "*.py"])
    print(f"Model downloaded to: {path}")
    print("Files:")
    for f in os.listdir(path):
        print(f)
except Exception as e:
    print(f"Error: {e}")
