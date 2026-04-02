import os
import sys
from huggingface_hub import snapshot_download

# Path to the EngineManager to get the AVAILABLE_MODELS registry
# We can just define them here to avoid import issues if the venv isn't set up perfectly
MODELS = {
    "qwen-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "qwen-1.7b-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "qwen-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

CORE_MODEL = "qwen-1.7b"

def download_all():
    print("=" * 60)
    print("  Resound Studio - Model Downloader")
    print("=" * 60)
    print("\nEnsuring all required models are downloaded to the HF cache...")
    print(f"Primary Model: {MODELS[CORE_MODEL]} (1.7B Base)\n")

    for model_id, hf_repo in MODELS.items():
        is_core = " (CORE)" if model_id == CORE_MODEL else ""
        print(f"[*] Checking/Downloading{is_core}: {hf_repo} ({model_id})")
        try:
            snapshot_download(repo_id=hf_repo)
            print(f"    [OK] {model_id} is ready.\n")
        except Exception as e:
            print(f"    [ERROR] Failed to download {model_id}: {e}\n")

    print("=" * 60)
    print("  ALL MODELS READY")
    print("  You can now use 'run.bat' or 'setup_docker.bat'")
    print("=" * 60)

if __name__ == "__main__":
    download_all()
