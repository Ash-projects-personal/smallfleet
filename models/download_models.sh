#!/bin/sh
# Download the SmallFleet demo models (GGUF files from Hugging Face).
# The model files are too large for GitHub, so they are not in the repo.
# Run this once before the demo:  sh models/download_models.sh
# Or the Python equivalent:  .venv/bin/python -m smallfleet.download_models
set -e
cd "$(dirname "$0")/.."
mkdir -p models
dl() {
  url="$1"
  dest="$2"
  if [ -f "$dest" ]; then
    echo "already present: $dest"
    return 0
  fi
  echo "downloading $url -> $dest"
  curl -L --fail -o "$dest" "$url"
  echo "done: $dest"
}
dl "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
dl "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" "models/qwen2.5-3b-instruct-q4_k_m.gguf"
echo "models ready."
