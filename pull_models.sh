#!/bin/bash

models=(
  "deepseek-r1:1.5b"
  "gemma3:4b"
  "gemma3:270m"
  "qwen3:4b"
  "qwen3:0.6b"
  "llama3.2:3b"
  "phi3:3.8b"
)

"""
#Install ollama
curl -fsSL https://ollama.com/install.sh | sh
"""

#Install models
for model in "${models[@]}"; do
  echo "------------------------------------------"
  echo "Pulling model: $model"
  echo "------------------------------------------"
  ollama pull "$model"
done

echo "Finished pulling all models."
