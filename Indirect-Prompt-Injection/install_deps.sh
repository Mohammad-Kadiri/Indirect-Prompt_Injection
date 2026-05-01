#!/bin/bash
# Install all dependencies into .env624
set -e

ENV_PIP="/users/student/idddp/mohammad.k/IE 624/.env624/bin/pip3.12"

echo "=== Installing project dependencies into .env624 ==="

$ENV_PIP install \
    transformers>=4.45.0 \
    bitsandbytes>=0.43.0 \
    accelerate>=0.34.0 \
    openai>=1.40.0 \
    "datasets>=2.20.0" \
    "scikit-learn>=1.5.0" \
    "huggingface_hub>=0.24.0" \
    sentencepiece>=0.2.0 \
    "protobuf>=4.25.0" \
    "peft>=0.12.0" \
    "scipy>=1.14.0"

echo "=== Installation complete ==="
$ENV_PIP list | grep -E "transformers|bitsandbytes|accelerate|openai|datasets|scikit|hugging"
