#!/bin/bash

MODELS=(
  # "gpt2"
  "mistralai/Mistral-7B-Instruct-v0.2"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "meta-llama/Llama-2-7b-chat-hf"
  "meta-llama/Llama-3-8b-Instruct"
  "google/gemma-7b-it"
  "01-ai/Yi-6B-Chat"
  "HuggingFaceH4/zephyr-7b-beta"
  # "microsoft/Phi-3-mini-4k-instruct"
  "BAAI/Qwen-7B-Chat"
  "Qwen/Qwen2-7B-Instruct"
  "internlm/internlm2-7b-chat"
  "stabilityai/StableBeluga-7B"
  "openchat/openchat-3.5-0106"
  "NousResearch/Nous-Hermes-2-Yi-34B"
  "mosaicml/mpt-7b-instruct"
  "openchat/openchat-3.5"
  "TheBloke/openchat_3.5-GPTQ"
  "lmstudio-ai/Meta-Llama-3-8B-Instruct-GGUF"
)

PROMPTS=(
  "prompt1.txt"
  "prompt2.txt"
  "prompt3.txt"
  "prompt4.txt"
)

mkdir -p logs reports

for model in "${MODELS[@]}"; do
  mname=$(echo "$model" | tr '/' '_')
  for prompt in "${PROMPTS[@]}"; do
    pname=$(basename "$prompt" .txt)
    OUTPUT="bash/${mname}/records_${pname}/diabetes_records.csv"

    if [[ -f "$OUTPUT" ]]; then
      echo "✅ Skipping $mname × $pname (already exists)"
      continue
    fi

    echo "🕒 Starting $mname × $pname"
    start_time=$(date +%s)

    timeout 30m \
      python solo-prompt.py \
        --model_name "$model" \
        --prompt_file "prompts/$prompt" \
        > "logs/${mname}_${pname}.log" 2>&1

    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "$mname,$pname,$duration" >> reports/timing.csv

    if [[ $exit_code -ne 0 ]]; then
      echo "⛔ Timeout or error for $mname × $pname"
      echo "{\"model\": \"$model\", \"prompt\": \"$prompt\", \"error\": \"timeout or crash\"}" >> reports/errors.jsonl
    fi
  done
done