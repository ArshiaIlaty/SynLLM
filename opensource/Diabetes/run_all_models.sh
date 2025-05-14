#!/bin/bash
set -euo pipefail

MODELS=(
  # "meta-llama/Llama-2-13b"
  # "meta-llama/Llama-2-13b-hf"
  # "meta-llama/Llama-2-13b-chat"
  # "meta-llama/Llama-2-13b-chat-hf"
  # "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
  # "meta-llama/Meta-Llama-3-8B"
  # "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
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

    timeout 60m \
      python solo-prompt.py \
        --model_name "$model" \
        --prompt_file "prompts/$prompt" \
        2>&1 | tee "logs/${mname}_${pname}.log"
        # > "logs/${mname}_${pname}.log" 2>&1

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

# chmod +x run_all_models.sh
#./run_all.sh