#!/bin/bash

MODELS=(
  # "openai-community/gpt2"
  # "mistralai/Mistral-7B-Instruct-v0.2"
  # "mistralai/Mixtral-8x7B-Instruct-v0.1"
  # "meta-llama/Llama-2-7b-chat-hf"
  # "meta-llama/Llama-3.1-8b-Instruct"
  # "google/gemma-7b-it"
  # "01-ai/Yi-6B-Chat"
  # "HuggingFaceH4/zephyr-7b-beta"
  # "Qwen/Qwen-7B-Chat"
  # "Qwen/Qwen2-7B-Instruct"
  # "internlm/internlm2_5-7b-chat"
  # "stabilityai/StableBeluga-7B"
  # "openchat/openchat-3.5-0106"
  # "NousResearch/Nous-Hermes-2-Yi-34B"
  # "openchat/openchat_3.5"
  # "TheBloke/openchat_3.5-GPTQ"
  # "openai-community/gpt2"
  # "openai-community/gpt2-medium"
  # "openai-community/gpt2-large"
  # "meta-llama/Llama-2-13b-hf"
  # "meta-llama/Llama-2-13b-chat"
  # "meta-llama/Llama-2-13b-chat-hf"
  # "meta-llama/Meta-Llama-3-8B"
  # "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
  "lmsys/vicuna-13b-v1.5"
  "Expert68/llama2_13b_instructed_version2"
  "mosaicml/mpt-7b-instruct"
)

PROMPTS=(
  "prompt1.txt"
  "prompt2.txt"
  "prompt3.txt"
  "prompt4.txt"
)

mkdir -p logs reports

# Set CUDA environment variables to help with errors
export CUDA_LAUNCH_BLOCKING=1

for model in "${MODELS[@]}"; do
  mname=$(echo "$model" | tr '/' '_')
  for prompt in "${PROMPTS[@]}"; do
    pname=$(basename "$prompt" .txt)
    OUTPUT="bash/${mname}/records_${pname}/cirrhosis_records.csv"

    if [[ -f "$OUTPUT" ]]; then
      echo "✅ Skipping $mname × $pname (already exists)"
      continue
    fi

    echo "🕒 Starting $mname × $pname"
    start_time=$(date +%s)

    # Special handling for GPT-2 model
    if [[ "$model" == *"gpt2"* ]]; then
      BATCH_SIZE="--batch_size 5"
      TIMEOUT="55m"
    else
      BATCH_SIZE=""
      TIMEOUT="50m"
    fi

    timeout $TIMEOUT \
      python cirrhosis_generator.py \
        --model_name "$model" \
        --prompt_file "prompts/$prompt" \
        $BATCH_SIZE \
        2>&1 | tee "logs/${mname}_${pname}.log"
        # > "logs/${mname}_${pname}.log" 2>&1

    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "$mname,$pname,$duration" >> reports/timing.csv

    if [[ $exit_code -ne 0 ]]; then
      echo "⛔ Timeout or error for $mname × $pname"
      echo "{\"model\": \"$model\", \"prompt\": \"$prompt\", \"error\": \"timeout or crash\"}" >> reports/errors.jsonl
    else
      echo "✅ Successfully completed $mname × $pname in $duration seconds"
    fi
    
    # Give the system a moment to free resources
    sleep 5
  done
done

echo "All model runs complete!"