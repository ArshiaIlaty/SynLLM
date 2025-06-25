# Diabetes
import argparse
import gc
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available

# Use the GPU specified by CUDA_VISIBLE_DEVICES, or default to 2 if not set
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Add memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

# Configuration
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RECORDS = 100

# Chat style mappings
MODELS = (
    "openai-community/gpt2"
    "openai-community/gpt2-medium"
    "openai-community/gpt2-large"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-3.1-8b-Instruct"
    "google/gemma-7b-it"
    "01-ai/Yi-6B-Chat"
    "HuggingFaceH4/zephyr-7b-beta"
    "Qwen/Qwen-7B-Chat"
    "Qwen/Qwen2-7B-Instruct"
    "internlm/internlm2_5-7b-chat"
    "stabilityai/StableBeluga-7B"
    "openchat/openchat-3.5-0106"
    "NousResearch/Nous-Hermes-2-Yi-34B"
    "mosaicml/mpt-7b-instruct"
    "openchat/openchat_3.5"
    "TheBloke/openchat_3.5-GPTQ"
    "lmsys/vicuna-13b-v1.5"
    "Expert68/llama2_13b_instructed_version2"
    # "meta-llama/Llama-2-70b-chat-hf"
    "meta-llama/Llama-2-13b"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-2-13b-chat"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    "meta-llama/Meta-Llama-3-8B"
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    # "Xenova/gpt-3.5-turbo-16k"
)

CHAT_STYLE_MODELS = {
    "gpt2": "plain",
    "mistral": "chatml",
    "mixtral": "chatml",
    "llama2": "llama",
    "llama3": "llama",
    "gemma": "openai",
    "yi": "im_start",
    "zephyr": "zephyr",
    # "phi": "phi",
    "qwen": "qwen",
    "qwen2": "chatml",
    "internlm": "internlm",
    "beluga": "beluga",
    "openchat": "openchat",
    "nous": "nous",
    "mpt": "mpt",
}

SYSTEM_PROMPT = """You are a synthetic medical data generator. Generate realistic patient records for diabetes research."""

EXAMPLE_RECORDS = [
    "Female,45.2,1,0,never,28.5,6.2,140,0",
    "Male,62.7,1,1,former,32.1,7.1,185,1",
    "Female,38.9,0,0,current,24.3,5.8,130,0",
    "Male,70.0,1,1,current,30.2,8.0,210,1",
    "Female,29.4,0,0,never,23.0,5.1,110,0",
]


def detect_chat_style(model_name):
    for key, style in CHAT_STYLE_MODELS.items():
        if key in model_name.lower():
            return style
    return "plain"


def build_prompt(model_name, system_prompt, user_prompt, tokenizer):
    style = detect_chat_style(model_name)

    if style == "llama":
        return tokenizer(
            f"<s>[INST] {system_prompt}\n{user_prompt} [/INST]", return_tensors="pt"
        )
    elif style == "chatml":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True)
    elif style == "openai":
        messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")
    elif style in ["im_start", "internlm", "nous"]:
        prompt_str = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer(prompt_str, return_tensors="pt")
    elif style in ["zephyr", "qwen", "qwen2"]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")
    elif style == "phi":
        return tokenizer(
            f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n",
            return_tensors="pt",
        )
    elif style == "openchat":
        return tokenizer(
            f"GPT4 System: {system_prompt}\nHuman: {user_prompt}\nAssistant:",
            return_tensors="pt",
        )
    elif style == "mpt":
        return tokenizer(
            f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:",
            return_tensors="pt",
        )
    else:
        return tokenizer(f"{system_prompt}\n\n{user_prompt}", return_tensors="pt")


def extract_records(text):
    print("=== DEBUG: GENERATED TEXT ===")
    print(text[:500])
    print("...\n=== END DEBUG ===")

    VALID_SMOKING = {"never", "former", "current", "not current", "no info", "unknown"}
    yes_no_map = {"yes": "1", "no": "0"}

    records = []
    skipped = 0
    bad_lines = []

    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+\.\s*", "", line)
        if not line or line.count(",") != 8:
            bad_lines.append(line)
            skipped += 1
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 9:
            bad_lines.append(line)
            skipped += 1
            continue

        gender, age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes = parts

        if gender.lower() not in ["male", "female"]:
            bad_lines.append(line)
            skipped += 1
            continue

        smoking = smoking.lower()
        if smoking not in VALID_SMOKING:
            bad_lines.append(line)
            smoking = "unknown"

        hyp = yes_no_map.get(hyp.lower(), hyp)
        heart = yes_no_map.get(heart.lower(), heart)
        diabetes = yes_no_map.get(diabetes.lower(), diabetes)

        # Create a new list with the corrected values
        record = [
            gender.title(),
            age,
            hyp,
            heart,
            smoking,
            bmi,
            hba1c,
            glucose,
            diabetes,
        ]
        records.append(record)

    if bad_lines:
        Path("reports").mkdir(parents=True, exist_ok=True)
        with open("reports/rejected_records.txt", "a") as bad:
            bad.write("\n".join(bad_lines) + "\n")

    print(f"[INFO] Extracted {len(records)} valid records, skipped {skipped}")
    return records


def log_system_metrics(model_name, prompt_name, start_time, end_time):
    gpu_info_before = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        .decode()
        .strip()
    )

    gpu_info_after = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        .decode()
        .strip()
    )

    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    report = {
        "model": model_name,
        "prompt": prompt_name,
        "duration_sec": end_time - start_time,
        "cpu_percent": cpu,
        "ram_percent": ram,
        "gpu_mem_start_MB": gpu_info_before,
        "gpu_mem_end_MB": gpu_info_after,
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/usage.jsonl", "a") as f:
        f.write(json.dumps(report) + "\n")


def log_generation_error(model_name, prompt_name, error):
    error_log = {"model": model_name, "prompt": prompt_name, "error": str(error)}
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/errors.jsonl", "a") as errfile:
        errfile.write(json.dumps(error_log) + "\n")


def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def generate_synthetic_data(prompt: str, model_name: str) -> pd.DataFrame:
    """Generate synthetic data using the specified model and prompt."""
    print(f"Loading model {model_name} on {DEVICE}")

    # Clear GPU memory before loading
    clear_gpu_memory()

    # Determine attention implementation
    attn_backend = "eager"  # Force eager implementation for all models

    # Start building kwargs
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": {
            "": "cuda:0"
        },  # Explicitly use the first visible GPU (which will be GPU 2 due to CUDA_VISIBLE_DEVICES)
        "trust_remote_code": True,
        "attn_implementation": attn_backend,  # Always use eager implementation
        "low_cpu_mem_usage": True,
    }

    # Always use 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Handle Llama model specific configurations
        if "llama" in model_name.lower():
            # Set default RoPE type if not present
            if not hasattr(config, "rope_type") or config.rope_type is None:
                config.rope_type = "dynamic"
            if not hasattr(config, "rope_scaling") or config.rope_scaling is None:
                config.rope_scaling = {"type": "dynamic", "factor": 8.0}
            # Ensure other required Llama configs are set
            if not hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = 4096
            if not hasattr(config, "hidden_size"):
                config.hidden_size = 4096

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            **model_kwargs,
        )
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

    try:
        from transformers.models.gpt_neox.tokenization_gpt_neox import GPTNeoXTokenizer
    except ImportError:
        print(
            "Could not import GPTNeoXTokenizer explicitly, falling back to AutoTokenizer."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_records = []
    batch_size = 20
    start_time = time.time()

    print("Starting generation...")

    while len(all_records) < NUM_RECORDS:
        needed = min(batch_size, NUM_RECORDS - len(all_records))
        print(
            f"Generating batch of {needed} records... ({len(all_records)}/{NUM_RECORDS} total)"
        )

        torch.cuda.empty_cache()
        gc.collect()

        chat_input = build_prompt(model_name, SYSTEM_PROMPT, prompt, tokenizer).to(
            DEVICE
        )

        try:
            max_context = tokenizer.model_max_length
            max_tokens = min(1000, max_context - chat_input["input_ids"].shape[-1] - 10)

            outputs = model.generate(
                input_ids=chat_input["input_ids"],
                attention_mask=chat_input.get("attention_mask"),
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            new_records = extract_records(generated_text)
            print(f"Found {len(new_records)} valid records in this batch")
            all_records.extend(new_records)
            time.sleep(2)

        except Exception as e:
            log_generation_error(model_name, Path(prompt).stem, e)
            print(f"⚠️ Error during generation: {e}")
            continue

    end_time = time.time()
    log_system_metrics(model_name, Path(prompt).stem, start_time, end_time)

    all_records = all_records[:NUM_RECORDS]

    if all_records:
        columns = [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level",
            "diabetes",
        ]
        df = pd.DataFrame(all_records, columns=columns)
        # Only convert numeric columns
        numeric_columns = [
            "age",
            "hypertension",
            "heart_disease",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level",
            "diabetes",
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        print("Failed to generate any valid records.")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Load the prompt
    with open(args.prompt_file, "r") as f:
        prompt = f.read()

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(prompt, args.model_name)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"bash/{args.model_name.replace('/', '_')}/records_prompt{Path(args.prompt_file).stem[-1]}"

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/diabetes_records.csv"

    # Save the synthetic data
    synthetic_data.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")

    # Print statistics
    print("\nData statistics:")
    print(
        f"Gender distribution: {synthetic_data['gender'].value_counts(normalize=True)}"
    )
    print(
        f"Diabetes prevalence: {synthetic_data['diabetes'].value_counts(normalize=True)}"
    )
    print(
        f"Age range: {synthetic_data['age'].min()} to {synthetic_data['age'].max()}, mean: {synthetic_data['age'].mean():.2f}"
    )
    print(
        f"BMI range: {synthetic_data['bmi'].min()} to {synthetic_data['bmi'].max()}, mean: {synthetic_data['bmi'].mean():.2f}"
    )


if __name__ == "__main__":
    main()
