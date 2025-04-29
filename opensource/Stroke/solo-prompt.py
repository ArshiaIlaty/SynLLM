import gc
import os
import re
import time
import argparse
import random
from pathlib import Path
import psutil
import subprocess
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available

# Restrict to first MIG instance
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Configuration
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RECORDS = 100

# Chat style mappings
MODELS=(
  "openai-community/gpt2"
  "mistralai/Mistral-7B-Instruct-v0.2"
  # "mistralai/Mixtral-8x7B-Instruct-v0.1"
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
)

CHAT_STYLE_MODELS = {
    "gpt2": "plain",
    "mistral": "chatml",
    # "mixtral": "chatml",
    "llama2": "llama",
    "llama3": "llama",
    "gemma": "openai",
    "yi": "im_start",
    "zephyr": "zephyr",
    "qwen": "qwen",
    "qwen2": "qwen2",
    "internlm": "internlm",
    "beluga": "beluga",
    "openchat": "openchat",
    "nous": "nous",
    "mpt": "mpt"
}

SYSTEM_PROMPT = """You are a synthetic medical data generator. Generate realistic patient records for stroke prediction research."""

EXAMPLE_RECORDS = [
    "Male,67.0,1,1,Yes,Private,Urban,228.69,36.6,formerly smoked,1",
    "Female,61.0,0,0,Yes,Self-employed,Rural,202.21,28.5,never smoked,0",
    "Male,52.0,1,0,Yes,Private,Rural,104.51,30.1,smokes,0",
    "Female,72.0,0,0,Yes,Private,Urban,221.29,34.6,never smoked,1",
    "Male,46.0,0,0,Yes,Self-employed,Urban,85.28,34.4,Unknown,0"
]

def detect_chat_style(model_name):
    for key, style in CHAT_STYLE_MODELS.items():
        if key in model_name.lower():
            return style
    return "plain"

def build_prompt(model_name, system_prompt, user_prompt, tokenizer):
    style = detect_chat_style(model_name)

    if style == "llama":
        return tokenizer(f"<s>[INST] {system_prompt}\n{user_prompt} [/INST]", return_tensors="pt")
    elif style == "chatml":
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")
    elif style == "openai":
        messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")
    elif style in ["im_start", "internlm", "nous"]:
        prompt_str = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer(prompt_str, return_tensors="pt")
    elif style in ["zephyr", "qwen", "qwen2"]:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")
    elif style == "phi":
        return tokenizer(f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n", return_tensors="pt")
    elif style == "openchat":
        return tokenizer(f"GPT4 System: {system_prompt}\nHuman: {user_prompt}\nAssistant:", return_tensors="pt")
    elif style == "mpt":
        return tokenizer(f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:", return_tensors="pt")
    else:
        return tokenizer(f"{system_prompt}\n\n{user_prompt}", return_tensors="pt")
        

def extract_records(text):
    print("=== DEBUG: GENERATED TEXT ===")
    print(text[:500])
    print("...\n=== END DEBUG ===")

    # Define valid values for categorical fields
    VALID_GENDER = {"male", "female", "other"}
    VALID_SMOKING = {"formerly smoked", "never smoked", "smokes", "unknown"}
    VALID_MARITAL = {"yes", "no"}
    VALID_WORK = {"private", "self-employed", "govt_job", "children", "never_worked"}
    VALID_RESIDENCE = {"urban", "rural"}
    
    yes_no_map = {"yes": "1", "no": "0"}

    records = []
    skipped = 0
    bad_lines = []

    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+\.\s*", "", line)
        
        # Skip lines that don't have 10 commas (11 fields)
        if not line or line.count(",") != 10:
            if line:  # Only log non-empty lines
                bad_lines.append(line)
                skipped += 1
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 11:
            bad_lines.append(line)
            skipped += 1
            continue

        gender, age, hyp, heart, married, work, residence, glucose, bmi, smoking, stroke = parts

        # Validate gender
        if gender.lower() not in VALID_GENDER:
            bad_lines.append(line)
            skipped += 1
            continue

        # Validate smoking status
        smoking = smoking.lower()
        if smoking not in VALID_SMOKING:
            if smoking == "no info" or smoking == "never":
                smoking = "unknown"
            else:
                bad_lines.append(line)
                skipped += 1
                continue

        # Validate marital status
        married = married.lower()
        if married not in VALID_MARITAL:
            bad_lines.append(line)
            skipped += 1
            continue

        # Validate work type
        work = work.lower()
        if work not in VALID_WORK:
            bad_lines.append(line)
            skipped += 1
            continue

        # Validate residence type
        residence = residence.lower()
        if residence not in VALID_RESIDENCE:
            bad_lines.append(line)
            skipped += 1
            continue

        # Process binary fields
        hyp = yes_no_map.get(hyp.lower(), hyp)
        heart = yes_no_map.get(heart.lower(), heart)
        stroke = yes_no_map.get(stroke.lower(), stroke)

        # Create a record with all processed values
        record = [
            gender.title(), age, hyp, heart, 
            married.title(), work.title(), residence.title(), 
            glucose, bmi, smoking.lower(), stroke
        ]
        records.append(record)

    if bad_lines:
        Path("reports").mkdir(parents=True, exist_ok=True)
        with open("reports/rejected_records.txt", "a") as bad:
            bad.write("\n".join(bad_lines) + "\n")

    print(f"[INFO] Extracted {len(records)} valid records, skipped {skipped}")
    
    # Debug output
    if records:
        print(f"[DEBUG] First few smoking status values: {[r[9] for r in records[:5]]}")
        print(f"[DEBUG] First few work type values: {[r[5] for r in records[:5]]}")
        print(f"[DEBUG] First few residence type values: {[r[6] for r in records[:5]]}")
    
    return records

def log_system_metrics(model_name, prompt_name, start_time, end_time):
    gpu_info_before = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"
    ]).decode().strip()

    gpu_info_after = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"
    ]).decode().strip()

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
    error_log = {
        "model": model_name,
        "prompt": prompt_name,
        "error": str(error)
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/errors.jsonl", "a") as errfile:
        errfile.write(json.dumps(error_log) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt_file", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    safe_model_name = model_name.replace("/", "_")
    prompt_name = Path(args.prompt_file).stem
    output_dir = Path("bash") / safe_model_name / f"records_{prompt_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stroke_records.csv"

    print(f"Loading model {model_name} on {DEVICE}")


    # Determine attention implementation
    if "phi" in model_name.lower() or "gpt2" in model_name.lower():
        attn_backend = "eager"
    else:
        attn_backend = "flash_attention_2"

    # Start building kwargs
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True
    }
    
    if "mpt" not in model_name.lower():  # MPT does not support this kwarg
        model_kwargs["attn_implementation"] = attn_backend
    
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        if "llama" in model_name.lower() and hasattr(config, "rope_scaling"):
            config.rope_scaling = {
                "name": "dynamic",
                "factor": config.rope_scaling.get("factor", 8.0)
            }

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            **model_kwargs
        )
    except Exception as e:
        print(f"⚠️ Could not load model with quantization ({e}), falling back to basic loading.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

    try:
        from transformers.models.gpt_neox.tokenization_gpt_neox import GPTNeoXTokenizer
    except ImportError:
        print("Could not import GPTNeoXTokenizer explicitly, falling back to AutoTokenizer.")
    
    # try:
    #     is_mpt = "mpt" in model_name.lower()
    
    #     if is_mpt:
    #         print("[INFO] Detected MPT model, skipping 4-bit loading for compatibility.")
    #         model = AutoModelForCausalLM.from_pretrained(
    #             model_name,
    #             torch_dtype=torch.float16,
    #             device_map="auto",
    #             trust_remote_code=True
    #         )
    #     else:
    #         quantization_config = BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             bnb_4bit_quant_type="nf4",
    #             bnb_4bit_compute_dtype=torch.float16
    #         )
    #         model = AutoModelForCausalLM.from_pretrained(
    #             model_name,
    #             torch_dtype=torch.float16,
    #             quantization_config=quantization_config,
    #             device_map="auto",
    #             trust_remote_code=True,
    #             attn_implementation="eager"
    #         )
    # except Exception as e:
    #     print(f"⚠️ Could not load model in 4-bit or custom config: {e}\nFalling back to basic loading.")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #         trust_remote_code=True
    #     )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.prompt_file, "r") as f:
        user_prompt = f.read()

    all_records = []
    batch_size = 20
    start_time = time.time()

    print("Starting generation...")

    while len(all_records) < NUM_RECORDS:
        needed = min(batch_size, NUM_RECORDS - len(all_records))
        print(f"Generating batch of {needed} records... ({len(all_records)}/{NUM_RECORDS} total)")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        chat_input = build_prompt(model_name, SYSTEM_PROMPT, user_prompt, tokenizer).to(DEVICE)


        # Add this to your batch processing logic
        if chat_input["input_ids"].max() >= tokenizer.vocab_size:
            print(f"Warning: input_ids contains invalid token IDs >= {tokenizer.vocab_size}")
            # Fix by clamping values
            chat_input["input_ids"] = torch.clamp(chat_input["input_ids"], 0, tokenizer.vocab_size-1)
        
        try:
            # Safe token limit based on model type and prompt complexity
            if model_name.lower() == "gpt2":
                max_tokens = 100  # GPT-2 has a 1024-token context limit
            elif "prompt4" in prompt_name.lower():
                max_tokens = 800  # Prompt 4 is large — lower to avoid OOM
            else:
                max_tokens = 2000
            outputs = model.generate(
                input_ids=chat_input["input_ids"],
                attention_mask=chat_input.get("attention_mask"),
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "gender" in generated_text.lower() and "age" in generated_text.lower():
                print("[DEBUG] Model likely returned schema, not data.")

            with open(output_dir / "raw_output.txt", "a") as f:
                f.write(generated_text + "\n\n")

            new_records = extract_records(generated_text)
            print(f"Found {len(new_records)} valid records in this batch")
            all_records.extend(new_records)
            time.sleep(2)
        except Exception as e:
            log_generation_error(model_name, prompt_name, e)
            continue

    end_time = time.time()
    log_system_metrics(model_name, prompt_name, start_time, end_time)

    all_records = all_records[:NUM_RECORDS]

    if all_records:
        # Define columns based on stroke dataset
        columns = [
            "gender", "age", "hypertension", "heart_disease", 
            "ever_married", "work_type", "Residence_type", 
            "avg_glucose_level", "bmi", "smoking_status", "stroke"
        ]
        
        df = pd.DataFrame(all_records, columns=columns)
        
        # Only convert numeric columns
        numeric_columns = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Debug output before saving
        print("\nDataFrame head before saving:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())
        
        # Generate a random ID for each record
        df['id'] = range(1, len(df) + 1)
        
        # Reorder columns to match the original dataset
        final_columns = ["id"] + columns
        df = df[final_columns]
        
        df.to_csv(output_path, index=False)

        print(f"\nSuccessfully generated {len(all_records)} records!")
        print(f"Data saved to {output_path}")
        print("\nData statistics:")
        print(f"Gender distribution: {df['gender'].value_counts(normalize=True)}")
        print(f"Work type distribution: {df['work_type'].value_counts(normalize=True)}")
        print(f"Smoking status distribution: {df['smoking_status'].value_counts(normalize=True)}")
        print(f"Stroke prevalence: {df['stroke'].value_counts(normalize=True)}")
        print(f"Age range: {df['age'].min()} to {df['age'].max()}, mean: {df['age'].mean():.2f}")
        print(f"BMI range: {df['bmi'].min()} to {df['bmi'].max()}, mean: {df['bmi'].mean():.2f}")
    else:
        print("Failed to generate any valid records.")

if __name__ == "__main__":
    main()