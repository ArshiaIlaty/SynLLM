import argparse
import gc
import os
import re
import time

import pandas as pd
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available

# Use all available GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Uncomment if you need to specify specific GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Configuration
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "opensource/new-models/llm_records"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RECORDS = 100

# Added more open source models with their respective chat styles
MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.2": "chatml",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "chatml",
    "meta-llama/Llama-2-7b-chat-hf": "llama2",
    "meta-llama/Llama-3-8b-Instruct": "llama3",
    "google/gemma-7b-it": "gemma",
    "01-ai/Yi-6B-Chat": "im_start",
    "HuggingFaceH4/zephyr-7b-beta": "zephyr",
    "microsoft/Phi-3-mini-4k-instruct": "phi",
    "BAAI/Qwen-7B-Chat": "qwen",
    "Qwen/Qwen2-7B-Instruct": "qwen2",
    "internlm/internlm2-7b-chat": "internlm",
    "stabilityai/StableBeluga-7B": "beluga",
    "openchat/openchat-3.5-0106": "openchat",
    "NousResearch/Nous-Hermes-2-Yi-34B": "nous",
    "mosaicml/mpt-7b-instruct": "mpt",
}

# Chat style mappings for different model families
CHAT_STYLE_MODELS = {
    "mistral": "chatml",
    "mixtral": "chatml",
    "llama2": "llama",
    "llama3": "llama",
    "gemma": "openai",
    "yi": "im_start",
    "zephyr": "zephyr",
    "phi": "phi",
    "qwen": "qwen",
    "qwen2": "qwen2",
    "internlm": "internlm",
    "beluga": "beluga",
    "openchat": "openchat",
    "nous": "nous",
    "mpt": "mpt",
}

SYSTEM_PROMPT = """You are a synthetic medical data generator. Generate realistic patient records for diabetes research."""


def generate_chat_prompt(batch_size=10):
    return f"""Generate {batch_size} realistic synthetic patient records for diabetes prediction. Here are the features with acceptable values and their definition:

Features:
1. gender: Patient's gender (Male/Female)
2. age: Patient's age in years (Float: 0.0-100.0)
3. hypertension: Whether patient has hypertension (0: No, 1: Yes)
4. heart_disease: Whether patient has heart disease (0: No, 1: Yes)
5. smoking_history: Patient's smoking history (never/former/current/not current)
6. bmi: Body Mass Index, measure of body fat based on weight and height (Float: 15.0-60.0)
7. HbA1c_level: Hemoglobin A1c level, measure of average blood sugar over past 3 months (Float: 4.0-9.0)
8. blood_glucose_level: Current blood glucose level in mg/dL (Integer: 70-300)
9. diabetes: Whether patient has diabetes (0: No, 1: Yes)

Examples from real data:
1. Female,45.2,1,0,never,28.5,6.2,140,0
2. Male,62.7,1,1,former,32.1,7.1,185,1
3. Female,38.9,0,0,current,24.3,5.8,130,0

Generate {batch_size} comma-separated records, one per line, maintaining realistic correlations between features."""


def extract_records(text):
    print("=== DEBUG: GENERATED TEXT ===")
    print(text[:500])
    print("...\n=== END DEBUG ===")

    records = []
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"^\d+\.\s*", "", line)

        if not line or line.count(",") != 8:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 9:
            continue

        gender, age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes = parts

        if gender not in ["Male", "Female"]:
            continue

        records.append(",".join(parts))

    return records


def detect_chat_style(model_name):
    for key, style in CHAT_STYLE_MODELS.items():
        if key in model_name.lower():
            return style
    return "plain"


def build_prompt(model_name, system_prompt, user_prompt, tokenizer):
    style = detect_chat_style(model_name)

    if style == "llama":
        # Llama 2 and Llama 3 style
        prompt = f"<s>[INST] {system_prompt}\n{user_prompt} [/INST]"
        return tokenizer(prompt, return_tensors="pt")

    elif style == "chatml":
        # ChatML format (Mistral, Mixtral)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")

    elif style == "openai":
        # Gemma style
        if "gemma" in model_name.lower():
            # Gemma only supports user and assistant roles
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")

    elif style == "im_start":
        # Yi style
        prompt_str = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer(prompt_str, return_tensors="pt")

    elif style == "zephyr":
        # Zephyr style
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")

    elif style == "phi":
        # Phi style
        prompt = (
            f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        )
        return tokenizer(prompt, return_tensors="pt")

    elif style == "qwen" or style == "qwen2":
        # Qwen style
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")

    elif style == "internlm":
        # InternLM style
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer(prompt, return_tensors="pt")

    elif style == "openchat":
        # OpenChat style
        prompt = f"GPT4 System: {system_prompt}\nHuman: {user_prompt}\nAssistant:"
        return tokenizer(prompt, return_tensors="pt")

    elif style == "nous":
        # Nous-Hermes style
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer(prompt, return_tensors="pt")

    elif style == "mpt":
        # MPT style
        prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        return tokenizer(prompt, return_tensors="pt")

    else:
        # Default plain style
        prompt = f"{system_prompt}\n\n{user_prompt}"
        return tokenizer(prompt, return_tensors="pt")


def generate_batch(
    model, tokenizer, model_name, system_prompt, batch_size, device_id=0
):
    """Generate a batch of records on a specific GPU"""
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    print(f"Generating batch of {batch_size} records on {device}...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    chat_input = build_prompt(
        model_name,
        system_prompt,
        generate_chat_prompt(batch_size),
        tokenizer,
    ).to(device)

    try:
        outputs = model.generate(
            input_ids=chat_input["input_ids"],
            attention_mask=chat_input.get("attention_mask"),
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text.strip()
        new_records = extract_records(response)

        print(f"Found {len(new_records)} valid records in this batch on {device}")
        return new_records

    except Exception as e:
        print(f"Error during generation on {device}: {str(e)}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--num_records", type=int, default=NUM_RECORDS)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    model_name = args.model_name
    num_records = args.num_records
    batch_size = args.batch_size
    output_dir = args.output_dir

    # Create model-specific output directory
    model_short_name = model_name.split("/")[-1].lower()
    output_dir = os.path.join(output_dir, model_short_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model {model_name} on {DEVICE}")

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    try:
        # Setup quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Distribute memory across available GPUs
        max_memory = {}
        for i in range(num_gpus):
            max_memory[i] = "40GiB"  # Adjust based on your GPU memory

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",  # Let the library handle distribution across GPUs
            trust_remote_code=True,
            max_memory=max_memory,
        )
    except Exception as e:
        print(f"⚠️ Could not load model in 4-bit: {e}\nFalling back to full precision.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Starting generation...")

    all_records = []

    # Calculate batch size per GPU
    batch_size_per_gpu = batch_size * 2 if num_gpus > 1 else batch_size

    while len(all_records) < num_records:
        records_needed = min(batch_size_per_gpu, num_records - len(all_records))
        print(
            f"Generating batch of {records_needed} records... ({len(all_records)}/{num_records} total)"
        )

        # Generate using the model (which is already distributed across GPUs)
        new_records = generate_batch(
            model, tokenizer, model_name, SYSTEM_PROMPT, records_needed
        )

        all_records.extend(new_records)

        if not new_records:
            print("Warning: No valid records found in this batch.")
            time.sleep(5)
        else:
            time.sleep(2)  # Shorter sleep between successful batches

    all_records = all_records[:num_records]

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

        df = pd.DataFrame([r.split(",") for r in all_records], columns=columns)

        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["hypertension"] = pd.to_numeric(df["hypertension"], errors="coerce")
        df["heart_disease"] = pd.to_numeric(df["heart_disease"], errors="coerce")
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["HbA1c_level"] = pd.to_numeric(df["HbA1c_level"], errors="coerce")
        df["blood_glucose_level"] = pd.to_numeric(
            df["blood_glucose_level"], errors="coerce"
        )
        df["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce")

        output_path = os.path.join(output_dir, "diabetes_records.csv")
        df.to_csv(output_path, index=False)

        print(f"\nSuccessfully generated {len(all_records)} records!")
        print(f"Data saved to {output_path}")
        print("\nData statistics:")
        print(f"Gender distribution: {df['gender'].value_counts(normalize=True)}")
        print(f"Diabetes prevalence: {df['diabetes'].value_counts(normalize=True)}")
        print(
            f"Age range: {df['age'].min()} to {df['age'].max()}, mean: {df['age'].mean():.2f}"
        )
        print(
            f"BMI range: {df['bmi'].min()} to {df['bmi'].max()}, mean: {df['bmi'].mean():.2f}"
        )

    else:
        print("Failed to generate any valid records.")


def list_available_models():
    print("Available models:")
    for model, style in MODELS.items():
        print(f"- {model} (style: {style})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models", action="store_true", help="List all available models"
    )
    args, _ = parser.parse_known_args()

    if args.list_models:
        list_available_models()
    else:
        main()
