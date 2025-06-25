import gc
import os
import re
import time

import pandas as pd
import torch
from huggingface_hub import hf_hub_url, model_info
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# info = model_info("mistralai/Mistral-7B-Instruct-v0.2")
# print(info.safetensors)

# Optional: uncomment if you need to force GPU visibility (not needed if all GPUs should be used)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuration
# MODEL_NAME = ""mistralai/Mistral-7B-Instruct-v0.2""
# OUTPUT_DIR = "mistral_records_withoutexample"
MODEL_NAME = "gpt2"
OUTPUT_DIR = "gpt2_records_withoutexample"
NUM_RECORDS = 100

SYSTEM_PROMPT = """You are a synthetic medical data generator. Generate realistic patient records for diabetes research."""


# def generate_chat_prompt(batch_size=10):
#     return f"""<s>[INST] {SYSTEM_PROMPT}
# Generate {batch_size} synthetic patient records for diabetes prediction with these exact fields:
# - gender (Male/Female)
# - age (0.0-100.0)
# - hypertension (0/1)
# - heart_disease (0/1)
# - smoking_history (never/former/current/not current/No Info)
# - bmi (15.0-60.0)
# - HbA1c_level (4.0-9.0)
# - blood_glucose_level (70-300)
# - diabetes (0/1

# Use comma-separated format, one record per line.

# ONLY output the records, with no additional text or explanation. Do not write any code. [/INST]
# """


def generate_chat_prompt(batch_size=10):
    return f"""{SYSTEM_PROMPT}

Generate {batch_size} synthetic patient records for diabetes prediction with these exact fields:
gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes

Use comma-separated format, one record per line.

ONLY output the records, with no additional text or explanation."""


def extract_records(text):
    print("=== DEBUG: GENERATED TEXT ===")
    print(text[:500])
    print("...")
    print("=== END DEBUG ===")

    records = []

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.count(",") != 8:
            continue

        try:
            parts = line.split(",")
            if len(parts) != 9:
                continue

            gender, age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes = parts

            if gender not in ["Male", "Female"]:
                continue

            if (
                gender == "Male"
                and age == "45.2"
                and hyp == "0"
                and heart == "0"
                and smoking == "never"
                and bmi == "26.3"
                and hba1c == "5.2"
                and glucose == "110"
                and diabetes == "0"
            ) or (
                gender == "Female"
                and age == "62.7"
                and hyp == "1"
                and heart == "1"
                and smoking == "former"
                and bmi == "32.1"
                and hba1c == "7.1"
                and glucose == "185"
                and diabetes == "1"
            ):
                continue

            records.append(line)

        except Exception as e:
            print(f"Error validating line: {line}")
            print(f"Error: {str(e)}")
            continue

    return records


def main():
    # Detect device
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {MODEL_NAME} on {DEVICE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # GPT-2 is small and fast
    quantization_config = None

    # 4-bit quantization config
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map={"": torch.device("cuda:0")},
        # device_map="auto",  # Use all available GPUs
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 has no pad token, so this line should remain
    # GPT-2 has no pad_token by default — we must define one manually
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # use EOS as pad token
        model.config.pad_token_id = tokenizer.eos_token_id

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    print("Starting generation...")

    all_records = []
    batch_size = 20

    while len(all_records) < NUM_RECORDS:
        records_needed = min(batch_size, NUM_RECORDS - len(all_records))
        print(
            f"Generating batch of {records_needed} records... ({len(all_records)}/{NUM_RECORDS} total)"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        prompt = generate_chat_prompt(records_needed)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            DEVICE
        )

        try:
            print("Input IDs:", inputs["input_ids"])
            print("Max token ID:", torch.max(inputs["input_ids"]))
            print("Tokenizer vocab size:", tokenizer.vocab_size)
            outputs = model.generate(
                **inputs,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,  # safe for GPT2
                # pad_token_id=tokenizer.eos_token_id, # mistral
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.replace(prompt, "").strip()
            new_records = extract_records(response)

            print(f"Found {len(new_records)} valid records in this batch")
            all_records.extend(new_records)

            if not new_records:
                print("Warning: No valid records found in this batch.")
            time.sleep(2)

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            time.sleep(5)

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

        output_path = os.path.join(OUTPUT_DIR, "diabetes_records.csv")
        df.to_csv(output_path, index=False)

        print(f"\n✅ Successfully generated {len(all_records)} records!")
        print(f"📁 Data saved to: {output_path}")

        print("\n📊 Data statistics:")
        print(f"Gender distribution:\n{df['gender'].value_counts(normalize=True)}\n")
        print(f"Diabetes prevalence:\n{df['diabetes'].value_counts(normalize=True)}\n")
        print(
            f"Age: {df['age'].min()} - {df['age'].max()} (mean: {df['age'].mean():.2f})"
        )
        print(
            f"BMI: {df['bmi'].min()} - {df['bmi'].max()} (mean: {df['bmi'].mean():.2f})"
        )
    else:
        print("❌ Failed to generate any valid records.")


if __name__ == "__main__":
    main()
