import os

import pandas as pd
import torch
from prompts import PROMPTS
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Configuration
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MODEL_NAME = "tiiuae/falcon-7b"
OUTPUT_DIR = "model_mistral"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 200
# set_seed(42)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def validate_record(record):
    """Validate if a generated record matches expected format and constraints"""
    try:
        parts = record.strip().split(",")
        if len(parts) != 9:
            return False

        # Validate each field
        gender, age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes = parts

        # Basic format checks
        if gender not in ["Male", "Female"]:
            return False
        if not (18 <= float(age) <= 80):
            return False
        if int(hyp) not in [0, 1]:
            return False
        if int(heart) not in [0, 1]:
            return False
        if smoking not in ["never", "former", "current", "not current"]:
            return False
        if not (15 <= float(bmi) <= 60):
            return False
        if not (4 <= float(hba1c) <= 9):
            return False
        if not (70 <= int(float(glucose)) <= 300):
            return False
        if int(diabetes) not in [0, 1]:
            return False

        return True
    except:
        return False


def extract_records(text):
    """Extract valid records from generated text"""
    lines = text.split("\n")
    potential_records = [line.strip() for line in lines if "," in line]
    valid_records = [record for record in potential_records if validate_record(record)]
    return valid_records


def generate_batch(model, tokenizer, prompt, num_records=1000):
    """Generate a batch of records"""
    records = []
    pbar = tqdm(total=num_records, desc="Generating records")

    while len(records) < num_records:
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        try:
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode and extract records
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_records = extract_records(generated_text)

            # Add valid records
            for record in new_records:
                if len(records) < num_records:
                    records.append(record)
                    pbar.update(1)
                else:
                    break

        except Exception as e:
            print(f"Error in generation: {str(e)}")
            continue

        # Update prompt with successful records to guide generation
        if len(new_records) > 0:
            prompt = prompt + "\n" + "\n".join(new_records[-2:])

    pbar.close()
    return records


def save_and_analyze_dataset(records, prompt_name):
    """Save and analyze the generated dataset"""
    output_path = os.path.join(OUTPUT_DIR, f"mistral_{prompt_name}_data.csv")

    # Create DataFrame
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

    df = pd.DataFrame([r.split(",") for r in records], columns=columns)

    # Convert datatypes
    numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    binary_cols = ["hypertension", "heart_disease", "diabetes"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print analysis
    print(f"\nDataset analysis for {prompt_name}:")
    print(f"Total records: {len(df)}")
    print("\nNumerical columns summary:")
    print(df[numeric_cols].describe())
    print("\nValue counts:")
    for col in ["gender", "smoking_history"] + binary_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True))

    # Check medical consistency
    medical_checks = {
        "diabetes_hba1c": len(df[(df["diabetes"] == 1) & (df["HbA1c_level"] > 6.5)])
        / len(df[df["diabetes"] == 1]),
        "diabetes_glucose": len(
            df[(df["diabetes"] == 1) & (df["blood_glucose_level"] > 180)]
        )
        / len(df[df["diabetes"] == 1]),
        "hypertension_age": len(df[(df["hypertension"] == 1) & (df["age"] > 50)])
        / len(df[df["hypertension"] == 1]),
    }
    print("\nMedical consistency checks (proportions):")
    print(medical_checks)


def main():
    print(f"Loading Falcon model on {DEVICE}")

    # Load model and tokenizer with 8-bit quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        load_in_8bit=True,  # Added 8-bit quantization
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",  # Specific to Falcon
        trust_remote_code=True,  # Required for Falcon
    )
    tokenizer.pad_token = tokenizer.eos_token  # Specific to Falcon

    # Generate for each prompt
    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n=== Generating for: {prompt_name} ===")
        records = generate_batch(model, tokenizer, prompt_text)
        save_and_analyze_dataset(records, prompt_name)
        print(f"Completed generation for {prompt_name}")


if __name__ == "__main__":
    main()

# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "mistralai/Mistral-7B-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# prompt = "Generate 10 synthetic diabetes records in CSV format..."
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# outputs = model.generate(**inputs, max_length=1500)
# print(tokenizer.decode(outputs[0]))
