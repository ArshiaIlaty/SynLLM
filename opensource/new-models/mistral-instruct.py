import gc
import os
import re
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration - Using Mistral Instruct model since it works well
MODEL_NAME = (
    "mistralai/Mistral-7B-Instruct-v0.2"  # Updated to v0.2 as shown in your output
)
OUTPUT_DIR = "mistral_records"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RECORDS = 100  # Total records to generate

# Create output directory - FIX: Make sure this happens at the right time and location
# Move this line to the main function to ensure it's executed before saving files
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mistral uses a specific chat format
SYSTEM_PROMPT = """You are a synthetic medical data generator. Generate realistic patient records for diabetes research."""


def generate_chat_prompt(batch_size=10):
    return f"""<s>[INST] {SYSTEM_PROMPT}
Generate {batch_size} synthetic patient records for diabetes prediction with these exact fields:
- gender (Male/Female)
- age (0.0-100.0)
- hypertension (0/1)
- heart_disease (0/1)
- smoking_history (never/former/current/not current/No Info)
- bmi (15.0-60.0)
- HbA1c_level (4.0-9.0)
- blood_glucose_level (70-300)
- diabetes (0/1)

Use this exact comma-separated format, one record per line:
Male,45.2,0,0,never,26.3,5.2,110,0
Female,62.7,1,1,former,32.1,7.1,185,1

ONLY output the records, with no additional text or explanation. Do not write any code. [/INST]
"""


def extract_records(text):
    """Extract valid records from generated text"""
    print("=== DEBUG: GENERATED TEXT ===")
    print(text[:500])  # Print first 500 chars for debugging
    print("...")
    print("=== END DEBUG ===")

    records = []

    # Extract lines that match our expected pattern
    for line in text.split("\n"):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip lines that don't have 8 commas (9 fields)
        if line.count(",") != 8:
            continue

        # Validate fields
        try:
            parts = line.split(",")
            if len(parts) != 9:
                continue

            gender, age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes = parts

            # Simple validation
            if gender not in ["Male", "Female"]:
                continue

            # Skip examples from prompt
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
            ):
                continue

            if (
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

            # Add record if it passes validation
            records.append(line)

        except Exception as e:
            print(f"Error validating line: {line}")
            print(f"Error: {str(e)}")
            continue

    return records


def main():
    print(f"Loading model {MODEL_NAME} on {DEVICE}")

    # FIX: Create output directory at the start of main function
    # This ensures the directory exists before we try to save files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load model with optimized settings
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Starting generation...")

    all_records = []
    batch_size = 20  # Generate more records per batch

    while len(all_records) < NUM_RECORDS:
        records_needed = min(batch_size, NUM_RECORDS - len(all_records))
        print(
            f"Generating batch of {records_needed} records... ({len(all_records)}/{NUM_RECORDS} total)"
        )

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Generate with Mistral chat format
        prompt = generate_chat_prompt(records_needed)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        try:
            # Generate with parameters tuned for Mistral
            outputs = model.generate(
                **inputs,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Process response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract assistant's response (after the prompt)
            response = generated_text.replace(prompt, "").strip()

            # Extract and validate records
            new_records = extract_records(response)

            print(f"Found {len(new_records)} valid records in this batch")

            # Add new records
            all_records.extend(new_records)

            # Safety check - if no records in multiple attempts, something's wrong
            if not new_records:
                print("Warning: No valid records found in this batch.")

            # Avoid rate limiting
            time.sleep(2)

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            time.sleep(5)

    # Trim to desired count
    all_records = all_records[:NUM_RECORDS]

    # Save results
    if all_records:
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

        df = pd.DataFrame([r.split(",") for r in all_records], columns=columns)

        # Convert types
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["hypertension"] = pd.to_numeric(df["hypertension"], errors="coerce")
        df["heart_disease"] = pd.to_numeric(df["heart_disease"], errors="coerce")
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["HbA1c_level"] = pd.to_numeric(df["HbA1c_level"], errors="coerce")
        df["blood_glucose_level"] = pd.to_numeric(
            df["blood_glucose_level"], errors="coerce"
        )
        df["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce")

        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, "diabetes_records.csv")
        df.to_csv(output_path, index=False)

        print(f"\nSuccessfully generated {len(all_records)} records!")
        print(f"Data saved to {output_path}")

        # Display basic statistics
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


if __name__ == "__main__":
    main()
