import os
import re

import pandas as pd
import torch
from prompts import PROMPTS
from tqdm import tqdm
from transformers import pipeline, set_seed

# Configuration
MODEL_NAME = "gpt2"
OUTPUT_DIR = "gpt2-old-test-new-prompts"
BATCH_SIZE = 10  # Reduced batch size for better stability
NUM_BATCHES = 10  # Increased number of batches to maintain total count
DEVICE = 0 if torch.cuda.is_available() else -1
MAX_NEW_TOKENS = 200  # Specific tokens to generate
# set_seed(42)  # Set seed for reproducibility

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
        if not (70 <= int(glucose) <= 300):
            return False
        if int(diabetes) not in [0, 1]:
            return False

        return True
    except:
        return False


def extract_records(text):
    """Extract valid records from generated text"""
    # Split into lines and find lines that might be records
    lines = text.split("\n")
    potential_records = [line.strip() for line in lines if "," in line]

    # Validate each potential record
    valid_records = [record for record in potential_records if validate_record(record)]
    return valid_records


def generate_dataset(generator, prompt, num_records=100):
    """Generate dataset with improved batch processing"""
    all_records = []
    pbar = tqdm(total=num_records, desc="Generating records")

    # Process in smaller batches
    batch_size = 5  # Reduced from 10 to 5 for better stability

    while len(all_records) < num_records:
        try:
            # Generate multiple sequences at once
            outputs = generator(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=batch_size,  # Generate multiple sequences per call
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256,
                batch_size=batch_size,
            )

            # Process all outputs from the batch
            for output in outputs:
                new_records = extract_records(output["generated_text"])

                # Add new valid records
                for record in new_records:
                    if len(all_records) < num_records:
                        all_records.append(record)
                        pbar.update(1)
                    else:
                        break

            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in generation: {str(e)}")
            continue

    pbar.close()
    return all_records


def save_and_validate_dataset(records, prompt_name):
    """Save dataset and perform basic validation"""
    output_path = os.path.join(OUTPUT_DIR, f"{prompt_name}_data.csv")

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
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["HbA1c_level"] = pd.to_numeric(df["HbA1c_level"], errors="coerce")
    df["blood_glucose_level"] = pd.to_numeric(
        df["blood_glucose_level"], errors="coerce"
    )
    df["hypertension"] = pd.to_numeric(df["hypertension"], errors="coerce")
    df["heart_disease"] = pd.to_numeric(df["heart_disease"], errors="coerce")
    df["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce")

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print basic statistics
    print(f"\nDataset statistics for {prompt_name}:")
    print(f"Total records: {len(df)}")
    print("\nNumerical columns summary:")
    print(df.describe())
    print("\nCategorical columns value counts:")
    for col in [
        "gender",
        "smoking_history",
        "hypertension",
        "heart_disease",
        "diabetes",
    ]:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True))


def main():
    # Initialize generator with better memory settings
    print(f"Initializing generator with {MODEL_NAME} on device {DEVICE}")
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE >= 0 else torch.float32,
        model_kwargs={"low_cpu_mem_usage": True},
    )  # Add memory optimization

    # Generate for each prompt
    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n=== Generating for: {prompt_name} ===")

        try:
            # Generate dataset with periodic memory clearing
            records = generate_dataset(generator, prompt_text)

            # Save and validate
            if records:  # Only save if we got records
                save_and_validate_dataset(records, prompt_name)
                print(f"Completed generation for {prompt_name}")
            else:
                print(f"No valid records generated for {prompt_name}")

        except Exception as e:
            print(f"Error processing {prompt_name}: {str(e)}")
            continue

        # Clear memory after each prompt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
