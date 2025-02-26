import gc
import os
import re
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration
MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "phi2_records"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RECORDS = 100

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modified prompt with chat format which works better with Phi-2
CHAT_PROMPT = """<human>
I need you to generate exactly 10 synthetic patient records for diabetes research.
The records should be in CSV format, with these fields:
- gender (Male/Female)
- age (0.0-100.0)
- hypertension (0/1)
- heart_disease (0/1)
- smoking_history (never/former/current/not current/No Info)
- bmi (15.0-60.0)
- HbA1c_level (4.0-9.0)
- blood_glucose_level (70-300)
- diabetes (0/1)

Here are two example records:
Male,45.2,0,0,never,26.3,5.2,110,0
Female,62.7,1,1,former,32.1,7.1,185,1

Just provide the 10 records in the exact same format, one per line. No explanations or other text.
</human>

<assistant>
"""


def process_output(text):
    """Process generated text to extract records"""
    # Look for lines with the right format (8 commas = 9 fields)
    records = []

    # Print the full text for debugging
    print("=== DEBUG: FULL GENERATED TEXT ===")
    print(text)
    print("=== END DEBUG ===")

    for line in text.split("\n"):
        line = line.strip()
        if line.count(",") == 8:
            parts = line.split(",")
            # Basic validation checks
            try:
                gender = parts[0]
                age = float(parts[1])
                hypertension = int(parts[2])
                heart_disease = int(parts[3])
                smoking = parts[4]
                bmi = float(parts[5])
                hba1c = float(parts[6])
                glucose = int(parts[7])
                diabetes = int(parts[8])

                # Simple validation
                if (
                    gender in ["Male", "Female"]
                    and 0 <= age <= 100
                    and hypertension in [0, 1]
                    and heart_disease in [0, 1]
                    and smoking
                    in ["never", "former", "current", "not current", "No Info"]
                    and 15 <= bmi <= 60
                    and 4 <= hba1c <= 9
                    and 70 <= glucose <= 300
                    and diabetes in [0, 1]
                ):
                    records.append(line)
                    print(f"Valid record found: {line}")
            except (ValueError, IndexError) as e:
                print(f"Validation failed for line: {line}")
                print(f"Error: {str(e)}")
                continue

    return records


def main():
    print(f"Loading model {MODEL_NAME} on {DEVICE}")

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Starting generation...")

    all_records = []
    batches_needed = (NUM_RECORDS + 9) // 10  # Ceiling division by 10

    for batch in range(batches_needed):
        print(f"Generating batch {batch+1}/{batches_needed}...")

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Generate in chat format
        inputs = tokenizer(CHAT_PROMPT, return_tensors="pt").to(DEVICE)

        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,  # Increased to ensure complete generation
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

            # Decode and extract the assistant's response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Ensure we only look at the assistant's part of the response
            assistant_text = generated_text.split("</human>")[-1].strip()

            # Handle the special tag if present
            if "<assistant>" in assistant_text:
                assistant_text = assistant_text.split("<assistant>")[-1].strip()

            new_records = process_output(assistant_text)

            print(f"Found {len(new_records)} valid records in this batch")
            all_records.extend(new_records)

            # Early exit if we've reached our goal
            if len(all_records) >= NUM_RECORDS:
                break

            # Avoid rate limiting/overheating
            time.sleep(2)

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            time.sleep(5)  # Longer pause after error

    # Trim to desired count
    all_records = all_records[:NUM_RECORDS]

    # Save to CSV if we have any records
    if all_records:
        df = pd.DataFrame(
            [r.split(",") for r in all_records],
            columns=[
                "gender",
                "age",
                "hypertension",
                "heart_disease",
                "smoking_history",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level",
                "diabetes",
            ],
        )

        # Convert types for analysis
        df["age"] = pd.to_numeric(df["age"])
        df["hypertension"] = pd.to_numeric(df["hypertension"])
        df["heart_disease"] = pd.to_numeric(df["heart_disease"])
        df["bmi"] = pd.to_numeric(df["bmi"])
        df["HbA1c_level"] = pd.to_numeric(df["HbA1c_level"])
        df["blood_glucose_level"] = pd.to_numeric(df["blood_glucose_level"])
        df["diabetes"] = pd.to_numeric(df["diabetes"])

        output_path = os.path.join(OUTPUT_DIR, "phi2_diabetes_records.csv")
        df.to_csv(output_path, index=False)

        print(f"\nGenerated {len(all_records)} records")
        print(f"Data saved to {output_path}")

        # Display some statistics
        print("\nSample statistics:")
        print(f"Gender distribution: {df['gender'].value_counts(normalize=True)}")
        print(f"Diabetes prevalence: {df['diabetes'].value_counts(normalize=True)}")
    else:
        print("No valid records were generated. Consider using a different model.")


if __name__ == "__main__":
    main()
