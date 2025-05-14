import gc
import os
import re
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Restrict to first MIG instance
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "opensource/new-models/mistral_records"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RECORDS = 100  # Total records to generate

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

        # Remove leading numbering like '1. ' or '23. '
        line = re.sub(r"^\d+\.\s*", "", line)

        if not line or line.count(",") != 8:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 9:
            continue

        gender, age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes = parts

        if gender not in ["Male", "Female"]:
            continue

        # Add additional validation if necessary

        records.append(",".join(parts))

    return records


def main():
    print(f"Loading model {MODEL_NAME} on {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "8GiB"},
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Starting generation...")

    all_records = []
    batch_size = 20

    while len(all_records) < NUM_RECORDS:
        records_needed = min(batch_size, NUM_RECORDS - len(all_records))
        print(f"Generating batch of {records_needed} records... ({len(all_records)}/{NUM_RECORDS} total)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": generate_chat_prompt(records_needed)},
        ]

        chat_str = tokenizer.apply_chat_template(
            messages, tokenize=False
        )

        inputs = tokenizer(chat_str, return_tensors="pt").to(DEVICE)

        try:
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.replace(chat_str, "").strip()
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
        df["blood_glucose_level"] = pd.to_numeric(df["blood_glucose_level"], errors="coerce")
        df["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce")

        output_path = os.path.join(OUTPUT_DIR, "diabetes_records.csv")
        df.to_csv(output_path, index=False)

        print(f"\nSuccessfully generated {len(all_records)} records!")
        print(f"Data saved to {output_path}")
        print("\nData statistics:")
        print(f"Gender distribution: {df['gender'].value_counts(normalize=True)}")
        print(f"Diabetes prevalence: {df['diabetes'].value_counts(normalize=True)}")
        print(f"Age range: {df['age'].min()} to {df['age'].max()}, mean: {df['age'].mean():.2f}")
        print(f"BMI range: {df['bmi'].min()} to {df['bmi'].max()}, mean: {df['bmi'].mean():.2f}")

    else:
        print("Failed to generate any valid records.")


if __name__ == "__main__":
    main()



# import gc
# import os
# import re
# import time

# import pandas as pd
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# # Restrict to first MIG instance
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# # Configuration - Using Mistral Instruct model since it works well
# MODEL_NAME = (
#     "mistralai/Mistral-7B-Instruct-v0.2"  # Updated to v0.2 as shown in your output
# )
# OUTPUT_DIR = "mistral_records_prompt-prompt1"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_RECORDS = 100  # Total records to generate

# # Create output directory - FIX: Make sure this happens at the right time and location
# # Move this line to the main function to ensure it's executed before saving files
# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Mistral uses a specific chat format
# SYSTEM_PROMPT = """You are a synthetic medical data generator. Generate realistic patient records for diabetes research."""


# def generate_chat_prompt(batch_size=10):
#     return f"""<s>[INST] {SYSTEM_PROMPT}
# Generate {batch_size} realistic synthetic patient records for diabetes prediction. Here are the features with acceptable values and their definition:

# Features:
# 1. gender: Patient's gender (Male/Female)
# 2. age: Patient's age in years (Float: 0.0-100.0)
# 3. hypertension: Whether patient has hypertension (0: No, 1: Yes)
# 4. heart_disease: Whether patient has heart disease (0: No, 1: Yes)
# 5. smoking_history: Patient's smoking history (never/former/current/not current)
# 6. bmi: Body Mass Index, measure of body fat based on weight and height (Float: 15.0-60.0)
# 7. HbA1c_level: Hemoglobin A1c level, measure of average blood sugar over past 3 months (Float: 4.0-9.0)
# 8. blood_glucose_level: Current blood glucose level in mg/dL (Integer: 70-300)
# 9. diabetes: Whether patient has diabetes (0: No, 1: Yes)

# Examples from real data:
# 1. Female,45.2,1,0,never,28.5,6.2,140,0
# 2. Male,62.7,1,1,former,32.1,7.1,185,1
# 3. Female,38.9,0,0,current,24.3,5.8,130,0

# Generate {batch_size} comma-separated records, one per line, maintaining realistic correlations between features. [/INST]
# """


# def extract_records(text):
#     """Extract valid records from generated text"""
#     print("=== DEBUG: GENERATED TEXT ===")
#     print(text[:500])  # Print first 500 chars for debugging
#     print("...")
#     print("=== END DEBUG ===")

#     records = []

#     # Extract lines that match our expected pattern
#     for line in text.split("\n"):
#         line = line.strip()

#         # Skip empty lines
#         if not line:
#             continue

#         # Skip lines that don't have 8 commas (9 fields)
#         if line.count(",") != 8:
#             continue

#         # Validate fields
#         try:
#             parts = line.split(",")
#             if len(parts) != 9:
#                 continue

#             gender, age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes = parts

#             # Simple validation
#             if gender not in ["Male", "Female"]:
#                 continue

#             # Skip examples from prompt
#             # if (
#             #     gender == "Male"
#             #     and age == "45.2"
#             #     and hyp == "0"
#             #     and heart == "0"
#             #     and smoking == "never"
#             #     and bmi == "26.3"
#             #     and hba1c == "5.2"
#             #     and glucose == "110"
#             #     and diabetes == "0"
#             # ):
#             #     continue

#             # if (
#             #     gender == "Female"
#             #     and age == "62.7"
#             #     and hyp == "1"
#             #     and heart == "1"
#             #     and smoking == "former"
#             #     and bmi == "32.1"
#             #     and hba1c == "7.1"
#             #     and glucose == "185"
#             #     and diabetes == "1"
#             # ):
#             #     continue

#             # Add record if it passes validation
#             records.append(line)

#         except Exception as e:
#             print(f"Error validating line: {line}")
#             print(f"Error: {str(e)}")
#             continue

#     return records


# def main():
#     print(f"Loading model {MODEL_NAME} on {DEVICE}")

#     # FIX: Create output directory at the start of main function
#     # This ensures the directory exists before we try to save files
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # Configure quantization
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#     )

#     # Load model with optimized settings
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         torch_dtype=torch.float16,
#         quantization_config=quantization_config,
#         device_map="auto",
#         trust_remote_code=True,
#         max_memory={0: "8GiB"},  # Match your MIG slice size
#     )

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

#     # Ensure pad token is set
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     print("Starting generation...")

#     all_records = []
#     batch_size = 20  # Generate more records per batch

#     while len(all_records) < NUM_RECORDS:
#         records_needed = min(batch_size, NUM_RECORDS - len(all_records))
#         print(
#             f"Generating batch of {records_needed} records... ({len(all_records)}/{NUM_RECORDS} total)"
#         )

#         # Clear memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()

#         # Generate with Mistral chat format
#         prompt = generate_chat_prompt(records_needed)
#         inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

#         try:
#             # Generate with parameters tuned for Mistral
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=2000,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.9,
#                 pad_token_id=tokenizer.eos_token_id,
#             )

#             # Process response
#             generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#             # Extract assistant's response (after the prompt)
#             response = generated_text.replace(prompt, "").strip()

#             # Extract and validate records
#             new_records = extract_records(response)

#             print(f"Found {len(new_records)} valid records in this batch")

#             # Add new records
#             all_records.extend(new_records)

#             # Safety check - if no records in multiple attempts, something's wrong
#             if not new_records:
#                 print("Warning: No valid records found in this batch.")

#             # Avoid rate limiting
#             time.sleep(2)

#         except Exception as e:
#             print(f"Error during generation: {str(e)}")
#             time.sleep(5)

#     # Trim to desired count
#     all_records = all_records[:NUM_RECORDS]

#     # Save results
#     if all_records:
#         # Create DataFrame
#         columns = [
#             "gender",
#             "age",
#             "hypertension",
#             "heart_disease",
#             "smoking_history",
#             "bmi",
#             "HbA1c_level",
#             "blood_glucose_level",
#             "diabetes",
#         ]

#         df = pd.DataFrame([r.split(",") for r in all_records], columns=columns)

#         # Convert types
#         df["age"] = pd.to_numeric(df["age"], errors="coerce")
#         df["hypertension"] = pd.to_numeric(df["hypertension"], errors="coerce")
#         df["heart_disease"] = pd.to_numeric(df["heart_disease"], errors="coerce")
#         df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
#         df["HbA1c_level"] = pd.to_numeric(df["HbA1c_level"], errors="coerce")
#         df["blood_glucose_level"] = pd.to_numeric(
#             df["blood_glucose_level"], errors="coerce"
#         )
#         df["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce")

#         # Save to CSV
#         output_path = os.path.join(OUTPUT_DIR, "diabetes_records.csv")
#         df.to_csv(output_path, index=False)

#         print(f"\nSuccessfully generated {len(all_records)} records!")
#         print(f"Data saved to {output_path}")

#         # Display basic statistics
#         print("\nData statistics:")
#         print(f"Gender distribution: {df['gender'].value_counts(normalize=True)}")
#         print(f"Diabetes prevalence: {df['diabetes'].value_counts(normalize=True)}")
#         print(
#             f"Age range: {df['age'].min()} to {df['age'].max()}, mean: {df['age'].mean():.2f}"
#         )
#         print(
#             f"BMI range: {df['bmi'].min()} to {df['bmi'].max()}, mean: {df['bmi'].mean():.2f}"
#         )

#     else:
#         print("Failed to generate any valid records.")


# if __name__ == "__main__":
#     main()
