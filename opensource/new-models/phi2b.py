import gc
import os
import re
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Be very specific about which MIG device to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0:9:0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

# Configuration
MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "phi2_records_new"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RECORDS = 100

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modified prompt with chat format which works better with Phi-2
CHAT_PROMPT = """<human>
I need you to generate exactly 10 synthetic patient records for diabetes research.
The records should be in CSV format, with these fields:
Features and Statistics:
1. gender
   - Distribution: Male: 48%, Female: 52%

2. age
   - Mean: 41.8, Std: 15.2
   - Range: 18.0-80.0
   - Distribution: Slightly right-skewed

3. hypertension
   - Distribution: No (0): 85%, Yes (1): 15%
   - Correlates with age and BMI

4. heart_disease
   - Distribution: No (0): 92%, Yes (1): 8%
   - Correlates with age and hypertension

5. smoking_history
   - Categories: never (60%), former (22%), current (15%), not current (3%)

6. bmi
   - Mean: 27.3, Std: 6.4
   - Range: 15.0-60.0
   - Distribution: Right-skewed

7. HbA1c_level
   - Mean: 5.7, Std: 0.9
   - Range: 4.0-9.0
   - Distribution: Right-skewed
   - Strong correlation with diabetes status

8. blood_glucose_level
   - Mean: 138.0, Std: 40.5
   - Range: 70-300
   - Distribution: Right-skewed
   - Strong correlation with HbA1c_level

9. diabetes
   - Distribution: No (0): 88%, Yes (1): 12%
   - Correlates strongly with HbA1c_level and blood_glucose_level

Important correlations to maintain:
1. Higher age correlates with increased hypertension and heart disease risk
2. Higher BMI correlates with increased diabetes risk
3. HbA1c_level strongly correlates with diabetes status
4. blood_glucose_level correlates with HbA1c_level and diabetes status
5. Hypertension and heart_disease are more common in older ages

Just provide the 10 records in the exact same format, one per line. No explanations or other text.
</human>

<assistant>
"""


"""<human>
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        quantization_config=quantization_config if DEVICE == "cuda" else None,
        trust_remote_code=True,
        # Remove device_map parameter
    )
    model = model.to(DEVICE)

    # # Load model
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     torch_dtype=torch.float16,
    #     quantization_config=quantization_config,
    #     trust_remote_code=True,
    #     device_map="auto",
    #     # device_map="cpu",
    # )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
