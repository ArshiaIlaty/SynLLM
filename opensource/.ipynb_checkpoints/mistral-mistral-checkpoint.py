import gc
import json
import os
import re
import time
from datetime import datetime

import GPUtil
import pandas as pd
import psutil
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
# MODEL_NAME = "mistralai/Mistral-2B-v0.1"
OUTPUT_DIR = "model_mistral-100-prompt-in-5experiments"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 200  # Specific tokens to generate
NUM_EXPERIMENTS = 5  # Number of experiments to run
BATCH_SIZE = 10  # Number of sequences to generate in parallel
NUM_RECORDS = 100  # Number of records to generate per prompt
# set_seed(42)  # Set seed for reproducibility

# Define prompts dictionary
PROMPTS = {
    # Step 1: Basic Prompt with Examples
    "PROMPT_1": """
    Generate 100 realistic synthetic patient records for diabetes prediction following this format:
    - gender (String: 'Male' or 'Female')
    - age (Float: 0.0-100.0)
    - hypertension (Integer: 0 or 1)
    - heart_disease (Integer: 0 or 1)
    - smoking_history (String: categories will be provided in examples)
    - bmi (Float: typical range 15.0-60.0)
    - HbA1c_level (Float: typical range 4.0-9.0)
    - blood_glucose_level (Integer: typical range 70-300)
    - diabetes (Integer: 0 or 1)

    Examples will be provided in the following format:
    1. Female,45.2,1,0,never,28.5,6.2,140,0
    2. Male,62.7,1,1,former,32.1,7.1,185,1
    3. Female,38.9,0,0,current,24.3,5.8,130,0
    4. Female,22.0,0,0,never,25.77,4.0,145,0
    5. Male,58.0,0,0,former,36.53,5.8,160,0
    6. Male,11.0,0,0,No Info,27.59,6.6,100,0

    Generate 100 comma-separated records, one per line, maintaining realistic correlations between features.
    """,
    # Step 2: Prompt with Definitions
    "PROMPT_2": """
    Generate 100 realistic synthetic patient records for diabetes prediction. Here are the features with definitions:

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
    4. Female,22.0,0,0,never,25.77,4.0,145,0
    5. Male,58.0,0,0,former,36.53,5.8,160,0
    6. Male,11.0,0,0,No Info,27.59,6.6,100,0

    Generate 100 comma-separated records, one per line, maintaining realistic correlations between features.
    """,
    # Step 3: Prompt with Definitions and Metadata
    "PROMPT_3": """
    Generate 100 realistic synthetic patient records for diabetes prediction. Here are the features with definitions and statistical metadata:

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

    Examples from real data:
    1. Female,45.2,1,0,never,28.5,6.2,140,0
    2. Male,62.7,1,1,former,32.1,7.1,185,1
    3. Female,38.9,0,0,current,24.3,5.8,130,0
    4. Female,22.0,0,0,never,25.77,4.0,145,0
    5. Male,58.0,0,0,former,36.53,5.8,160,0
    6. Male,11.0,0,0,No Info,27.59,6.6,100,0

    Generate 100 comma-separated records, one per line, maintaining realistic correlations between features.
    """,
    # Step 4: Prompt with only Definitions and Metadata (No Examples)
    "PROMPT_4": """
    Generate 100 realistic synthetic patient records for diabetes prediction. Here are the features with definitions and statistical metadata:

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

    Generate 100 comma-separated records, one per line, maintaining these relationships and statistical properties.
    """,
}

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
        if not (0 <= float(age) <= 100):  # Changed from 18-80 to match new prompts
            return False
        if int(hyp) not in [0, 1]:
            return False
        if int(heart) not in [0, 1]:
            return False
        if smoking not in [
            "never",
            "former",
            "current",
            "not current",
            "No Info",
        ]:  # Added 'No Info' from examples
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
    # Split into lines and find lines that might be records
    lines = text.split("\n")
    potential_records = [line.strip() for line in lines if "," in line]

    # Validate each potential record
    valid_records = [record for record in potential_records if validate_record(record)]
    return valid_records


def generate_dataset(
    model, tokenizer, prompt, num_records=100, batch_size=BATCH_SIZE, seed=None
):
    """Generate dataset with batching, improved handling and experiment-specific seed"""
    if seed is not None:
        set_seed(seed)

    all_records = []
    pbar = tqdm(total=num_records, desc="Generating records")
    max_attempts = 3

    def clear_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    while len(all_records) < num_records:
        current_batch_size = min(batch_size, num_records - len(all_records))

        for attempt in range(max_attempts):
            try:
                clear_memory()  # Clear memory before generation

                # Adjust max tokens for PROMPT_3 which may need more output tokens
                if (
                    "statistical metadata" in prompt.lower()
                ):  # For PROMPT_3 and PROMPT_4
                    max_tokens = (
                        MAX_NEW_TOKENS * 2
                    )  # Double tokens for statistical prompts
                else:
                    max_tokens = MAX_NEW_TOKENS

                # Prepare input - duplicate prompt for batch processing
                batch_prompts = [prompt] * current_batch_size
                inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(
                    DEVICE
                )

                # Generate in batch
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                # Process each output in the batch
                for output in outputs:
                    # Decode and extract records
                    generated_text = tokenizer.decode(output, skip_special_tokens=True)
                    new_records = extract_records(generated_text)

                    # Add new valid records
                    for record in new_records:
                        if len(all_records) < num_records:
                            all_records.append(record)
                            pbar.update(1)
                        else:
                            break

                    # If we have enough records, stop processing this batch
                    if len(all_records) >= num_records:
                        break

                break  # Success, break attempt loop

            except RuntimeError as e:
                if "out of memory" in str(e) and attempt < max_attempts - 1:
                    print(f"OOM error, attempt {attempt + 1}/{max_attempts}")
                    clear_memory()
                    # Reduce batch size if we encounter an OOM error
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"Reduced batch size to {current_batch_size}")
                    continue
                else:
                    print(f"Error in generation: {str(e)}")
                    # Reduce batch size permanently if we keep having issues
                    batch_size = max(1, batch_size // 2)
                    print(f"Permanently reduced batch size to {batch_size}")
                    break
            except Exception as e:
                print(f"Error in generation: {str(e)}")
                break

    pbar.close()
    return all_records


def save_and_validate_dataset(records, prompt_name, experiment_num):
    """Save dataset and perform basic validation with experiment number"""
    # Create experiment-specific directory
    experiment_dir = os.path.join(OUTPUT_DIR, f"experiment_{experiment_num}")
    os.makedirs(experiment_dir, exist_ok=True)

    output_path = os.path.join(experiment_dir, f"{prompt_name}_data.csv")

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
    print(f"\nDataset statistics for {prompt_name} (Experiment {experiment_num}):")
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


def get_system_info():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    system_info = {
        "cpu_percent": cpu_percent,
        "memory_total_gb": memory.total / (1024**3),
        "memory_used_gb": memory.used / (1024**3),
        "memory_percent": memory.percent,
    }

    # Add GPU information if available
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        system_info.update(
            {
                "gpu_name": gpu.name,
                "gpu_memory_total_mb": gpu.memoryTotal,
                "gpu_memory_used_mb": gpu.memoryUsed,
                "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                "gpu_temperature": gpu.temperature,
            }
        )

    return system_info


def log_experiment_metrics(
    experiment_num,
    prompt_name,
    start_time,
    end_time,
    system_info,
    num_records,
    output_dir,
):
    """Log experiment metrics to JSON"""
    metrics = {
        "experiment_number": experiment_num,
        "prompt_name": prompt_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": (end_time - start_time),
        "records_generated": num_records,
        "records_per_second": num_records / (end_time - start_time),
        "system_info": system_info,
    }

    # Create metrics directory if it doesn't exist
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Save metrics to JSON file
    metrics_file = os.path.join(
        metrics_dir, f"metrics_exp{experiment_num}_{prompt_name}.json"
    )
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def main():
    print(f"Loading model {MODEL_NAME} on {DEVICE}")

    # Load model and tokenizer with 8-bit quantization using BitsAndBytesConfig
    from transformers import BitsAndBytesConfig

    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, padding_side="left"  # Set padding side
    )

    # Handle pad token for Mistral tokenizer
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for Mistral

    # Store all experiment metrics
    all_metrics = []

    # Run multiple experiments with different seeds
    base_seed = int(time.time())

    for experiment_num in range(1, NUM_EXPERIMENTS + 1):
        experiment_seed = base_seed + experiment_num
        print(
            f"\n=== Starting Experiment {experiment_num} (Seed: {experiment_seed}) ==="
        )

        for prompt_name, prompt_text in PROMPTS.items():
            print(
                f"\n=== Generating for: {prompt_name} (Experiment {experiment_num}) ==="
            )

            # Record start time and initial system info
            start_time = time.time()
            initial_system_info = get_system_info()

            # Generate dataset
            records = generate_dataset(
                model,
                tokenizer,
                prompt_text,
                num_records=NUM_RECORDS,
                batch_size=BATCH_SIZE,
                seed=experiment_seed,
            )

            # Record end time and final system info
            end_time = time.time()
            final_system_info = get_system_info()

            # Save dataset
            save_and_validate_dataset(records, prompt_name, experiment_num)

            # Log metrics
            metrics = log_experiment_metrics(
                experiment_num,
                prompt_name,
                start_time,
                end_time,
                {"initial": initial_system_info, "final": final_system_info},
                len(records),
                OUTPUT_DIR,
            )

            all_metrics.append(metrics)

            print(
                f"Completed generation for {prompt_name} (Experiment {experiment_num})"
            )
            print(f"Generation time: {metrics['duration_seconds']:.2f} seconds")
            print(f"Records per second: {metrics['records_per_second']:.2f}")

        print(f"\n=== Completed Experiment {experiment_num} ===")

    # Save summary of all experiments
    summary_file = os.path.join(OUTPUT_DIR, "metrics", "experiments_summary.json")
    with open(summary_file, "w") as f:
        json.dump(
            {
                "total_experiments": NUM_EXPERIMENTS,
                "model_name": MODEL_NAME,
                "device": DEVICE,
                "batch_size": BATCH_SIZE,
                "records_per_prompt": NUM_RECORDS,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": all_metrics,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
