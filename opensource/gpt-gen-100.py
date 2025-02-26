import gc
import json
import os
import re

# from prompts import PROMPTS
import time
from datetime import datetime

import GPUtil
import pandas as pd
import psutil
import torch
from tqdm import tqdm
from transformers import pipeline, set_seed

# Configuration
MODEL_NAME = "gpt2"
OUTPUT_DIR = "gpt2-100-prompt-in-5experiments"
BATCH_SIZE = 10  # Reduced batch size for better stability
NUM_BATCHES = 10  # Increased number of batches to maintain total count
DEVICE = 0 if torch.cuda.is_available() else -1
MAX_NEW_TOKENS = 200  # Specific tokens to generate
NUM_EXPERIMENTS = 5  # Number of experiments to run

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define prompts dictionary
PROMPTS = {
    # Step 1: Basic Prompt with Examples
    "PROMPT_1": """
    Generate synthetic diabetes data in CSV format:
    gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

    Examples:
    Female,45.2,0,0,never,28.5,5.7,155,0
    Male,62.7,1,1,former,32.1,6.8,185,1
    Female,38.9,0,0,current,24.3,5.2,130,0

    Generate more records following these rules:
    - gender: Male or Female
    - age: between 18 and 80
    - hypertension: 0 or 1
    - heart_disease: 0 or 1
    - smoking_history: never, former, current, or not current
    - bmi: between 15 and 60
    - HbA1c_level: between 4 and 9
    - blood_glucose_level: between 70 and 300
    - diabetes: 0 or 1
    """,
    # Step 2: Prompt with Definitions
    "PROMPT_2": """
    Generate synthetic diabetes data with these definitions:
    gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

    Definitions:
    - gender: Patient's biological sex (Male/Female)
    - age: Age in years (18-80)
    - hypertension: High blood pressure diagnosis (0=No, 1=Yes)
    - heart_disease: Heart disease diagnosis (0=No, 1=Yes)
    - smoking_history: Smoking status (never/former/current/not current)
    - bmi: Body Mass Index (15-60)
    - HbA1c_level: Average blood sugar level (4-9)
    - blood_glucose_level: Current blood glucose (70-300)
    - diabetes: Diabetes diagnosis (0=No, 1=Yes)

    Examples:
    Female,45.2,0,0,never,28.5,5.7,155,0
    Male,62.7,1,1,former,32.1,6.8,185,1
    Female,38.9,0,0,current,24.3,5.2,130,0
    """,
    # Step 3: Prompt with Metadata
    "PROMPT_3": """
    Generate synthetic diabetes data with these statistics:
    gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

    Statistical properties:
    - gender: Male (48%), Female (52%)
    - age: Mean=41.8, range 18-80
    - hypertension: No (85%), Yes (15%)
    - heart_disease: No (92%), Yes (8%)
    - smoking_history: never (60%), former (22%), current (15%), not current (3%)
    - bmi: Mean=27.3, range 15-60
    - HbA1c_level: Mean=5.7, range 4-9
    - blood_glucose_level: Mean=138, range 70-300
    - diabetes: No (88%), Yes (12%)

    Examples:
    Female,45.2,0,0,never,28.5,5.7,155,0
    Male,62.7,1,1,former,32.1,6.8,185,1
    Female,38.9,0,0,current,24.3,5.2,130,0
    """,
    # Step 4: Prompt with Rules
    "PROMPT_4": """
    Generate synthetic diabetes data following these rules:
    gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

    Rules:
    1. Higher HbA1c (>6.5) usually means diabetes=1
    2. Higher blood glucose (>180) usually means diabetes=1
    3. Older age increases hypertension and heart_disease risk
    4. Higher BMI (>30) increases diabetes risk
    5. Must follow these ranges:
       - gender: Male or Female only
       - age: 18-80 only
       - hypertension: 0 or 1 only
       - heart_disease: 0 or 1 only
       - smoking_history: never/former/current/not current only
       - bmi: 15-60 only
       - HbA1c_level: 4-9 only
       - blood_glucose_level: 70-300 only
       - diabetes: 0 or 1 only

    Examples:
    Female,45.2,0,0,never,28.5,5.7,155,0
    Male,62.7,1,1,former,32.1,6.8,185,1
    Female,38.9,0,0,current,24.3,5.2,130,0
    """,
}


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


def generate_dataset(generator, prompt, num_records=100, seed=None):
    """Generate dataset with improved handling and experiment-specific seed"""
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
        for attempt in range(max_attempts):
            try:
                clear_memory()  # Clear memory before generation

                # Generate text with smaller chunks for PROMPT_3
                if "Statistical properties:" in prompt:  # PROMPT_3 detection
                    max_tokens = MAX_NEW_TOKENS * 2  # Double tokens for PROMPT_3
                else:
                    max_tokens = MAX_NEW_TOKENS

                output = generator(
                    prompt,
                    max_new_tokens=max_tokens,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=50256,
                )

                # Extract and validate records
                new_records = extract_records(output[0]["generated_text"])

                # Add new valid records
                for record in new_records:
                    if len(all_records) < num_records:
                        all_records.append(record)
                        pbar.update(1)
                    else:
                        break

                break  # Success, break attempt loop

            except RuntimeError as e:
                if "out of memory" in str(e) and attempt < max_attempts - 1:
                    print(f"OOM error, attempt {attempt + 1}/{max_attempts}")
                    clear_memory()
                    continue
                else:
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
    # Initialize generator
    print(f"Initializing generator with {MODEL_NAME} on device {DEVICE}")
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE >= 0 else torch.float32,
    )

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
            records = generate_dataset(generator, prompt_text, seed=experiment_seed)

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
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": all_metrics,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
