import json
import os
import re
import time
from datetime import datetime

import GPUtil
import pandas as pd
import psutil
import torch
from prompts import PROMPTS
from tqdm import tqdm
from transformers import pipeline, set_seed

# Configuration
MODEL_NAME = "gpt2"
OUTPUT_DIR = "gpt2-5experiments-prompts"
BATCH_SIZE = 10  # Reduced batch size for better stability
NUM_BATCHES = 100  # Increased number of batches to maintain total count
DEVICE = 0 if torch.cuda.is_available() else -1
MAX_NEW_TOKENS = 200  # Specific tokens to generate
NUM_EXPERIMENTS = 5  # Number of experiments to run
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


def generate_dataset(generator, prompt, num_records=1000, seed=None):
    """Generate dataset with improved handling and experiment-specific seed"""
    if seed is not None:
        set_seed(seed)

    all_records = []
    pbar = tqdm(total=num_records, desc="Generating records")

    while len(all_records) < num_records:
        try:
            # Generate text
            output = generator(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
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

        except Exception as e:
            print(f"Error in generation: {str(e)}")
            continue

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
