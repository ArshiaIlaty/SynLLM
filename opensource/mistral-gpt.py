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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

"""
IMPORTANT: This code is optimized for NVIDIA A100 GPUs with MIG configuration.
It uses specific memory management techniques and model loading approaches that
are designed for enterprise-grade GPUs.

Compatibility:
- Optimized for: NVIDIA A100 GPUs
- Not compatible with: Consumer GPUs like GTX 1080 Ti
- Requires: CUDA 12.x, PyTorch 2.x, transformers 4.x

If running on consumer-grade GPUs, consider:
1. Removing the MIG-specific configurations
2. Using smaller models or stronger quantization
3. Modifying memory management approaches
"""

# Add this to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Add explicit device selection for MIG environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "mistral-100-prompt-in-5experiments"
BATCH_SIZE = 1  # Smaller batch size to reduce memory usage
NUM_BATCHES = 100  # Adjusted to maintain total count
MAX_NEW_TOKENS = 200  # Reduced token generation to save memory
NUM_EXPERIMENTS = 5  # Number of experiments to run

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def check_hardware_compatibility():
    """Check if running on compatible hardware"""
    if not torch.cuda.is_available():
        print(
            "WARNING: No CUDA device available. This code is optimized for NVIDIA A100 GPUs."
        )
        return False

    try:
        device_name = torch.cuda.get_device_name(0).lower()
        if "a100" in device_name:
            print(f"Compatible hardware detected: {device_name}")
            return True
        else:
            print(
                f"WARNING: Running on {device_name}. This code is optimized for NVIDIA A100 GPUs and may not work correctly."
            )
            return False
    except Exception as e:
        print(f"Error checking hardware compatibility: {e}")
        return False


def find_least_used_gpu():
    """Find the GPU with the most available memory with better error handling for MIG"""
    try:
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")

        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("No CUDA devices found, falling back to CPU")
            return -1

        if device_count == 1:
            print(f"Only one CUDA device available, using device 0")
            return 0

        # Find the GPU with the most available memory
        best_gpu = 0
        max_free_memory = 0

        for i in range(device_count):
            try:
                # Get free memory for this device
                free_memory = torch.cuda.get_device_properties(
                    i
                ).total_memory - torch.cuda.memory_allocated(i)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu = i
            except Exception as e:
                print(f"Warning: Error checking GPU {i}: {e}")
                continue

        print(
            f"Selected GPU {best_gpu} with {max_free_memory / 1024**2:.2f} MiB free memory"
        )
        return best_gpu
    except Exception as e:
        print(f"Error finding GPUs: {e}")
        return -1


def clean_gpu_memory():
    """Clean GPU memory and cache with more robust error handling"""
    try:
        if torch.cuda.is_available():
            # Empty cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Garbage collection
            gc.collect()

            # Print available memory for each device
            try:
                for i in range(torch.cuda.device_count()):
                    free_memory = torch.cuda.get_device_properties(
                        i
                    ).total_memory - torch.cuda.memory_allocated(i)
                    print(f"GPU {i}: {free_memory / 1024**2:.2f} MiB free")
            except Exception as e:
                print(f"Warning: Could not get GPU memory info: {e}")

    except Exception as e:
        print(f"Warning: Error cleaning GPU memory: {e}")
        # Still do garbage collection even if GPU operations fail
        gc.collect()


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
        # Convert to float first to handle decimal points in glucose value
        if not (70 <= int(float(glucose)) <= 300):
            return False
        if int(diabetes) not in [0, 1]:
            return False

        return True
    except Exception as e:
        # Print detailed error for debugging
        # print(f"Validation error: {e} for record: {record}")
        return False


def extract_records(text):
    """Extract valid records from generated text"""
    # Split into lines and find lines that might be records
    lines = text.split("\n")
    potential_records = [line.strip() for line in lines if "," in line]

    # Validate each potential record
    valid_records = [record for record in potential_records if validate_record(record)]

    # Print some debug info
    if valid_records:
        print(
            f"Successfully extracted {len(valid_records)} valid records from generated text"
        )
    elif potential_records:
        print(f"Found {len(potential_records)} potential records but none were valid")
        # Show a sample of invalid records for debugging
        if len(potential_records) > 0:
            print(f"Sample invalid record: {potential_records[0]}")

    return valid_records


def generate_dataset(tokenizer, model, prompt, num_records=100, seed=None):
    """Generate dataset using direct model access instead of pipeline"""
    if seed is not None:
        set_seed(seed)

    all_records = []
    pbar = tqdm(total=num_records, desc="Generating records")

    # Track unique records
    unique_records = set()

    attempts = 0
    max_attempts = num_records * 3  # Allow more attempts than needed records

    while len(all_records) < num_records and attempts < max_attempts:
        attempts += 1

        try:
            clean_gpu_memory()

            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate text directly with model
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=1,
                )

            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the newly generated part (after the prompt)
            if prompt in generated_text:
                response_text = generated_text[len(prompt) :]
            else:
                response_text = generated_text

            # Debug - print a preview of the generated text
            print(
                f"\nGenerated text preview (first 200 chars):\n{response_text[:200]}...\n"
            )

            # Extract records
            new_records = extract_records(response_text)
            print(f"Found {len(new_records)} records in attempt {attempts}")

            # Add valid records
            for record in new_records:
                if record not in unique_records and len(all_records) < num_records:
                    unique_records.add(record)
                    all_records.append(record)
                    pbar.update(1)

            # If we didn't find any records, wait a bit before trying again
            if not new_records:
                time.sleep(1)

        except Exception as e:
            print(f"Error in generation: {e}")
            time.sleep(2)

        # If we got some records, print them for visibility
        if len(all_records) > 0 and len(all_records) % 5 == 0:
            print(f"Progress: {len(all_records)}/{num_records} records")
            print(f"Latest record: {all_records[-1]}")

    pbar.close()

    # Final report
    print(
        f"\nGeneration complete: collected {len(all_records)} records in {attempts} attempts"
    )

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

    # Safety check
    if not records:
        print(
            f"Warning: No records to save for {prompt_name} (Experiment {experiment_num})"
        )
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=columns)
    else:
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

    if not df.empty:
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
    """Get current system resource usage with better error handling for MIG"""
    system_info = {}

    try:
        # Get CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        system_info.update(
            {
                "cpu_percent": cpu_percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
            }
        )
    except Exception as e:
        print(f"Warning: Error getting CPU info: {e}")
        system_info["cpu_error"] = str(e)

    # Add GPU information if available - with careful error handling for MIG
    try:
        if torch.cuda.is_available():
            # Add CUDA info from torch directly
            system_info.update(
                {
                    "cuda_available": True,
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_current_device": torch.cuda.current_device(),
                }
            )

            # Try to get device name and memory info
            try:
                system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
                system_info["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated(
                    0
                ) / (1024**3)
                system_info["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved(
                    0
                ) / (1024**3)
            except Exception as e:
                system_info["cuda_device_info_error"] = str(e)

            # Try GPUtil as fallback
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info = []
                    for i, gpu in enumerate(gpus):
                        gpu_info.append(
                            {
                                "gpu_id": i,
                                "gpu_name": gpu.name,
                                "gpu_memory_total_mb": gpu.memoryTotal,
                                "gpu_memory_used_mb": gpu.memoryUsed,
                                "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal)
                                * 100,
                                "gpu_temperature": gpu.temperature,
                            }
                        )
                    system_info["gpus"] = gpu_info
            except Exception as e:
                system_info["gputil_error"] = str(e)
    except Exception as e:
        print(f"Warning: Error getting GPU info: {e}")
        system_info["gpu_error"] = str(e)

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
        "records_per_second": num_records / max(1, (end_time - start_time)),
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
    print(f"Starting Mistral data generation experiment")
    print(f"Output directory: {OUTPUT_DIR}")

    # Check hardware compatibility
    check_hardware_compatibility()

    # Select the best GPU to use
    gpu_id = find_least_used_gpu()

    # Set device based on GPU detection
    device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"

    print(f"Using device: {device}")

    try:
        # Initialize model with adaptations for MIG
        print(f"Loading model {MODEL_NAME}")

        # Try loading with 4-bit quantization first (better for MIG memory constraints)
        try:
            print("Attempting to load with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model without device_map="auto" for MIG compatibility
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                # Avoid device_map="auto" for MIG
                device_map={"": 0},  # Explicitly map to first device
            )

            print("Successfully loaded model with 4-bit quantization")

        except Exception as e:
            print(f"Error loading with 4-bit quantization: {e}")

            print("Falling back to 8-bit quantization...")
            try:
                # Try 8-bit as fallback
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )

                # Load tokenizer first
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map={"": 0},  # Explicitly map to first device
                )

                print("Successfully loaded model with 8-bit quantization")

            except Exception as e2:
                print(f"Error loading with 8-bit quantization: {e2}")

                print("Falling back to regular model loading with explicit device...")
                # Final fallback: regular loading to specific device
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,
                ).to(device)

                print(f"Model loaded on {device} without quantization")

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

                # Generate dataset with appropriate number of records
                target_records = BATCH_SIZE * NUM_BATCHES
                records = generate_dataset(
                    tokenizer,
                    model,
                    prompt_text,
                    num_records=target_records,
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

                # Additional cleanup between prompts for memory management
                clean_gpu_memory()
                time.sleep(3)  # Extra wait time for memory to fully clear

            print(f"\n=== Completed Experiment {experiment_num} ===")

        # Save summary of all experiments
        summary_file = os.path.join(OUTPUT_DIR, "metrics", "experiments_summary.json")
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "total_experiments": NUM_EXPERIMENTS,
                    "model_name": MODEL_NAME,
                    "device": device,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": all_metrics,
                },
                f,
                indent=4,
            )

        print(f"\n=== All experiments completed ===")
        print(f"Results saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"Critical error in main function: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
