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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed

# Fix the environment variable name (max_split_size_mb not max_split_mb)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Allow all GPUs to be visible
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Configuration
MODEL_NAME = "gpt2"
OUTPUT_DIR = "gpt2-breast-100-prompt-in-5experiments"
MAX_NEW_TOKENS = 500  # Tokens to generate
NUM_EXPERIMENTS = 5  # Number of experiments to run
# How many GPUs to use (up to 4 on your system)
NUM_GPUS_TO_USE = 4


# Check CUDA availability - using a safer approach
def initialize_devices():
    devices = ["cpu"]  # Always include CPU as fallback

    try:
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.is_available()}")
            num_gpus = torch.cuda.device_count()
            print(f"CUDA device count: {num_gpus}")

            # Test each GPU
            available_gpus = []
            for i in range(min(num_gpus, NUM_GPUS_TO_USE)):
                try:
                    # Test CUDA with a simple tensor operation
                    device_name = f"cuda:{i}"
                    test_tensor = torch.zeros(1, device=device_name)
                    test_tensor = test_tensor + 1  # Simple operation
                    print(f"CUDA device {i} initialization successful")
                    available_gpus.append(device_name)
                except Exception as e:
                    print(f"Error initializing CUDA device {i}: {e}")

            if available_gpus:
                devices = available_gpus + devices  # Add CPU as fallback
                print(f"Using {len(available_gpus)} GPU(s): {available_gpus}")
            else:
                print("No GPUs could be initialized, using CPU")
        else:
            print("CUDA not available, using CPU")
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        print("Falling back to CPU")

    return devices


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define prompts dictionary
PROMPTS = {
    # Step 1: Basic Prompt with Examples
    "PROMPT_1": """
    I need you to generate synthetic breast-cancer data that closely resembles real-world data. The dataset should contain 100 samples with the following columns:

    id, diagnosis (M/B), radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean,  fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst, Unnamed: 32

    Here are 3 example records from a real dataset to guide your generation:

    Example 1:
    9110732, M, 17.75, 28.03, 117.3, 981.6, 0.09997, 0.1314, 0.1698, 0.08293, 0.1713, 0.05916, 0.3897, 1.077, 2.873, 43.95, 0.004714, 0.02015, 0.03697, 0.0111, 0.01237, 0.002556, 21.53, 38.54, 145.4, 1437, 0.1401, 0.3762, 0.6399, 0.197, 0.2972, 0.09075, 0

    Example 2:
    8911670, M, 18.81, 19.98, 120.9, 1102, 0.08923, 0.05884, 0.0802, 0.05843, 0.155, 0.04996, 0.3283, 0.828, 2.363, 36.74, 0.007571, 0.01114, 0.02623, 0.01463, 0.0193, 0.001676, 19.96, 24.3, 129, 1236, 0.1243, 0.116, 0.221, 0.1294, 0.2567, 0.05737, 0


    Please generate 100 records in a CSV format that follows these patterns and maintains realistic relationships between the features. The data should be plausible and preserve the correlations between features that would be found in real breast-cancer data.
    """
}


def extract_records(text):
    """Extract records using regex patterns"""
    # This regex looks for lines with numbers and commas in the expected format
    pattern = re.compile(
        r"\d+,\s*[MB],\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+"
    )
    matches = pattern.findall(text)
    return matches


def verify_dataset_format(records):
    """Verify that records match expected format and clean them"""
    verified_records = []
    for record in records:
        try:
            # Split the record into its components
            parts = record.strip().split(",")
            # Must have expected number of fields
            if len(parts) == 32:
                # Check ID (should be numeric)
                if not parts[0].strip().isdigit():
                    continue
                # Check diagnosis (should be M or B)
                if parts[1].strip() not in ["M", "B"]:
                    continue
                # Try to convert all numerical values to float
                try:
                    values = [float(part.strip()) for part in parts[2:]]
                    # All checks passed, record is valid
                    verified_records.append(record)
                except ValueError:
                    # Some values couldn't be converted to float
                    continue
        except Exception as e:
            # Skip any record that triggers an exception
            continue

    return verified_records


def clear_memory():
    """Clear GPU and system memory"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: Memory clearing issue: {e}")
    gc.collect()


def generate_text_directly(
    model, tokenizer, prompt, device, max_new_tokens=500, seed=None
):
    """
    Generate text directly using the model and tokenizer instead of pipeline
    to avoid CUDA indexing errors
    """
    if seed is not None:
        set_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Add attention mask to avoid warning
    encoded = tokenizer.encode_plus(
        prompt, return_tensors="pt", padding=True, max_length=1024, truncation=True
    )

    # Handle device placement carefully
    try:
        # Move inputs to device
        if "cuda" in device:
            try:
                inputs = {k: v.to(device) for k, v in encoded.items()}
            except Exception as e:
                print(f"Error moving inputs to GPU {device}: {e}")
                print("Falling back to CPU")
                model = model.cpu()
                device = "cpu"
                inputs = encoded
        else:
            inputs = encoded

        # Generate with careful error handling
        try:
            # Set smaller max_length to avoid exceeding model's limit
            with torch.no_grad():
                output = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=min(1024, inputs["input_ids"].shape[1] + max_new_tokens),
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                )

            # Decode the output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            return generated_text, device

        except Exception as e:
            print(f"Error in generation: {e}")
            # If we get a CUDA error, try again on CPU
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"Error with {device}, falling back to CPU")
                model = model.cpu()
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].cpu()
                device = "cpu"

                with torch.no_grad():
                    output = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=min(
                            1024, inputs["input_ids"].shape[1] + max_new_tokens
                        ),
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1,
                    )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                return generated_text, device
            else:
                raise e

    except Exception as e:
        print(f"Unexpected error: {e}")
        return prompt, "cpu"  # Return at least the prompt on failure


def generate_breast_cancer_records(
    models, tokenizer, prompt, devices, num_records=100, seed=None
):
    """Generate breast cancer dataset using multiple models/devices in parallel"""
    if seed is not None:
        set_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    all_records = []
    pbar = tqdm(total=num_records, desc="Generating records")

    # Track which devices are still working
    active_devices = list(range(len(devices)))

    while len(all_records) < num_records and active_devices:
        # Round-robin through active devices
        for i in active_devices.copy():
            if len(all_records) >= num_records:
                break

            try:
                device = devices[i]
                model = models[i]

                # Generate text with the model
                generated_text, new_device = generate_text_directly(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    max_new_tokens=MAX_NEW_TOKENS,
                    seed=seed + i if seed else None,  # Different seed per device
                )

                # Update device if it changed
                if new_device != device:
                    devices[i] = new_device
                    if new_device == "cpu" and "cuda" in device:
                        print(f"Device {i} ({device}) is now using CPU")

                # Extract records
                new_records = extract_records(generated_text)

                # Verify and clean records
                verified_records = verify_dataset_format(new_records)

                # Add verified records
                for record in verified_records:
                    if len(all_records) < num_records:
                        all_records.append(record)
                        pbar.update(1)
                    else:
                        break

                # Print informative message if we got records
                if verified_records:
                    print(
                        f"Found {len(verified_records)} valid records from device {i}. Total: {len(all_records)}/{num_records}"
                    )

                # If this device is producing records, keep using it
                if not verified_records:
                    print(f"No valid records from device {i}")

            except Exception as e:
                print(f"Error with device {i}: {str(e)}")
                active_devices.remove(i)
                print(
                    f"Removed device {i} from active devices. Remaining: {active_devices}"
                )

            # Take a short break between generations
            time.sleep(0.1)

            # Clear memory between generations
            clear_memory()

        # If we've gone through all devices but still have no records, sleep a bit
        if len(all_records) == 0 and active_devices:
            time.sleep(1)

    pbar.close()
    return all_records


def save_dataset(records, prompt_name, experiment_num):
    """Save dataset to CSV file"""
    # Create experiment-specific directory
    experiment_dir = os.path.join(OUTPUT_DIR, f"experiment_{experiment_num}")
    os.makedirs(experiment_dir, exist_ok=True)

    output_path = os.path.join(experiment_dir, f"{prompt_name}_data.csv")

    # Create DataFrame with breast cancer dataset columns
    columns = [
        "id",
        "diagnosis",
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
        "Unnamed: 32",
    ]

    # Parse records with proper handling of whitespace
    parsed_records = []
    for record in records:
        parts = [part.strip() for part in record.split(",")]
        if len(parts) == 32:
            parsed_records.append(parts)

    df = pd.DataFrame(parsed_records, columns=columns)

    # Convert numeric columns to proper datatypes
    numeric_columns = columns[2:]  # All columns except id and diagnosis
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print basic statistics
    print(f"\nDataset statistics for {prompt_name} (Experiment {experiment_num}):")
    print(f"Total records: {len(df)}")

    # Print diagnosis distribution
    print("\nDiagnosis distribution:")
    print(df["diagnosis"].value_counts(normalize=True))

    # Print summary of numeric columns
    print("\nNumeric columns summary:")
    print(df.describe())

    return df


def get_system_info():
    """Get current system resource usage - with better error handling for MIG"""
    system_info = {}

    try:
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
        print(f"Warning: Error getting system info: {e}")
        system_info["system_error"] = str(e)

    # Add GPU information if available - with careful error handling
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = {}
                for i, gpu in enumerate(gpus):
                    gpu_info[f"gpu_{i}"] = {
                        "name": gpu.name,
                        "memory_total_mb": gpu.memoryTotal,
                        "memory_used_mb": gpu.memoryUsed,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "temperature": gpu.temperature,
                    }
                system_info["gpus"] = gpu_info
            else:
                system_info["gpu_info"] = "No GPUs found by GPUtil"

            # Add torch.cuda info as fallback
            system_info.update(
                {
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_current_device": torch.cuda.current_device(),
                }
            )
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
    # Initialize devices
    devices = initialize_devices()
    active_devices = devices.copy()

    print(f"Starting GPT-2 breast cancer data generation on devices: {active_devices}")

    try:
        # Initialize model and tokenizer
        print(f"Loading model {MODEL_NAME} and tokenizer")
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load one model per device
        models = []
        for device in devices:
            model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
            if device != "cpu":
                try:
                    model = model.to(device)
                    print(f"Model loaded on {device}")
                except Exception as e:
                    print(f"Error moving model to {device}: {e}")
                    print("Will use CPU for this model")
                    device = "cpu"
            else:
                print("Model loaded on CPU")
            models.append(model)

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

                # Generate dataset using the direct approach with multiple devices
                records = generate_breast_cancer_records(
                    models, tokenizer, prompt_text, devices, seed=experiment_seed
                )

                # Record end time and final system info
                end_time = time.time()
                final_system_info = get_system_info()

                # Save dataset
                save_dataset(records, prompt_name, experiment_num)

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

                # Extra cleanup between prompts
                clear_memory()
                time.sleep(2)

            print(f"\n=== Completed Experiment {experiment_num} ===")

        # Save summary of all experiments
        summary_file = os.path.join(OUTPUT_DIR, "metrics", "experiments_summary.json")
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "total_experiments": NUM_EXPERIMENTS,
                    "model_name": MODEL_NAME,
                    "devices": devices,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": all_metrics,
                },
                f,
                indent=4,
            )

        print(f"\n=== All experiments completed successfully ===")
        print(f"Results saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"Critical error in main function: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

# import gc
# import json
# import os
# import re
# import time
# from datetime import datetime

# import GPUtil
# import pandas as pd
# import psutil
# import torch
# from datasets import Dataset
# from tqdm import tqdm
# from transformers import pipeline, set_seed

# # Explicitly set environment variables for GPU visibility
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust based on your MIG configuration
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# # Configuration
# MODEL_NAME = "gpt2"
# OUTPUT_DIR = "gpt2-breast-100-prompt-in-5experiments"
# BATCH_SIZE = 4  # Batch size for dataset processing
# MAX_NEW_TOKENS = 200  # Specific tokens to generate
# NUM_EXPERIMENTS = 5  # Number of experiments to run

# # Safely initialize device
# try:
#     if torch.cuda.is_available():
#         print(f"CUDA available: {torch.cuda.is_available()}")
#         print(f"CUDA device count: {torch.cuda.device_count()}")
#         test_tensor = torch.zeros(1, device="cuda:0")
#         print("CUDA initialization successful")
#         DEVICE = 0
#     else:
#         print("CUDA not available")
#         DEVICE = -1
# except Exception as e:
#     print(f"Error initializing CUDA: {e}")
#     print("Falling back to CPU")
#     DEVICE = -1

# # Create output directory
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Define prompts dictionary
# PROMPTS = {
#     # Step 1: Basic Prompt with Examples
#     "PROMPT_1": """
#     I need you to generate synthetic breast-cancer data that closely resembles real-world data. The dataset should contain 100 samples with the following columns:

#     id, diagnosis (M/B), radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean,  fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst, Unnamed: 32

#     Here are 3 example records from a real dataset to guide your generation:

#     Example 1:
#     9110732, M, 17.75, 28.03, 117.3, 981.6, 0.09997, 0.1314, 0.1698, 0.08293, 0.1713, 0.05916, 0.3897, 1.077, 2.873, 43.95, 0.004714, 0.02015, 0.03697, 0.0111, 0.01237, 0.002556, 21.53, 38.54, 145.4, 1437, 0.1401, 0.3762, 0.6399, 0.197, 0.2972, 0.09075, 0

#     Example 2:
#     8911670, M, 18.81, 19.98, 120.9, 1102, 0.08923, 0.05884, 0.0802, 0.05843, 0.155, 0.04996, 0.3283, 0.828, 2.363, 36.74, 0.007571, 0.01114, 0.02623, 0.01463, 0.0193, 0.001676, 19.96, 24.3, 129, 1236, 0.1243, 0.116, 0.221, 0.1294, 0.2567, 0.05737, 0


#     Please generate 100 records in a CSV format that follows these patterns and maintains realistic relationships between the features. The data should be plausible and preserve the correlations between features that would be found in real breast-cancer data.
#     """,
#     # Step 2: Prompt with Definitions
#     "PROMPT_2": """
#     I need you to generate synthetic breast-cancer data. Please create 100 samples that realistically represent the patterns and relationships in this type of data.

#     Generate breast-cancer data:
#     ID, diagnosis (M: 37.3%, B: 62.7%), followed by 30 numerical features.

#     Here are 3 example records from a real dataset:

#     Example 1 (M):
#     85638502, M, 13.17, 21.81, 85.42, 531.5, 0.09714, 0.1047, 0.08259, 0.05252, 0.1746, 0.06177, 0.1938, 0.6123, 1.334, 14.49, 0.00335, 0.01384, 0.01452, 0.006853, 0.01113, 0.00172, 16.23, 29.89, 105.5, 740.7, 0.1503, 0.3904, 0.3728, 0.1607, 0.3693, 0.09618, 0

#     Example 2 (M):
#     842517, M, 20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 24.99, 23.41, 158.8, 1956, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902, 0

#     Example 3 (B):
#     91544002, B, 11.06, 17.12, 71.25, 366.5, 0.1194, 0.1071, 0.04063, 0.04268, 0.1954, 0.07976, 0.1779, 1.03, 1.318, 12.3, 0.01262, 0.02348, 0.018, 0.01285, 0.0222, 0.008313, 11.69, 20.74, 76.08, 411.1, 0.1662, 0.2031, 0.1256, 0.09514, 0.278, 0.1168, 0

#     Please provide 100 synthetic records in CSV format, with values that are plausible and maintain the natural relationships between features.
#     """,
#     # Step 3: Prompt with Metadata
#     "PROMPT_3": """
#     I need you to generate synthetic breast-cancer data based on real statistical properties. Please generate 100 records that accurately represent the data while maintaining the statistical properties and correlations found in real data.

#     Key correlations to maintain:
#     - Strong positive correlation among radius_mean, perimeter_worst, area_mean, radius_worst, perimeter_mean
#     - Moderate positive correlation among concavity_se, concave points_worst, radius_worst, perimeter_mean, concave points_se
#     - diagnosis class differences: B and M samples show significant differences in concavity_mean, area_se, concave points_mean, concavity_worst, area_worst

#     Here are 3 example records from the real dataset:

#     Example 1 (M):
#     885429, M, 19.73, 19.82, 130.7, 1206, 0.1062, 0.1849, 0.2417, 0.0974, 0.1733, 0.06697, 0.7661, 0.78, 4.115, 92.81, 0.008482, 0.05057, 0.068, 0.01971, 0.01467, 0.007259, 25.28, 25.59, 159.8, 1933, 0.171, 0.5955, 0.8489, 0.2507, 0.2749, 0.1297, 0

#     Example 2 (M):
#     9012000, M, 22.01, 21.9, 147.2, 1482, 0.1063, 0.1954, 0.2448, 0.1501, 0.1824, 0.0614, 1.008, 0.6999, 7.561, 130.2, 0.003978, 0.02821, 0.03576, 0.01471, 0.01518, 0.003796, 27.66, 25.8, 195, 2227, 0.1294, 0.3885, 0.4756, 0.2432, 0.2741, 0.08574, 0

#     Please provide 100 synthetic records in CSV format, with values that are plausible and maintain both the statistical properties and natural relationships between features. Ensure the data could be useful for machine learning algorithms to differentiate between different diagnosis values.
#     """,
#     # Step 4: Prompt with Rules
#     "PROMPT_4": """
#     I need you to generate synthetic breast-cancer data based exclusively on statistical properties without using any real examples. Please create 100 synthetic records that represent breast-cancer measurements while preserving the statistical distributions and correlations in the real-world data.


#     Please provide 100 synthetic records in CSV format that satisfy these statistical properties and mathematical relationships. The synthetic data should be suitable for analysis or training machine learning models while preserving privacy by not containing any actual data points.
#     """,
# }


# def validate_record(record):
#     """Validate if a generated record matches expected format and constraints for breast cancer dataset"""
#     try:
#         parts = record.strip().split(",")
#         # Breast cancer dataset has 32 columns (id, diagnosis, 30 features)
#         if len(parts) != 32:
#             return False

#         # Extract key fields
#         id_val, diagnosis = parts[0], parts[1]

#         # Basic format checks
#         if not id_val.strip().isdigit():  # ID should be numeric
#             return False

#         if diagnosis.strip() not in ["M", "B"]:  # Diagnosis should be M or B
#             return False

#         # Convert all feature values to float for validation
#         try:
#             values = [float(p.strip()) for p in parts[2:]]
#         except ValueError:
#             return False

#         # Check ranges for key features
#         # Radius mean (index 2)
#         if not (6.5 <= values[0] <= 29):
#             return False

#         # Area mean (index 5-2=3)
#         if not (140 <= values[3] <= 2600):
#             return False

#         # Check relationships between related features
#         radius_mean, perimeter_mean, area_mean = values[0], values[2], values[3]

#         # Check if perimeter and area are reasonably proportional to radius
#         # Perimeter should be roughly 2πr
#         if not (5.5 * radius_mean <= perimeter_mean <= 7 * radius_mean):
#             return False

#         # Area should be roughly πr²
#         if not (2.5 * radius_mean**2 <= area_mean <= 4 * radius_mean**2):
#             return False

#         # Check "worst" values are >= corresponding "mean" values
#         if values[20] < values[0]:  # radius_worst < radius_mean
#             return False

#         if values[23] < values[3]:  # area_worst < area_mean
#             return False

#         # Malignant samples typically have higher values for certain features
#         if diagnosis.strip() == "M" and values[6] < 0.04:  # concavity_mean too low for M
#             return False

#         # Check Unnamed: 32 is always 0
#         if values[29] != 0:
#             return False

#         return True
#     except Exception as e:
#         # print(f"Validation error: {e} for record: {record}")
#         return False


# def extract_records(text):
#     """Extract valid records from generated text"""
#     # Split into lines and find lines that might be records
#     lines = text.split("\n")
#     potential_records = [line.strip() for line in lines if "," in line]

#     # Validate each potential record
#     valid_records = [record for record in potential_records if validate_record(record)]
#     return valid_records


# def clear_memory():
#     """Clear GPU and system memory"""
#     if torch.cuda.is_available():
#         try:
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()
#         except Exception as e:
#             print(f"Warning: Memory clearing issue: {e}")
#     gc.collect()


# def generate_dataset_efficiently(generator, prompt, num_records=100, seed=None):
#     """Generate dataset with improved batch processing"""
#     if seed is not None:
#         set_seed(seed)

#     all_records = []
#     pbar = tqdm(total=num_records, desc="Generating records")

#     # Create a dataset of multiple prompts to process in batches
#     prompts = [prompt] * 10  # Generate 10 completions at a time
#     dataset = Dataset.from_dict({"prompt": prompts})

#     while len(all_records) < num_records:
#         clear_memory()  # Clear memory before generation

#         try:
#             # Process in batches
#             for batch in dataset.iter(batch_size=BATCH_SIZE):
#                 if len(all_records) >= num_records:
#                     break

#                 outputs = generator(
#                     batch["prompt"],
#                     max_new_tokens=MAX_NEW_TOKENS,
#                     do_sample=True,
#                     temperature=0.7,
#                     pad_token_id=50256,
#                 )

#                 # Process each output in the batch
#                 for output in outputs:
#                     if isinstance(output, dict):
#                         generated_text = output["generated_text"]
#                     else:
#                         generated_text = output[0]["generated_text"]

#                     # Extract and validate records
#                     new_records = extract_records(generated_text)

#                     # Add new valid records
#                     for record in new_records:
#                         if len(all_records) < num_records:
#                             all_records.append(record)
#                             pbar.update(1)
#                         else:
#                             break

#             # If we haven't made progress, wait before trying again
#             if len(all_records) < num_records:
#                 time.sleep(1)

#         except Exception as e:
#             print(f"Error in generation: {str(e)}")
#             time.sleep(3)  # Wait before retrying

#     pbar.close()
#     return all_records


# def save_and_validate_dataset(records, prompt_name, experiment_num):
#     """Save dataset and perform basic validation with experiment number"""
#     # Create experiment-specific directory
#     experiment_dir = os.path.join(OUTPUT_DIR, f"experiment_{experiment_num}")
#     os.makedirs(experiment_dir, exist_ok=True)

#     output_path = os.path.join(experiment_dir, f"{prompt_name}_data.csv")

#     # Create DataFrame with breast cancer dataset columns
#     columns = [
#         "id",
#         "diagnosis",
#         "radius_mean",
#         "texture_mean",
#         "perimeter_mean",
#         "area_mean",
#         "smoothness_mean",
#         "compactness_mean",
#         "concavity_mean",
#         "concave points_mean",
#         "symmetry_mean",
#         "fractal_dimension_mean",
#         "radius_se",
#         "texture_se",
#         "perimeter_se",
#         "area_se",
#         "smoothness_se",
#         "compactness_se",
#         "concavity_se",
#         "concave points_se",
#         "symmetry_se",
#         "fractal_dimension_se",
#         "radius_worst",
#         "texture_worst",
#         "perimeter_worst",
#         "area_worst",
#         "smoothness_worst",
#         "compactness_worst",
#         "concavity_worst",
#         "concave points_worst",
#         "symmetry_worst",
#         "fractal_dimension_worst",
#         "Unnamed: 32"
#     ]

#     # Parse records with proper handling of whitespace
#     parsed_records = []
#     for record in records:
#         parts = [part.strip() for part in record.split(",")]
#         if len(parts) == 32:
#             parsed_records.append(parts)

#     df = pd.DataFrame(parsed_records, columns=columns)

#     # Convert numeric columns to proper datatypes
#     numeric_columns = columns[2:]  # All columns except id and diagnosis
#     for col in numeric_columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     # Save to CSV
#     df.to_csv(output_path, index=False)

#     # Print basic statistics
#     print(f"\nDataset statistics for {prompt_name} (Experiment {experiment_num}):")
#     print(f"Total records: {len(df)}")

#     # Print diagnosis distribution
#     print("\nDiagnosis distribution:")
#     print(df["diagnosis"].value_counts(normalize=True))

#     # Print summary of numeric columns
#     print("\nNumeric columns summary:")
#     print(df.describe())


# def get_system_info():
#     """Get current system resource usage - with better error handling for MIG"""
#     system_info = {}

#     try:
#         cpu_percent = psutil.cpu_percent(interval=1)
#         memory = psutil.virtual_memory()

#         system_info.update({
#             "cpu_percent": cpu_percent,
#             "memory_total_gb": memory.total / (1024**3),
#             "memory_used_gb": memory.used / (1024**3),
#             "memory_percent": memory.percent,
#         })
#     except Exception as e:
#         print(f"Warning: Error getting system info: {e}")
#         system_info["system_error"] = str(e)

#     # Add GPU information if available - with careful error handling
#     if torch.cuda.is_available():
#         try:
#             gpus = GPUtil.getGPUs()
#             if gpus:
#                 gpu = gpus[0]
#                 system_info.update({
#                     "gpu_name": gpu.name,
#                     "gpu_memory_total_mb": gpu.memoryTotal,
#                     "gpu_memory_used_mb": gpu.memoryUsed,
#                     "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
#                     "gpu_temperature": gpu.temperature,
#                 })
#             else:
#                 system_info["gpu_info"] = "No GPUs found by GPUtil"

#             # Add torch.cuda info as fallback
#             system_info.update({
#                 "cuda_device_count": torch.cuda.device_count(),
#                 "cuda_current_device": torch.cuda.current_device(),
#                 "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
#             })
#         except Exception as e:
#             print(f"Warning: Error getting GPU info: {e}")
#             system_info["gpu_error"] = str(e)

#     return system_info


# def log_experiment_metrics(
#     experiment_num,
#     prompt_name,
#     start_time,
#     end_time,
#     system_info,
#     num_records,
#     output_dir,
# ):
#     """Log experiment metrics to JSON"""
#     metrics = {
#         "experiment_number": experiment_num,
#         "prompt_name": prompt_name,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "duration_seconds": (end_time - start_time),
#         "records_generated": num_records,
#         "records_per_second": num_records / max(1, (end_time - start_time)),
#         "system_info": system_info,
#     }

#     # Create metrics directory if it doesn't exist
#     metrics_dir = os.path.join(output_dir, "metrics")
#     os.makedirs(metrics_dir, exist_ok=True)

#     # Save metrics to JSON file
#     metrics_file = os.path.join(
#         metrics_dir, f"metrics_exp{experiment_num}_{prompt_name}.json"
#     )
#     with open(metrics_file, "w") as f:
#         json.dump(metrics, f, indent=4)

#     return metrics


# def main():
#     print(f"Starting GPT-2 breast cancer data generation on device {DEVICE}")
#     print(f"Device is {'CUDA' if DEVICE >= 0 else 'CPU'}")

#     try:
#         # Initialize generator with careful error handling
#         print(f"Initializing generator with {MODEL_NAME} on device {DEVICE}")
#         try:
#             # Use PyTorch explicitly (not TensorFlow) to avoid Keras compatibility issues
#             if DEVICE >= 0:
#                 device_str = f"cuda:{DEVICE}"
#                 print(f"Device set to use {device_str}")
#                 generator = pipeline(
#                     "text-generation",
#                     model=MODEL_NAME,
#                     device=DEVICE,
#                     framework="pt",  # Explicitly use PyTorch
#                     torch_dtype=torch.float16,  # Use half precision for efficiency
#                 )
#             else:
#                 generator = pipeline(
#                     "text-generation",
#                     model=MODEL_NAME,
#                     device=DEVICE,
#                     framework="pt",  # Explicitly use PyTorch
#                 )
#             print("Successfully initialized generator")
#         except Exception as e:
#             print(f"Error initializing with half precision: {e}")
#             print("Falling back to CPU or full precision")
#             generator = pipeline(
#                 "text-generation",
#                 model=MODEL_NAME,
#                 device=-1,  # Use CPU if GPU fails
#                 framework="pt",  # Explicitly use PyTorch
#             )

#         # Store all experiment metrics
#         all_metrics = []

#         # Run multiple experiments with different seeds
#         base_seed = int(time.time())

#         for experiment_num in range(1, NUM_EXPERIMENTS + 1):
#             experiment_seed = base_seed + experiment_num
#             print(
#                 f"\n=== Starting Experiment {experiment_num} (Seed: {experiment_seed}) ==="
#             )

#             for prompt_name, prompt_text in PROMPTS.items():
#                 print(
#                     f"\n=== Generating for: {prompt_name} (Experiment {experiment_num}) ==="
#                 )

#                 # Record start time and initial system info
#                 start_time = time.time()
#                 initial_system_info = get_system_info()

#                 # Generate dataset using the efficient batch method
#                 records = generate_dataset_efficiently(generator, prompt_text, seed=experiment_seed)

#                 # Record end time and final system info
#                 end_time = time.time()
#                 final_system_info = get_system_info()

#                 # Save dataset
#                 save_and_validate_dataset(records, prompt_name, experiment_num)

#                 # Log metrics
#                 metrics = log_experiment_metrics(
#                     experiment_num,
#                     prompt_name,
#                     start_time,
#                     end_time,
#                     {"initial": initial_system_info, "final": final_system_info},
#                     len(records),
#                     OUTPUT_DIR,
#                 )

#                 all_metrics.append(metrics)

#                 print(
#                     f"Completed generation for {prompt_name} (Experiment {experiment_num})"
#                 )
#                 print(f"Generation time: {metrics['duration_seconds']:.2f} seconds")
#                 print(f"Records per second: {metrics['records_per_second']:.2f}")

#                 # Extra cleanup between prompts
#                 clear_memory()
#                 time.sleep(2)

#             print(f"\n=== Completed Experiment {experiment_num} ===")

#         # Save summary of all experiments
#         summary_file = os.path.join(OUTPUT_DIR, "metrics", "experiments_summary.json")
#         with open(summary_file, "w") as f:
#             json.dump(
#                 {
#                     "total_experiments": NUM_EXPERIMENTS,
#                     "model_name": MODEL_NAME,
#                     "device": DEVICE,
#                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "metrics": all_metrics,
#                 },
#                 f,
#                 indent=4,
#             )

#         print(f"\n=== All experiments completed successfully ===")
#         print(f"Results saved to {OUTPUT_DIR}")

#     except Exception as e:
#         print(f"Critical error in main function: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()

# import gc
# import json
# import os
# import re

# # from prompts import PROMPTS
# import time
# from datetime import datetime

# import GPUtil
# import pandas as pd
# import psutil
# import torch
# from tqdm import tqdm
# from transformers import pipeline, set_seed

# # Configuration
# MODEL_NAME = "gpt2"
# OUTPUT_DIR = "gpt2-100-prompt-in-5experiments"
# BATCH_SIZE = 10  # Reduced batch size for better stability
# NUM_BATCHES = 10  # Increased number of batches to maintain total count
# DEVICE = 0 if torch.cuda.is_available() else -1
# MAX_NEW_TOKENS = 200  # Specific tokens to generate
# NUM_EXPERIMENTS = 5  # Number of experiments to run

# # Create output directory
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Define prompts dictionary
# # Define prompts dictionary
# PROMPTS = {
#     # Step 1: Basic Prompt with Examples
#     "PROMPT_1": """
#     I need you to generate synthetic breast-cancer data that closely resembles real-world data. The dataset should contain 100 samples with the following columns:

#     - id: Unique identifier for each record
#     - diagnosis: Classification with 'M' (37.3% of cases), 'B' (62.7% of cases)
#     - radius_mean: Mean radius value
#     - texture_mean: Mean texture value
#     - perimeter_mean: Mean perimeter value
#     - area_mean: Mean area value
#     - smoothness_mean: Mean smoothness value
#     - compactness_mean: Mean compactness value
#     - concavity_mean: Mean concavity value
#     - concave points_mean: Mean concave points value
#     - symmetry_mean: Mean symmetry value
#     - fractal_dimension_mean: Mean fractal dimension value
#     - radius_se: Numerical measurement of radius se
#     - texture_se: Numerical measurement of texture se
#     - perimeter_se: Numerical measurement of perimeter se
#     - area_se: Numerical measurement of area se
#     - smoothness_se: Numerical measurement of smoothness se
#     - compactness_se: Numerical measurement of compactness se
#     - concavity_se: Numerical measurement of concavity se
#     - concave points_se: Numerical measurement of concave points se
#     - symmetry_se: Numerical measurement of symmetry se
#     - fractal_dimension_se: Numerical measurement of fractal dimension se
#     - radius_worst: Numerical measurement of radius worst
#     - texture_worst: Numerical measurement of texture worst
#     - perimeter_worst: Numerical measurement of perimeter worst
#     - area_worst: Numerical measurement of area worst
#     - smoothness_worst: Numerical measurement of smoothness worst
#     - compactness_worst: Numerical measurement of compactness worst
#     - concavity_worst: Numerical measurement of concavity worst
#     - concave points_worst: Numerical measurement of concave points worst
#     - symmetry_worst: Numerical measurement of symmetry worst
#     - fractal_dimension_worst: Numerical measurement of fractal dimension worst
#     - Unnamed: 32: Numerical measurement of unnamed: 32

#     Here are 3 example records from a real dataset to guide your generation:

#     Example 1:
#     9110732, M, 17.75, 28.03, 117.3, 981.6, 0.09997, 0.1314, 0.1698, 0.08293, 0.1713, 0.05916, 0.3897, 1.077, 2.873, 43.95, 0.004714, 0.02015, 0.03697, 0.0111, 0.01237, 0.002556, 21.53, 38.54, 145.4, 1437, 0.1401, 0.3762, 0.6399, 0.197, 0.2972, 0.09075, 0

#     Example 2:
#     8911670, M, 18.81, 19.98, 120.9, 1102, 0.08923, 0.05884, 0.0802, 0.05843, 0.155, 0.04996, 0.3283, 0.828, 2.363, 36.74, 0.007571, 0.01114, 0.02623, 0.01463, 0.0193, 0.001676, 19.96, 24.3, 129, 1236, 0.1243, 0.116, 0.221, 0.1294, 0.2567, 0.05737, 0

#     Example 3:
#     904689, B, 12.96, 18.29, 84.18, 525.2, 0.07351, 0.07899, 0.04057, 0.01883, 0.1874, 0.05899, 0.2357, 1.299, 2.397, 20.21, 0.003629, 0.03713, 0.03452, 0.01065, 0.02632, 0.003705, 14.13, 24.61, 96.31, 621.9, 0.09329, 0.2318, 0.1604, 0.06608, 0.3207, 0.07247, 0

#     Please generate 100 records in a CSV format that follows these patterns and maintains realistic relationships between the features. The data should be plausible and preserve the correlations between features that would be found in real breast-cancer data.
#     """,
#     # Step 2: Prompt with Definitions
#     "PROMPT_2": """
#     I need you to generate synthetic breast-cancer data. Please create 100 samples that realistically represent the patterns and relationships in this type of data.

#     Each record should include:

#     - id: Unique identifier for each record
#     - diagnosis: Classification with 'M' (37.3% of cases), 'B' (62.7% of cases)
#     - radius_mean: Mean radius value (range: ~6.981-28.11)
#     - texture_mean: Mean texture value (range: ~9.71-39.28)
#     - perimeter_mean: Mean perimeter value (range: ~43.79-188.5)
#     - area_mean: Mean area value (range: ~143.5-2501.0)
#     - smoothness_mean: Mean smoothness value (range: ~0.0526-0.1634)
#     - compactness_mean: Mean compactness value (range: ~0.0194-0.3454)
#     - concavity_mean: Mean concavity value (range: ~0.0-0.4268)
#     - concave points_mean: Mean concave points value (range: ~0.0-0.2012)
#     - symmetry_mean: Mean symmetry value (range: ~0.106-0.304)
#     - fractal_dimension_mean: Mean fractal dimension value (range: ~0.05-0.0974)
#     - radius_se: Numerical measurement of radius se (range: ~0.1115-2.873)
#     - texture_se: Numerical measurement of texture se (range: ~0.3602-4.885)
#     - perimeter_se: Numerical measurement of perimeter se (range: ~0.757-21.98)
#     - area_se: Numerical measurement of area se (range: ~6.802-542.2)
#     - smoothness_se: Numerical measurement of smoothness se (range: ~0.0017-0.0311)
#     - compactness_se: Numerical measurement of compactness se (range: ~0.0023-0.1354)
#     - concavity_se: Numerical measurement of concavity se (range: ~0.0-0.396)
#     - concave points_se: Numerical measurement of concave points se (range: ~0.0-0.0528)
#     - symmetry_se: Numerical measurement of symmetry se (range: ~0.0079-0.079)
#     - fractal_dimension_se: Numerical measurement of fractal dimension se (range: ~0.0009-0.0298)
#     - radius_worst: Numerical measurement of radius worst (range: ~7.93-36.04)
#     - texture_worst: Numerical measurement of texture worst (range: ~12.02-49.54)
#     - perimeter_worst: Numerical measurement of perimeter worst (range: ~50.41-251.2)
#     - area_worst: Numerical measurement of area worst (range: ~185.2-4254.0)
#     - smoothness_worst: Numerical measurement of smoothness worst (range: ~0.0712-0.2226)
#     - compactness_worst: Numerical measurement of compactness worst (range: ~0.0273-1.058)
#     - concavity_worst: Numerical measurement of concavity worst (range: ~0.0-1.252)
#     - concave points_worst: Numerical measurement of concave points worst (range: ~0.0-0.291)
#     - symmetry_worst: Numerical measurement of symmetry worst (range: ~0.1565-0.6638)
#     - fractal_dimension_worst: Numerical measurement of fractal dimension worst (range: ~0.055-0.2075)
#     - Unnamed: 32: Numerical measurement of unnamed: 32 (range: ~0.0-0.0)

#     The data should maintain realistic correlations: M typically has higher values for radius_mean, higher values for texture_mean and higher values for perimeter_mean compared to B. There should be natural variance in the data while maintaining these relationships.

#     Here are 3 example records from a real dataset:

#     Example 1 (M):
#     85638502, M, 13.17, 21.81, 85.42, 531.5, 0.09714, 0.1047, 0.08259, 0.05252, 0.1746, 0.06177, 0.1938, 0.6123, 1.334, 14.49, 0.00335, 0.01384, 0.01452, 0.006853, 0.01113, 0.00172, 16.23, 29.89, 105.5, 740.7, 0.1503, 0.3904, 0.3728, 0.1607, 0.3693, 0.09618, 0

#     Example 2 (M):
#     842517, M, 20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 24.99, 23.41, 158.8, 1956, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902, 0

#     Example 3 (B):
#     91544002, B, 11.06, 17.12, 71.25, 366.5, 0.1194, 0.1071, 0.04063, 0.04268, 0.1954, 0.07976, 0.1779, 1.03, 1.318, 12.3, 0.01262, 0.02348, 0.018, 0.01285, 0.0222, 0.008313, 11.69, 20.74, 76.08, 411.1, 0.1662, 0.2031, 0.1256, 0.09514, 0.278, 0.1168, 0

#     Please provide 100 synthetic records in CSV format, with values that are plausible and maintain the natural relationships between features.
#     """,
#     # Step 3: Prompt with Metadata
#     "PROMPT_3": """
#     I need you to generate synthetic breast-cancer data based on real statistical properties. Please generate 100 records that accurately represent the data while maintaining the statistical properties and correlations found in real data.

#     Each record should include:

#     - id: Unique identifier for each record
#     - diagnosis: Classification with 'M' (37.3% of cases), 'B' (62.7% of cases)

#     Breast-cancer measurements with their definitions and statistical properties:

#     | Feature | Definition | Overall Mean | B Mean | M Mean | Min | Max | Std Dev |
#     |---------|------------|--------------|-------------|-------------|-----|-----|---------|
#     | radius_mean | Mean radius value | 14.1273 | 12.1465 | 17.4628 | 6.981 | 28.11 | 3.524 |
#     | texture_mean | Mean texture value | 19.2896 | 17.9148 | 21.6049 | 9.71 | 39.28 | 4.301 |
#     | perimeter_mean | Mean perimeter value | 91.969 | 78.0754 | 115.3654 | 43.79 | 188.5 | 24.299 |
#     | area_mean | Mean area value | 654.8891 | 462.7902 | 978.3764 | 143.5 | 2501.0 | 351.9141 |
#     | smoothness_mean | Mean smoothness value | 0.0964 | 0.0925 | 0.1029 | 0.0526 | 0.1634 | 0.0141 |
#     | compactness_mean | Mean compactness value | 0.1043 | 0.0801 | 0.1452 | 0.0194 | 0.3454 | 0.0528 |
#     | concavity_mean | Mean concavity value | 0.0888 | 0.0461 | 0.1608 | 0.0 | 0.4268 | 0.0797 |
#     | concave points_mean | Mean concave points value | 0.0489 | 0.0257 | 0.088 | 0.0 | 0.2012 | 0.0388 |
#     | symmetry_mean | Mean symmetry value | 0.1812 | 0.1742 | 0.1929 | 0.106 | 0.304 | 0.0274 |
#     | fractal_dimension_mean | Mean fractal dimension value | 0.0628 | 0.0629 | 0.0627 | 0.05 | 0.0974 | 0.0071 |
#     | radius_se | Numerical measurement of radius se | 0.4052 | 0.2841 | 0.6091 | 0.1115 | 2.873 | 0.2773 |
#     | texture_se | Numerical measurement of texture se | 1.2169 | 1.2204 | 1.2109 | 0.3602 | 4.885 | 0.5516 |
#     | perimeter_se | Numerical measurement of perimeter se | 2.8661 | 2.0003 | 4.3239 | 0.757 | 21.98 | 2.0219 |
#     | area_se | Numerical measurement of area se | 40.3371 | 21.1351 | 72.6724 | 6.802 | 542.2 | 45.491 |
#     | smoothness_se | Numerical measurement of smoothness se | 0.007 | 0.0072 | 0.0068 | 0.0017 | 0.0311 | 0.003 |
#     | compactness_se | Numerical measurement of compactness se | 0.0255 | 0.0214 | 0.0323 | 0.0023 | 0.1354 | 0.0179 |
#     | concavity_se | Numerical measurement of concavity se | 0.0319 | 0.026 | 0.0418 | 0.0 | 0.396 | 0.0302 |
#     | concave points_se | Numerical measurement of concave points se | 0.0118 | 0.0099 | 0.0151 | 0.0 | 0.0528 | 0.0062 |
#     | symmetry_se | Numerical measurement of symmetry se | 0.0205 | 0.0206 | 0.0205 | 0.0079 | 0.079 | 0.0083 |
#     | fractal_dimension_se | Numerical measurement of fractal dimension se | 0.0038 | 0.0036 | 0.0041 | 0.0009 | 0.0298 | 0.0026 |
#     | radius_worst | Numerical measurement of radius worst | 16.2692 | 13.3798 | 21.1348 | 7.93 | 36.04 | 4.8332 |
#     | texture_worst | Numerical measurement of texture worst | 25.6772 | 23.5151 | 29.3182 | 12.02 | 49.54 | 6.1463 |
#     | perimeter_worst | Numerical measurement of perimeter worst | 107.2612 | 87.0059 | 141.3703 | 50.41 | 251.2 | 33.6025 |
#     | area_worst | Numerical measurement of area worst | 880.5831 | 558.8994 | 1422.2863 | 185.2 | 4254.0 | 569.357 |
#     | smoothness_worst | Numerical measurement of smoothness worst | 0.1324 | 0.125 | 0.1448 | 0.0712 | 0.2226 | 0.0228 |
#     | compactness_worst | Numerical measurement of compactness worst | 0.2543 | 0.1827 | 0.3748 | 0.0273 | 1.058 | 0.1573 |
#     | concavity_worst | Numerical measurement of concavity worst | 0.2722 | 0.1662 | 0.4506 | 0.0 | 1.252 | 0.2086 |
#     | concave points_worst | Numerical measurement of concave points worst | 0.1146 | 0.0744 | 0.1822 | 0.0 | 0.291 | 0.0657 |
#     | symmetry_worst | Numerical measurement of symmetry worst | 0.2901 | 0.2702 | 0.3235 | 0.1565 | 0.6638 | 0.0619 |
#     | fractal_dimension_worst | Numerical measurement of fractal dimension worst | 0.0839 | 0.0794 | 0.0915 | 0.055 | 0.2075 | 0.0181 |
#     | Unnamed: 32 | Numerical measurement of unnamed: 32 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

#     Key correlations to maintain:
#     - Strong positive correlation among radius_mean, perimeter_worst, area_mean, radius_worst, perimeter_mean
#     - Moderate positive correlation among concavity_se, concave points_worst, radius_worst, perimeter_mean, concave points_se
#     - diagnosis class differences: B and M samples show significant differences in concavity_mean, area_se, concave points_mean, concavity_worst, area_worst

#     Here are 3 example records from the real dataset:

#     Example 1 (M):
#     885429, M, 19.73, 19.82, 130.7, 1206, 0.1062, 0.1849, 0.2417, 0.0974, 0.1733, 0.06697, 0.7661, 0.78, 4.115, 92.81, 0.008482, 0.05057, 0.068, 0.01971, 0.01467, 0.007259, 25.28, 25.59, 159.8, 1933, 0.171, 0.5955, 0.8489, 0.2507, 0.2749, 0.1297, 0

#     Example 2 (M):
#     9012000, M, 22.01, 21.9, 147.2, 1482, 0.1063, 0.1954, 0.2448, 0.1501, 0.1824, 0.0614, 1.008, 0.6999, 7.561, 130.2, 0.003978, 0.02821, 0.03576, 0.01471, 0.01518, 0.003796, 27.66, 25.8, 195, 2227, 0.1294, 0.3885, 0.4756, 0.2432, 0.2741, 0.08574, 0

#     Example 3 (B):
#     904969, B, 12.34, 14.95, 78.29, 469.1, 0.08682, 0.04571, 0.02109, 0.02054, 0.1571, 0.05708, 0.3833, 0.9078, 2.602, 30.15, 0.007702, 0.008491, 0.01307, 0.0103, 0.0297, 0.001432, 13.18, 16.85, 84.11, 533.1, 0.1048, 0.06744, 0.04921, 0.04793, 0.2298, 0.05974, 0

#     Please provide 100 synthetic records in CSV format, with values that are plausible and maintain both the statistical properties and natural relationships between features. Ensure the data could be useful for machine learning algorithms to differentiate between different diagnosis values.
#     """,
#     # Step 4: Prompt with Rules
#     "PROMPT_4": """
#     I need you to generate synthetic breast-cancer data based exclusively on statistical properties without using any real examples. Please create 100 synthetic records that represent breast-cancer measurements while preserving the statistical distributions and correlations in the real-world data.

#     Each record should include:

#     - id: Unique identifier for each record
#     - diagnosis: Classification with 'M' (37.3% of cases), 'B' (62.7% of cases)

#     The data should contain breast-cancer measurements with their statistical properties:

#     | Feature | Definition | Overall Mean | B Mean | M Mean | Min | Max | Std Dev |
#     |---------|------------|--------------|-------------|-------------|-----|-----|---------|
#     | radius_mean | Mean radius value | 14.1273 | 12.1465 | 17.4628 | 6.981 | 28.11 | 3.524 |
#     | texture_mean | Mean texture value | 19.2896 | 17.9148 | 21.6049 | 9.71 | 39.28 | 4.301 |
#     | perimeter_mean | Mean perimeter value | 91.969 | 78.0754 | 115.3654 | 43.79 | 188.5 | 24.299 |
#     | area_mean | Mean area value | 654.8891 | 462.7902 | 978.3764 | 143.5 | 2501.0 | 351.9141 |
#     | smoothness_mean | Mean smoothness value | 0.0964 | 0.0925 | 0.1029 | 0.0526 | 0.1634 | 0.0141 |
#     | compactness_mean | Mean compactness value | 0.1043 | 0.0801 | 0.1452 | 0.0194 | 0.3454 | 0.0528 |
#     | concavity_mean | Mean concavity value | 0.0888 | 0.0461 | 0.1608 | 0.0 | 0.4268 | 0.0797 |
#     | concave points_mean | Mean concave points value | 0.0489 | 0.0257 | 0.088 | 0.0 | 0.2012 | 0.0388 |
#     | symmetry_mean | Mean symmetry value | 0.1812 | 0.1742 | 0.1929 | 0.106 | 0.304 | 0.0274 |
#     | fractal_dimension_mean | Mean fractal dimension value | 0.0628 | 0.0629 | 0.0627 | 0.05 | 0.0974 | 0.0071 |
#     | radius_se | Numerical measurement of radius se | 0.4052 | 0.2841 | 0.6091 | 0.1115 | 2.873 | 0.2773 |
#     | texture_se | Numerical measurement of texture se | 1.2169 | 1.2204 | 1.2109 | 0.3602 | 4.885 | 0.5516 |
#     | perimeter_se | Numerical measurement of perimeter se | 2.8661 | 2.0003 | 4.3239 | 0.757 | 21.98 | 2.0219 |
#     | area_se | Numerical measurement of area se | 40.3371 | 21.1351 | 72.6724 | 6.802 | 542.2 | 45.491 |
#     | smoothness_se | Numerical measurement of smoothness se | 0.007 | 0.0072 | 0.0068 | 0.0017 | 0.0311 | 0.003 |
#     | compactness_se | Numerical measurement of compactness se | 0.0255 | 0.0214 | 0.0323 | 0.0023 | 0.1354 | 0.0179 |
#     | concavity_se | Numerical measurement of concavity se | 0.0319 | 0.026 | 0.0418 | 0.0 | 0.396 | 0.0302 |
#     | concave points_se | Numerical measurement of concave points se | 0.0118 | 0.0099 | 0.0151 | 0.0 | 0.0528 | 0.0062 |
#     | symmetry_se | Numerical measurement of symmetry se | 0.0205 | 0.0206 | 0.0205 | 0.0079 | 0.079 | 0.0083 |
#     | fractal_dimension_se | Numerical measurement of fractal dimension se | 0.0038 | 0.0036 | 0.0041 | 0.0009 | 0.0298 | 0.0026 |
#     | radius_worst | Numerical measurement of radius worst | 16.2692 | 13.3798 | 21.1348 | 7.93 | 36.04 | 4.8332 |
#     | texture_worst | Numerical measurement of texture worst | 25.6772 | 23.5151 | 29.3182 | 12.02 | 49.54 | 6.1463 |
#     | perimeter_worst | Numerical measurement of perimeter worst | 107.2612 | 87.0059 | 141.3703 | 50.41 | 251.2 | 33.6025 |
#     | area_worst | Numerical measurement of area worst | 880.5831 | 558.8994 | 1422.2863 | 185.2 | 4254.0 | 569.357 |
#     | smoothness_worst | Numerical measurement of smoothness worst | 0.1324 | 0.125 | 0.1448 | 0.0712 | 0.2226 | 0.0228 |
#     | compactness_worst | Numerical measurement of compactness worst | 0.2543 | 0.1827 | 0.3748 | 0.0273 | 1.058 | 0.1573 |
#     | concavity_worst | Numerical measurement of concavity worst | 0.2722 | 0.1662 | 0.4506 | 0.0 | 1.252 | 0.2086 |
#     | concave points_worst | Numerical measurement of concave points worst | 0.1146 | 0.0744 | 0.1822 | 0.0 | 0.291 | 0.0657 |
#     | symmetry_worst | Numerical measurement of symmetry worst | 0.2901 | 0.2702 | 0.3235 | 0.1565 | 0.6638 | 0.0619 |
#     | fractal_dimension_worst | Numerical measurement of fractal dimension worst | 0.0839 | 0.0794 | 0.0915 | 0.055 | 0.2075 | 0.0181 |
#     | Unnamed: 32 | Numerical measurement of unnamed: 32 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

#     Please provide 100 synthetic records in CSV format that satisfy these statistical properties and mathematical relationships. The synthetic data should be suitable for analysis or training machine learning models while preserving privacy by not containing any actual data points.
#     """,
# }


# def validate_record(record):
#     """Validate if a generated record matches expected format and constraints for breast cancer dataset"""
#     try:
#         parts = record.strip().split(",")
#         # Breast cancer dataset has 32 columns (id, diagnosis, 30 features)
#         if len(parts) != 32:
#             return False

#         # Extract key fields
#         id_val, diagnosis = parts[0], parts[1]

#         # Basic format checks
#         if not id_val.strip().isdigit():  # ID should be numeric
#             return False

#         if diagnosis not in ["M", "B"]:  # Diagnosis should be M or B
#             return False

#         # Convert all feature values to float for validation
#         try:
#             values = [float(p) for p in parts[2:]]
#         except ValueError:
#             return False

#         # Check ranges for key features
#         # Radius mean (index 2)
#         if not (6.5 <= values[0] <= 29):
#             return False

#         # Area mean (index 5-2=3)
#         if not (140 <= values[3] <= 2600):
#             return False

#         # Check relationships between related features
#         radius_mean, perimeter_mean, area_mean = values[0], values[2], values[3]

#         # Check if perimeter and area are reasonably proportional to radius
#         # Perimeter should be roughly 2πr
#         if not (5.5 * radius_mean <= perimeter_mean <= 7 * radius_mean):
#             return False

#         # Area should be roughly πr²
#         if not (2.5 * radius_mean**2 <= area_mean <= 4 * radius_mean**2):
#             return False

#         # Check "worst" values are >= corresponding "mean" values
#         if values[20] < values[0]:  # radius_worst < radius_mean
#             return False

#         if values[23] < values[3]:  # area_worst < area_mean
#             return False

#         # Malignant samples typically have higher values for certain features
#         if diagnosis == "M" and values[6] < 0.04:  # concavity_mean too low for M
#             return False

#         # Check Unnamed: 32 is always 0
#         if values[29] != 0:
#             return False

#         return True
#     except Exception as e:
#         # print(f"Validation error: {e} for record: {record}")
#         return False


# def extract_records(text):
#     """Extract valid records from generated text"""
#     # Split into lines and find lines that might be records
#     lines = text.split("\n")
#     potential_records = [line.strip() for line in lines if "," in line]

#     # Validate each potential record
#     valid_records = [record for record in potential_records if validate_record(record)]
#     return valid_records


# def generate_dataset(generator, prompt, num_records=100, seed=None):
#     """Generate dataset with improved handling and experiment-specific seed"""
#     if seed is not None:
#         set_seed(seed)

#     all_records = []
#     pbar = tqdm(total=num_records, desc="Generating records")
#     max_attempts = 3

#     def clear_memory():
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()
#         gc.collect()

#     while len(all_records) < num_records:
#         for attempt in range(max_attempts):
#             try:
#                 clear_memory()  # Clear memory before generation

#                 # Generate text with smaller chunks for PROMPT_3
#                 if "Statistical properties:" in prompt:  # PROMPT_3 detection
#                     max_tokens = MAX_NEW_TOKENS * 2  # Double tokens for PROMPT_3
#                 else:
#                     max_tokens = MAX_NEW_TOKENS

#                 output = generator(
#                     prompt,
#                     max_new_tokens=max_tokens,
#                     num_return_sequences=1,
#                     temperature=0.7,
#                     do_sample=True,
#                     pad_token_id=50256,
#                 )

#                 # Extract and validate records
#                 new_records = extract_records(output[0]["generated_text"])

#                 # Add new valid records
#                 for record in new_records:
#                     if len(all_records) < num_records:
#                         all_records.append(record)
#                         pbar.update(1)
#                     else:
#                         break

#                 break  # Success, break attempt loop

#             except RuntimeError as e:
#                 if "out of memory" in str(e) and attempt < max_attempts - 1:
#                     print(f"OOM error, attempt {attempt + 1}/{max_attempts}")
#                     clear_memory()
#                     continue
#                 else:
#                     print(f"Error in generation: {str(e)}")
#                     break

#     pbar.close()
#     return all_records


# def save_and_validate_dataset(records, prompt_name, experiment_num):
#     """Save dataset and perform basic validation with experiment number"""
#     # Create experiment-specific directory
#     experiment_dir = os.path.join(OUTPUT_DIR, f"experiment_{experiment_num}")
#     os.makedirs(experiment_dir, exist_ok=True)

#     output_path = os.path.join(experiment_dir, f"{prompt_name}_data.csv")

#     # Create DataFrame
#     columns = [
#         "gender",
#         "age",
#         "hypertension",
#         "heart_disease",
#         "smoking_history",
#         "bmi",
#         "HbA1c_level",
#         "blood_glucose_level",
#         "diabetes",
#     ]

#     df = pd.DataFrame([r.split(",") for r in records], columns=columns)

#     # Convert datatypes
#     df["age"] = pd.to_numeric(df["age"], errors="coerce")
#     df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
#     df["HbA1c_level"] = pd.to_numeric(df["HbA1c_level"], errors="coerce")
#     df["blood_glucose_level"] = pd.to_numeric(
#         df["blood_glucose_level"], errors="coerce"
#     )
#     df["hypertension"] = pd.to_numeric(df["hypertension"], errors="coerce")
#     df["heart_disease"] = pd.to_numeric(df["heart_disease"], errors="coerce")
#     df["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce")

#     # Save to CSV
#     df.to_csv(output_path, index=False)

#     # Print basic statistics
#     print(f"\nDataset statistics for {prompt_name} (Experiment {experiment_num}):")
#     print(f"Total records: {len(df)}")
#     print("\nNumerical columns summary:")
#     print(df.describe())
#     print("\nCategorical columns value counts:")
#     for col in [
#         "gender",
#         "smoking_history",
#         "hypertension",
#         "heart_disease",
#         "diabetes",
#     ]:
#         print(f"\n{col}:")
#         print(df[col].value_counts(normalize=True))


# def get_system_info():
#     """Get current system resource usage"""
#     cpu_percent = psutil.cpu_percent(interval=1)
#     memory = psutil.virtual_memory()

#     system_info = {
#         "cpu_percent": cpu_percent,
#         "memory_total_gb": memory.total / (1024**3),
#         "memory_used_gb": memory.used / (1024**3),
#         "memory_percent": memory.percent,
#     }

#     # Add GPU information if available
#     if torch.cuda.is_available():
#         gpu = GPUtil.getGPUs()[0]
#         system_info.update(
#             {
#                 "gpu_name": gpu.name,
#                 "gpu_memory_total_mb": gpu.memoryTotal,
#                 "gpu_memory_used_mb": gpu.memoryUsed,
#                 "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
#                 "gpu_temperature": gpu.temperature,
#             }
#         )

#     return system_info


# def log_experiment_metrics(
#     experiment_num,
#     prompt_name,
#     start_time,
#     end_time,
#     system_info,
#     num_records,
#     output_dir,
# ):
#     """Log experiment metrics to JSON"""
#     metrics = {
#         "experiment_number": experiment_num,
#         "prompt_name": prompt_name,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "duration_seconds": (end_time - start_time),
#         "records_generated": num_records,
#         "records_per_second": num_records / (end_time - start_time),
#         "system_info": system_info,
#     }

#     # Create metrics directory if it doesn't exist
#     metrics_dir = os.path.join(output_dir, "metrics")
#     os.makedirs(metrics_dir, exist_ok=True)

#     # Save metrics to JSON file
#     metrics_file = os.path.join(
#         metrics_dir, f"metrics_exp{experiment_num}_{prompt_name}.json"
#     )
#     with open(metrics_file, "w") as f:
#         json.dump(metrics, f, indent=4)

#     return metrics


# def main():
#     # Initialize generator
#     print(f"Initializing generator with {MODEL_NAME} on device {DEVICE}")
#     generator = pipeline(
#         "text-generation",
#         model=MODEL_NAME,
#         device=DEVICE,
#         torch_dtype=torch.float16 if DEVICE >= 0 else torch.float32,
#     )

#     # Store all experiment metrics
#     all_metrics = []

#     # Run multiple experiments with different seeds
#     base_seed = int(time.time())

#     for experiment_num in range(1, NUM_EXPERIMENTS + 1):
#         experiment_seed = base_seed + experiment_num
#         print(
#             f"\n=== Starting Experiment {experiment_num} (Seed: {experiment_seed}) ==="
#         )

#         for prompt_name, prompt_text in PROMPTS.items():
#             print(
#                 f"\n=== Generating for: {prompt_name} (Experiment {experiment_num}) ==="
#             )

#             # Record start time and initial system info
#             start_time = time.time()
#             initial_system_info = get_system_info()

#             # Generate dataset
#             records = generate_dataset(generator, prompt_text, seed=experiment_seed)

#             # Record end time and final system info
#             end_time = time.time()
#             final_system_info = get_system_info()

#             # Save dataset
#             save_and_validate_dataset(records, prompt_name, experiment_num)

#             # Log metrics
#             metrics = log_experiment_metrics(
#                 experiment_num,
#                 prompt_name,
#                 start_time,
#                 end_time,
#                 {"initial": initial_system_info, "final": final_system_info},
#                 len(records),
#                 OUTPUT_DIR,
#             )

#             all_metrics.append(metrics)

#             print(
#                 f"Completed generation for {prompt_name} (Experiment {experiment_num})"
#             )
#             print(f"Generation time: {metrics['duration_seconds']:.2f} seconds")
#             print(f"Records per second: {metrics['records_per_second']:.2f}")

#         print(f"\n=== Completed Experiment {experiment_num} ===")

#     # Save summary of all experiments
#     summary_file = os.path.join(OUTPUT_DIR, "metrics", "experiments_summary.json")
#     with open(summary_file, "w") as f:
#         json.dump(
#             {
#                 "total_experiments": NUM_EXPERIMENTS,
#                 "model_name": MODEL_NAME,
#                 "device": DEVICE,
#                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "metrics": all_metrics,
#             },
#             f,
#             indent=4,
#         )


# if __name__ == "__main__":
#     main()
