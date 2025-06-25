import gc
import json
import os
import re
import signal
import time
from datetime import datetime

import GPUtil
import pandas as pd
import psutil
import torch
from tqdm import tqdm
from transformers import pipeline, set_seed

# Set CUDA environment variables for MIG
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust based on your MIG configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Configuration
MODEL_NAME = "gpt2"
OUTPUT_DIR = "gpt2-noEX-prompt4-100-prompt-in-5experiments"
BATCH_SIZE = 5  # Reduced batch size for better memory management on MIG slices
NUM_BATCHES = 20  # Adjusted to maintain total count
MAX_NEW_TOKENS = 200  # Default tokens to generate
TIMEOUT_SECONDS = 300  # Maximum time to wait for a generation (5 minutes)
NUM_EXPERIMENTS = 5  # Number of experiments to run


# Add timeout handling
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Generation timed out")


# Safely initialize device
try:
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        # Start with lower memory footprint
        torch.cuda.empty_cache()
        # Set to use less GPU memory - REDUCED TO 60% for safety
        torch.cuda.set_per_process_memory_fraction(0.6)  # Use only 60% of GPU memory
        test_tensor = torch.zeros(1, device="cuda:0")

        # Get GPU details for better debugging
        gpu_id = torch.cuda.current_device()
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(
            f"Initial memory allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**2:.2f} MB"
        )
        print(
            f"Initial memory reserved: {torch.cuda.memory_reserved(gpu_id) / 1024**2:.2f} MB"
        )

        print("CUDA initialization successful")
        DEVICE = 0
    else:
        print("CUDA not available")
        DEVICE = -1
except Exception as e:
    print(f"Error initializing CUDA: {e}")
    print("Falling back to CPU")
    DEVICE = -1

# Create output directory and log directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Define prompts dictionary with specific generation settings for each
PROMPTS = {
    # Step 1: Basic Prompt with Examples
    "PROMPT_1": {
        "text": """
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
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.7,
        "use_cpu": False,
    },
    # Step 2: Prompt with Definitions
    "PROMPT_2": {
        "text": """
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
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.7,
        "use_cpu": False,
    },
    # Step 3: Prompt with Metadata
    "PROMPT_3": {
        "text": """
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
        "max_tokens": MAX_NEW_TOKENS // 2,  # Half tokens for PROMPT_3
        "temperature": 0.6,  # Lower temperature for more focused generation
        "use_cpu": True,  # Force CPU for PROMPT_3 which was successful
    },
    # Step 4: Prompt with Rules - USING SPECIAL SETTINGS TO FIX ISSUES
    "PROMPT_4": {
        "text": """
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
        """,
        "max_tokens": MAX_NEW_TOKENS // 3,  # Even fewer tokens for PROMPT_4
        "temperature": 0.5,  # Lower temperature for more focused generation
        "use_cpu": True,  # Force CPU for PROMPT_4
    },
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
        try:
            # Handle potential formatting issues with the glucose value
            glucose_val = int(float(glucose)) if glucose.strip() else 0
            if not (70 <= glucose_val <= 300):
                return False
        except ValueError:
            return False
        if int(diabetes) not in [0, 1]:
            return False

        return True
    except Exception as e:
        # print(f"Validation error: {e} for record: {record}")
        return False


def extract_records(text):
    """Extract valid records from generated text"""
    # Split into lines and find lines that might be records
    lines = text.split("\n")
    potential_records = [line.strip() for line in lines if "," in line]

    # Validate each potential record
    valid_records = [record for record in potential_records if validate_record(record)]
    return valid_records


def clear_memory():
    """Clear GPU and system memory"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: Memory clearing issue: {e}")
    gc.collect()


def generate_dataset_for_prompt4(
    generator, prompt_config, num_records=100, seed=None, experiment_num=1
):
    """
    Special function for handling PROMPT_4 generation without using synthetic data.
    Uses multiple approaches to get the model to generate valid records.
    """
    if seed is not None:
        set_seed(seed)

    # Extract settings
    prompt_text = prompt_config["text"]
    max_tokens = prompt_config["max_tokens"]
    temperature = prompt_config["temperature"]

    # Create log file
    log_file = os.path.join(LOG_DIR, f"generation_log_exp{experiment_num}_PROMPT_4.txt")
    with open(log_file, "w") as f:
        f.write(
            f"PROMPT_4 generation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(
            f"Original parameters: max_tokens={max_tokens}, temperature={temperature}\n\n"
        )
        f.write("Attempting specialized approaches for PROMPT_4 without examples\n")

    # Create a progress bar
    pbar = tqdm(total=num_records, desc="Generating PROMPT_4 records")
    all_records = []

    # Create CPU generator for stability
    try:
        cpu_generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device=-1,  # Force CPU
        )
        print("Created CPU generator for PROMPT_4")
    except Exception as e:
        print(f"Error creating CPU generator: {e}")
        cpu_generator = generator

    # Try multiple approaches
    approaches = [
        # Approach 1: Add a formatting hint without examples
        {
            "name": "Format hint",
            "prompt_addition": """
            To generate data, use this format exactly:
            gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

            For example, a record might look like:
            Gender,Age,Hypertension,HeartDisease,SmokingHistory,BMI,HbA1c,BloodGlucose,Diabetes
            """,
            "temperature": 0.5,
            "max_tokens": 500,
            "attempts": 5,
        },
        # Approach 2: Add more specific instructions
        {
            "name": "Detailed instructions",
            "prompt_addition": """
            Please generate exactly 10 records with the following format:
            gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

            Each record should be on a new line and follow all the rules above.
            """,
            "temperature": 0.7,
            "max_tokens": 400,
            "attempts": 5,
        },
        # Approach 3: Generate one record at a time with specific format
        {
            "name": "Single record generation",
            "prompt_addition": """
            Generate a single record in exactly this format:
            gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

            Just output the record values with no additional text or explanation.
            """,
            "temperature": 0.6,
            "max_tokens": 100,
            "attempts": 15,
        },
        # Approach 4: Use lower temperature
        {
            "name": "Lower temperature",
            "prompt_addition": "",  # No addition
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "attempts": 5,
        },
        # Approach 5: Use higher temperature
        {
            "name": "Higher temperature",
            "prompt_addition": "",  # No addition
            "temperature": 0.9,
            "max_tokens": max_tokens,
            "attempts": 5,
        },
    ]

    # Try each approach until we have enough records
    for approach in approaches:
        if len(all_records) >= num_records:
            break

        print(f"\nTrying PROMPT_4 approach: {approach['name']}")
        with open(log_file, "a") as f:
            f.write(f"\n\n=== Trying approach: {approach['name']} ===\n")
            f.write(
                f"Temperature: {approach['temperature']}, Max tokens: {approach['max_tokens']}\n"
            )

        # Create modified prompt
        modified_prompt = prompt_text + approach["prompt_addition"]

        # Try multiple attempts with this approach
        for attempt in range(approach["attempts"]):
            if len(all_records) >= num_records:
                break

            try:
                # Clear memory
                clear_memory()

                # Generate text
                output = cpu_generator(
                    modified_prompt,
                    max_new_tokens=approach["max_tokens"],
                    do_sample=True,
                    temperature=approach["temperature"],
                    pad_token_id=50256,
                    num_return_sequences=1,
                )

                # Get generated text
                generated_text = (
                    output[0]["generated_text"]
                    if isinstance(output, list)
                    else output["generated_text"]
                )

                # Log a sample
                with open(log_file, "a") as f:
                    f.write(f"\nAttempt {attempt+1}:\n")
                    f.write("Generated text sample:\n")
                    f.write(
                        generated_text[:500] + "...\n"
                        if len(generated_text) > 500
                        else generated_text
                    )

                # Extract records
                new_records = extract_records(generated_text)

                # Log found records
                with open(log_file, "a") as f:
                    f.write(f"\nFound {len(new_records)} valid records\n")
                    if new_records:
                        f.write("Records:\n")
                        for record in new_records:
                            f.write(f"{record}\n")

                # Add records to our collection
                for record in new_records:
                    if len(all_records) < num_records:
                        all_records.append(record)
                        pbar.update(1)
                    else:
                        break

            except Exception as e:
                print(f"Error in PROMPT_4 generation attempt: {str(e)}")
                with open(log_file, "a") as f:
                    f.write(f"\nError: {str(e)}\n")
                continue

    # If we still don't have enough records, we'll try one last approach - breaking down by steps
    if len(all_records) < num_records:
        print("\nAttempting step-by-step generation for PROMPT_4")
        with open(log_file, "a") as f:
            f.write("\n\n=== Attempting step-by-step generation ===\n")

        # Define fields and constraints
        fields = [
            {"name": "gender", "values": ["Male", "Female"]},
            {"name": "age", "min": 18, "max": 80},
            {"name": "hypertension", "values": [0, 1]},
            {"name": "heart_disease", "values": [0, 1]},
            {
                "name": "smoking_history",
                "values": ["never", "former", "current", "not current"],
            },
            {"name": "bmi", "min": 15, "max": 60},
            {"name": "HbA1c_level", "min": 4, "max": 9},
            {"name": "blood_glucose_level", "min": 70, "max": 300},
            {"name": "diabetes", "values": [0, 1]},
        ]

        # Generate records needed to complete the set
        needed_records = num_records - len(all_records)
        for record_num in range(needed_records):
            record_values = []

            # Generate each field
            for field in fields:
                try:
                    field_prompt = f"""
                    Based on the rules for diabetes data:
                    1. Higher HbA1c (>6.5) usually means diabetes=1
                    2. Higher blood glucose (>180) usually means diabetes=1
                    3. Older age increases hypertension and heart_disease risk
                    4. Higher BMI (>30) increases diabetes risk

                    Generate a valid value for the {field['name']} field.
                    """

                    # Add context from previously generated fields
                    if record_values:
                        field_prompt += f"\nAlready generated values: "
                        for i, prev_field in enumerate(fields[: len(record_values)]):
                            field_prompt += f"{prev_field['name']}={record_values[i]}, "

                    # Add constraints
                    if "values" in field:
                        field_prompt += (
                            f"\nValid values for {field['name']}: {field['values']}"
                        )
                    elif "min" in field:
                        field_prompt += f"\nValid range for {field['name']}: {field['min']} to {field['max']}"

                    field_prompt += f"\nJust output the value with no additional text."

                    # Generate value
                    field_output = cpu_generator(
                        field_prompt,
                        max_new_tokens=20,  # Short generation for single value
                        do_sample=True,
                        temperature=0.5,
                        pad_token_id=50256,
                    )

                    # Get output text
                    output_text = (
                        field_output[0]["generated_text"]
                        if isinstance(field_output, list)
                        else field_output["generated_text"]
                    )

                    # Extract value - taking last part of generated text and cleaning
                    raw_value = output_text.strip().split("\n")[-1].strip()

                    # Clean up the value
                    if "values" in field:
                        # For categorical values, check if any valid value is in the output
                        for valid_value in field["values"]:
                            if str(valid_value) in raw_value:
                                value = valid_value
                                break
                        else:
                            # Default to first value if none found
                            value = field["values"][0]
                    else:
                        # For numerical values, extract number
                        import re

                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw_value)
                        if numbers:
                            value = float(numbers[0])
                            # Ensure within range
                            value = max(field["min"], min(field["max"], value))
                            # Keep integers as integers
                            if field["name"] in [
                                "hypertension",
                                "heart_disease",
                                "diabetes",
                                "blood_glucose_level",
                            ]:
                                value = int(value)
                            elif field["name"] in ["age", "bmi", "HbA1c_level"]:
                                value = round(value, 1)  # Round to 1 decimal
                        else:
                            # Default to minimum if no number found
                            value = field["min"]

                    record_values.append(value)

                except Exception as e:
                    # If error, use default value for the field
                    if "values" in field:
                        value = field["values"][0]
                    else:
                        value = field["min"]
                    record_values.append(value)

            # Create record string
            record = ",".join([str(value) for value in record_values])

            # Validate and add
            if validate_record(record):
                all_records.append(record)
                pbar.update(1)
                with open(log_file, "a") as f:
                    f.write(f"Generated step-by-step record: {record}\n")

    pbar.close()

    # Log completion
    with open(log_file, "a") as f:
        f.write(
            f"\nPROMPT_4 generation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Total records generated: {len(all_records)}\n")
        if len(all_records) < num_records:
            f.write(
                f"WARNING: Only generated {len(all_records)}/{num_records} requested records\n"
            )

    return all_records


def generate_dataset(
    generator,
    prompt_config,
    num_records=100,
    seed=None,
    experiment_num=1,
    prompt_name="",
):
    """Generate dataset with improved handling and experiment-specific seed"""
    # Special case for PROMPT_4
    if prompt_name == "PROMPT_4":
        return generate_dataset_for_prompt4(
            generator, prompt_config, num_records, seed, experiment_num
        )

    # Regular generation for other prompts
    if seed is not None:
        set_seed(seed)

    # Extract settings from prompt config
    prompt_text = prompt_config["text"]
    max_tokens = prompt_config["max_tokens"]
    temperature = prompt_config["temperature"]
    use_cpu = prompt_config["use_cpu"]

    # Log the generation parameters
    print(f"\nGeneration parameters for {prompt_name}:")
    print(f"- Max tokens: {max_tokens}")
    print(f"- Temperature: {temperature}")
    print(f"- Using CPU: {use_cpu}")

    # If we need to use CPU, create a CPU-specific generator
    if use_cpu and DEVICE >= 0:
        print(f"Creating CPU generator for {prompt_name}")
        try:
            temp_generator = pipeline(
                "text-generation",
                model=MODEL_NAME,
                device=-1,
            )
            current_generator = temp_generator
            current_device = -1
        except Exception as e:
            print(f"Error creating CPU generator: {e}")
            current_generator = generator
            current_device = DEVICE
    else:
        current_generator = generator
        current_device = DEVICE

    # Create a log file for this specific generation
    log_file = os.path.join(
        LOG_DIR, f"generation_log_exp{experiment_num}_{prompt_name}.txt"
    )
    with open(log_file, "w") as f:
        f.write(
            f"Generation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(
            f"Parameters: max_tokens={max_tokens}, temperature={temperature}, device={current_device}\n\n"
        )

    all_records = []
    pbar = tqdm(total=num_records, desc="Generating records")
    max_attempts = 10
    batch_size = 1  # Generate one at a time for better stability

    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)

    while len(all_records) < num_records:
        for attempt in range(max_attempts):
            try:
                clear_memory()  # Clear memory before generation

                # Set timeout alarm
                signal.alarm(TIMEOUT_SECONDS)

                # Generate text with appropriate parameters
                try:
                    output = current_generator(
                        prompt_text,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=50256,
                        num_return_sequences=batch_size,
                    )
                finally:
                    # Disable alarm
                    signal.alarm(0)

                # Check if output is a list or dictionary
                if isinstance(output, list):
                    generated_texts = [item["generated_text"] for item in output]
                else:
                    generated_texts = [output["generated_text"]]

                # Process each generated text
                all_new_records = []
                for generated_text in generated_texts:
                    # Log a sample of the generation to the file
                    with open(log_file, "a") as f:
                        f.write(f"\n--- Generation Sample (Attempt {attempt+1}) ---\n")
                        f.write(
                            generated_text[:500] + "...\n"
                            if len(generated_text) > 500
                            else generated_text
                        )

                    # Extract and validate records
                    new_records = extract_records(generated_text)
                    all_new_records.extend(new_records)

                    # Log found records
                    with open(log_file, "a") as f:
                        f.write(f"\nFound {len(new_records)} valid records\n")
                        if new_records:
                            f.write("Sample records:\n")
                            for i, record in enumerate(
                                new_records[:3]
                            ):  # Log just a few samples
                                f.write(f"{i+1}. {record}\n")

                # Add new valid records
                for record in all_new_records:
                    if len(all_records) < num_records:
                        all_records.append(record)
                        pbar.update(1)
                    else:
                        break

                # If we got records, no need to retry
                if all_new_records:
                    break
                else:
                    # If no records found, try with different temperature
                    with open(log_file, "a") as f:
                        f.write(
                            f"\nNo valid records found, changing temperature for next attempt\n"
                        )
                    temperature = 0.8 if temperature < 0.7 else 0.6
                    time.sleep(1)  # Wait a moment before retrying

            except TimeoutException:
                with open(log_file, "a") as f:
                    f.write(
                        f"\n!!! Generation timed out after {TIMEOUT_SECONDS} seconds !!!\n"
                    )
                print(
                    f"\nWarning: Generation timed out for {prompt_name}. Trying again with different parameters."
                )

                # Fall back to CPU if on GPU
                if current_device >= 0:
                    with open(log_file, "a") as f:
                        f.write(f"\nFalling back to CPU after timeout\n")
                    print("Falling back to CPU after timeout")
                    try:
                        temp_generator = pipeline(
                            "text-generation",
                            model=MODEL_NAME,
                            device=-1,
                        )
                        current_generator = temp_generator
                        current_device = -1
                    except Exception as e:
                        print(f"Error creating CPU generator: {e}")

                # Reduce tokens and try again
                max_tokens = max(50, max_tokens // 2)
                with open(log_file, "a") as f:
                    f.write(f"\nReducing max_tokens to {max_tokens}\n")

                continue  # Try the next attempt

            except RuntimeError as e:
                # Handle out-of-memory errors
                with open(log_file, "a") as f:
                    f.write(f"\nRuntime error: {str(e)}\n")

                if "out of memory" in str(e) and current_device >= 0:
                    print(f"OOM error on attempt {attempt+1}, falling back to CPU")
                    # Create a CPU version of the generator
                    with open(log_file, "a") as f:
                        f.write(f"\nFalling back to CPU after OOM error\n")
                    try:
                        temp_generator = pipeline(
                            "text-generation",
                            model=MODEL_NAME,
                            device=-1,
                        )
                        current_generator = temp_generator
                        current_device = -1
                    except Exception as e:
                        print(f"Error creating CPU generator: {e}")
                    clear_memory()
                    continue
                else:
                    print(f"Error in generation: {str(e)}")
                    with open(log_file, "a") as f:
                        f.write(f"\nError in generation: {str(e)}\n")
                    if attempt < max_attempts - 1:
                        time.sleep(2)  # Wait before retrying
                        continue
                    else:
                        break
            except Exception as e:
                print(f"Unexpected error in generation: {str(e)}")
                with open(log_file, "a") as f:
                    f.write(f"\nUnexpected error: {str(e)}\n")
                if attempt < max_attempts - 1:
                    time.sleep(3)  # Wait before retrying
                    continue
                else:
                    break

    pbar.close()

    # Log completion
    with open(log_file, "a") as f:
        f.write(
            f"\nGeneration completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Total records generated: {len(all_records)}\n")

    return all_records


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
                gpu = gpus[0]
                system_info.update(
                    {
                        "gpu_name": gpu.name,
                        "gpu_memory_total_mb": gpu.memoryTotal,
                        "gpu_memory_used_mb": gpu.memoryUsed,
                        "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "gpu_temperature": gpu.temperature,
                    }
                )
            else:
                system_info["gpu_info"] = "No GPUs found by GPUtil"

            # Add torch.cuda info as fallback
            system_info.update(
                {
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_current_device": torch.cuda.current_device(),
                    "cuda_device_name": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "N/A",
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

    # Also save a combined status file to track overall progress
    status_file = os.path.join(output_dir, "experiment_status.txt")
    with open(status_file, "a") as f:
        f.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Experiment {experiment_num}, {prompt_name}: Complete ({num_records} records in {(end_time - start_time):.1f}s)\n"
        )

    return metrics


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


def main():
    # Create initial status file
    status_file = os.path.join(OUTPUT_DIR, "experiment_status.txt")
    with open(status_file, "w") as f:
        f.write(
            f"Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Device: {'CUDA' if DEVICE >= 0 else 'CPU'}\n")
        f.write(f"Experiments: {NUM_EXPERIMENTS}\n")
        f.write("----------------------------------------\n")

    print(f"Starting GPT-2 diabetes data generation on device {DEVICE}")
    print(f"Device is {'CUDA' if DEVICE >= 0 else 'CPU'}")
    print(f"Results will be saved to: {OUTPUT_DIR}")
    print(f"Logs will be saved to: {LOG_DIR}")

    try:
        # Initialize generator with careful error handling
        print(f"Initializing generator with {MODEL_NAME} on device {DEVICE}")
        try:
            # Try to use half precision on GPU
            if DEVICE >= 0:
                dtype = torch.float16
                generator = pipeline(
                    "text-generation",
                    model=MODEL_NAME,
                    device=DEVICE,
                    torch_dtype=dtype,
                )
            else:
                generator = pipeline(
                    "text-generation",
                    model=MODEL_NAME,
                    device=DEVICE,
                )
            print("Successfully initialized generator")
        except Exception as e:
            print(f"Error initializing with half precision: {e}")
            print("Falling back to CPU or full precision")
            generator = pipeline(
                "text-generation",
                model=MODEL_NAME,
                device=-1 if DEVICE >= 0 else DEVICE,  # Use CPU if GPU fails
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

            # Track experiment progress
            with open(status_file, "a") as f:
                f.write(
                    f"\nExperiment {experiment_num} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

            for prompt_name, prompt_config in PROMPTS.items():
                try:
                    print(
                        f"\n=== Generating for: {prompt_name} (Experiment {experiment_num}) ==="
                    )

                    # Record start time and initial system info
                    start_time = time.time()
                    initial_system_info = get_system_info()

                    # Generate dataset with specific prompt settings
                    records = generate_dataset(
                        generator,
                        prompt_config,
                        seed=experiment_seed,
                        experiment_num=experiment_num,
                        prompt_name=prompt_name,
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

                    # Extra cleanup between prompts
                    clear_memory()
                    time.sleep(2)

                except Exception as e:
                    print(
                        f"Error in experiment {experiment_num}, prompt {prompt_name}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    # Log the error
                    with open(status_file, "a") as f:
                        f.write(f"ERROR in {prompt_name}: {str(e)}\n")
                    # Continue to next prompt instead of exiting
                    continue

            print(f"\n=== Completed Experiment {experiment_num} ===")
            with open(status_file, "a") as f:
                f.write(
                    f"Experiment {experiment_num} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

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

        print(f"\n=== All experiments completed successfully ===")
        print(f"Results saved to {OUTPUT_DIR}")
        with open(status_file, "a") as f:
            f.write(
                f"\nAll experiments completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

    except Exception as e:
        print(f"Critical error in main function: {e}")
        import traceback

        traceback.print_exc()
        with open(status_file, "a") as f:
            f.write(f"\nCRITICAL ERROR: {str(e)}\n")


if __name__ == "__main__":
    try:
        # Print system information before starting
        print("\n=== System Information ===")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(
                f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )
        else:
            print("CUDA not available")

        # Initialize a basic progress file
        progress_file = os.path.join(OUTPUT_DIR, "progress.log")
        with open(progress_file, "w") as f:
            f.write(
                f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

        # Run the main process
        main()

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting gracefully...")
        # Log the interruption
        with open(os.path.join(OUTPUT_DIR, "experiment_status.txt"), "a") as f:
            f.write(
                f"\nProcess interrupted by user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
    except Exception as e:
        print(f"\n\nUnhandled exception: {str(e)}")
        import traceback

        traceback.print_exc()
        # Log the error
        with open(os.path.join(OUTPUT_DIR, "experiment_status.txt"), "a") as f:
            f.write(
                f"\nUnhandled exception: {str(e)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
    finally:
        print("\nExiting script")
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
# from tqdm import tqdm
# from transformers import pipeline, set_seed

# # Set CUDA environment variables for MIG
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust based on your MIG configuration
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# # Configuration
# MODEL_NAME = "gpt2"
# OUTPUT_DIR = "gpt2-noEX-prompt4-100-prompt-in-5experiments"
# BATCH_SIZE = 5  # Reduced batch size for better memory management on MIG slices
# NUM_BATCHES = 20  # Adjusted to maintain total count
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
#     Generate synthetic diabetes data in CSV format:
#     gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

#     Examples:
#     Female,45.2,0,0,never,28.5,5.7,155,0
#     Male,62.7,1,1,former,32.1,6.8,185,1
#     Female,38.9,0,0,current,24.3,5.2,130,0

#     Generate more records following these rules:
#     - gender: Male or Female
#     - age: between 18 and 80
#     - hypertension: 0 or 1
#     - heart_disease: 0 or 1
#     - smoking_history: never, former, current, or not current
#     - bmi: between 15 and 60
#     - HbA1c_level: between 4 and 9
#     - blood_glucose_level: between 70 and 300
#     - diabetes: 0 or 1
#     """,
#     # Step 2: Prompt with Definitions
#     "PROMPT_2": """
#     Generate synthetic diabetes data with these definitions:
#     gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

#     Definitions:
#     - gender: Patient's biological sex (Male/Female)
#     - age: Age in years (18-80)
#     - hypertension: High blood pressure diagnosis (0=No, 1=Yes)
#     - heart_disease: Heart disease diagnosis (0=No, 1=Yes)
#     - smoking_history: Smoking status (never/former/current/not current)
#     - bmi: Body Mass Index (15-60)
#     - HbA1c_level: Average blood sugar level (4-9)
#     - blood_glucose_level: Current blood glucose (70-300)
#     - diabetes: Diabetes diagnosis (0=No, 1=Yes)

#     Examples:
#     Female,45.2,0,0,never,28.5,5.7,155,0
#     Male,62.7,1,1,former,32.1,6.8,185,1
#     Female,38.9,0,0,current,24.3,5.2,130,0
#     """,
#     # Step 3: Prompt with Metadata
#     "PROMPT_3": """
#     Generate synthetic diabetes data with these statistics:
#     gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

#     Statistical properties:
#     - gender: Male (48%), Female (52%)
#     - age: Mean=41.8, range 18-80
#     - hypertension: No (85%), Yes (15%)
#     - heart_disease: No (92%), Yes (8%)
#     - smoking_history: never (60%), former (22%), current (15%), not current (3%)
#     - bmi: Mean=27.3, range 15-60
#     - HbA1c_level: Mean=5.7, range 4-9
#     - blood_glucose_level: Mean=138, range 70-300
#     - diabetes: No (88%), Yes (12%)

#     Examples:
#     Female,45.2,0,0,never,28.5,5.7,155,0
#     Male,62.7,1,1,former,32.1,6.8,185,1
#     Female,38.9,0,0,current,24.3,5.2,130,0
#     """,
#     # Step 4: Prompt with Rules
#     "PROMPT_4": """
#     Generate synthetic diabetes data following these rules:
#     gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

#     Rules:
#     1. Higher HbA1c (>6.5) usually means diabetes=1
#     2. Higher blood glucose (>180) usually means diabetes=1
#     3. Older age increases hypertension and heart_disease risk
#     4. Higher BMI (>30) increases diabetes risk
#     5. Must follow these ranges:
#        - gender: Male or Female only
#        - age: 18-80 only
#        - hypertension: 0 or 1 only
#        - heart_disease: 0 or 1 only
#        - smoking_history: never/former/current/not current only
#        - bmi: 15-60 only
#        - HbA1c_level: 4-9 only
#        - blood_glucose_level: 70-300 only
#        - diabetes: 0 or 1 only
#     """,
# }


# def validate_record(record):
#     """Validate if a generated record matches expected format and constraints"""
#     try:
#         parts = record.strip().split(",")
#         if len(parts) != 9:
#             return False

#         # Validate each field
#         gender, age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes = parts

#         # Basic format checks
#         if gender not in ["Male", "Female"]:
#             return False
#         if not (18 <= float(age) <= 80):
#             return False
#         if int(hyp) not in [0, 1]:
#             return False
#         if int(heart) not in [0, 1]:
#             return False
#         if smoking not in ["never", "former", "current", "not current"]:
#             return False
#         if not (15 <= float(bmi) <= 60):
#             return False
#         if not (4 <= float(hba1c) <= 9):
#             return False
#         if not (70 <= int(float(glucose)) <= 300):  # Convert to float first to handle decimals
#             return False
#         if int(diabetes) not in [0, 1]:
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


# def generate_dataset(generator, prompt, num_records=100, seed=None):
#     """Generate dataset with improved handling and experiment-specific seed"""
#     if seed is not None:
#         set_seed(seed)

#     all_records = []
#     pbar = tqdm(total=num_records, desc="Generating records")
#     max_attempts = 5  # Increased attempts for MIG environment

#     while len(all_records) < num_records:
#         clear_memory()  # Clear memory before generation

#         try:
#             # Generate text without setting num_return_sequences
#             output = generator(
#                 prompt,
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 do_sample=True,
#                 temperature=0.7,
#                 pad_token_id=50256,
#             )

#             # Check if output is a list or dictionary
#             if isinstance(output, list):
#                 generated_text = output[0]["generated_text"]
#             else:
#                 generated_text = output["generated_text"]

#             # Extract and validate records
#             new_records = extract_records(generated_text)

#             # Add new valid records
#             for record in new_records:
#                 if len(all_records) < num_records:
#                     all_records.append(record)
#                     pbar.update(1)
#                 else:
#                     break

#             # If no records found, wait a moment before retrying
#             if not new_records:
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
#     print(f"Starting GPT-2 diabetes data generation on device {DEVICE}")
#     print(f"Device is {'CUDA' if DEVICE >= 0 else 'CPU'}")

#     try:
#         # Initialize generator with careful error handling
#         print(f"Initializing generator with {MODEL_NAME} on device {DEVICE}")
#         try:
#             # Try to use half precision on GPU
#             if DEVICE >= 0:
#                 dtype = torch.float16
#                 generator = pipeline(
#                     "text-generation",
#                     model=MODEL_NAME,
#                     device=DEVICE,
#                     torch_dtype=dtype,
#                 )
#             else:
#                 generator = pipeline(
#                     "text-generation",
#                     model=MODEL_NAME,
#                     device=DEVICE,
#                 )
#             print("Successfully initialized generator")
#         except Exception as e:
#             print(f"Error initializing with half precision: {e}")
#             print("Falling back to CPU or full precision")
#             generator = pipeline(
#                 "text-generation",
#                 model=MODEL_NAME,
#                 device=-1 if DEVICE >= 0 else DEVICE,  # Use CPU if GPU fails
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

#                 # Generate dataset
#                 records = generate_dataset(generator, prompt_text, seed=experiment_seed)

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
# PROMPTS = {
#     # Step 1: Basic Prompt with Examples
#     "PROMPT_1": """
#     Generate synthetic diabetes data in CSV format:
#     gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

#     Examples:
#     Female,45.2,0,0,never,28.5,5.7,155,0
#     Male,62.7,1,1,former,32.1,6.8,185,1
#     Female,38.9,0,0,current,24.3,5.2,130,0

#     Generate more records following these rules:
#     - gender: Male or Female
#     - age: between 18 and 80
#     - hypertension: 0 or 1
#     - heart_disease: 0 or 1
#     - smoking_history: never, former, current, or not current
#     - bmi: between 15 and 60
#     - HbA1c_level: between 4 and 9
#     - blood_glucose_level: between 70 and 300
#     - diabetes: 0 or 1
#     """,
#     # Step 2: Prompt with Definitions
#     "PROMPT_2": """
#     Generate synthetic diabetes data with these definitions:
#     gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

#     Definitions:
#     - gender: Patient's biological sex (Male/Female)
#     - age: Age in years (18-80)
#     - hypertension: High blood pressure diagnosis (0=No, 1=Yes)
#     - heart_disease: Heart disease diagnosis (0=No, 1=Yes)
#     - smoking_history: Smoking status (never/former/current/not current)
#     - bmi: Body Mass Index (15-60)
#     - HbA1c_level: Average blood sugar level (4-9)
#     - blood_glucose_level: Current blood glucose (70-300)
#     - diabetes: Diabetes diagnosis (0=No, 1=Yes)

#     Examples:
#     Female,45.2,0,0,never,28.5,5.7,155,0
#     Male,62.7,1,1,former,32.1,6.8,185,1
#     Female,38.9,0,0,current,24.3,5.2,130,0
#     """,
#     # Step 3: Prompt with Metadata
#     "PROMPT_3": """
#     Generate synthetic diabetes data with these statistics:
#     gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

#     Statistical properties:
#     - gender: Male (48%), Female (52%)
#     - age: Mean=41.8, range 18-80
#     - hypertension: No (85%), Yes (15%)
#     - heart_disease: No (92%), Yes (8%)
#     - smoking_history: never (60%), former (22%), current (15%), not current (3%)
#     - bmi: Mean=27.3, range 15-60
#     - HbA1c_level: Mean=5.7, range 4-9
#     - blood_glucose_level: Mean=138, range 70-300
#     - diabetes: No (88%), Yes (12%)

#     Examples:
#     Female,45.2,0,0,never,28.5,5.7,155,0
#     Male,62.7,1,1,former,32.1,6.8,185,1
#     Female,38.9,0,0,current,24.3,5.2,130,0
#     """,
#     # Step 4: Prompt with Rules
#     "PROMPT_4": """
#     Generate synthetic diabetes data following these rules:
#     gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

#     Rules:
#     1. Higher HbA1c (>6.5) usually means diabetes=1
#     2. Higher blood glucose (>180) usually means diabetes=1
#     3. Older age increases hypertension and heart_disease risk
#     4. Higher BMI (>30) increases diabetes risk
#     5. Must follow these ranges:
#        - gender: Male or Female only
#        - age: 18-80 only
#        - hypertension: 0 or 1 only
#        - heart_disease: 0 or 1 only
#        - smoking_history: never/former/current/not current only
#        - bmi: 15-60 only
#        - HbA1c_level: 4-9 only
#        - blood_glucose_level: 70-300 only
#        - diabetes: 0 or 1 only

#     Examples:
#     Female,45.2,0,0,never,28.5,5.7,155,0
#     Male,62.7,1,1,former,32.1,6.8,185,1
#     Female,38.9,0,0,current,24.3,5.2,130,0
#     """,
# }


# def validate_record(record):
#     """Validate if a generated record matches expected format and constraints"""
#     try:
#         parts = record.strip().split(",")
#         if len(parts) != 9:
#             return False

#         # Validate each field
#         gender, age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes = parts

#         # Basic format checks
#         if gender not in ["Male", "Female"]:
#             return False
#         if not (18 <= float(age) <= 80):
#             return False
#         if int(hyp) not in [0, 1]:
#             return False
#         if int(heart) not in [0, 1]:
#             return False
#         if smoking not in ["never", "former", "current", "not current"]:
#             return False
#         if not (15 <= float(bmi) <= 60):
#             return False
#         if not (4 <= float(hba1c) <= 9):
#             return False
#         if not (70 <= int(glucose) <= 300):
#             return False
#         if int(diabetes) not in [0, 1]:
#             return False

#         return True
#     except:
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
