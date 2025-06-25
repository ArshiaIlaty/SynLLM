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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

# Set CUDA environment variables for MIG
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust based on your MIG configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Change gpt2 this to the model you want to use
# Options: "gpt2", "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "google/gemma-2b", "microsoft/phi-2"
MODEL_FAMILY = (
    MODEL_NAME.split("/")[-1].split("-")[0].lower()
)  # Extract model family name
OUTPUT_DIR = f"{MODEL_FAMILY}-stroke-data-100-5experiments"
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


def adjust_memory_settings(model_name):
    """Adjust memory settings based on model type"""
    model_name_lower = model_name.lower()

    # Default memory fraction for GPT-2
    memory_fraction = 0.6

    # Larger models need more memory
    if any(m in model_name_lower for m in ["llama", "llama2", "llama-3"]):
        memory_fraction = 0.8  # LLaMA models are larger
    elif any(m in model_name_lower for m in ["mistral", "mixtral"]):
        memory_fraction = 0.85  # Mixtral models can be very large
    elif any(m in model_name_lower for m in ["gemma"]):
        memory_fraction = 0.75
    elif any(m in model_name_lower for m in ["phi"]):
        memory_fraction = 0.7

    if torch.cuda.is_available():
        print(f"Setting memory fraction to {memory_fraction} for {model_name}")
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(memory_fraction)


# Safely initialize device
try:
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        # Start with lower memory footprint
        torch.cuda.empty_cache()
        # Set memory based on model
        # Memory fraction will be set later based on model type
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
        Generate synthetic stroke data in CSV format:
        id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke

        Examples:
        36338,Female,39,1,0,Yes,Private,Rural,58.09,39.2,smokes,1
        39186,Female,57,0,1,Yes,Private,Urban,216.58,31,Unknown,1
        71444,Female,53,0,0,Yes,Private,Rural,97.89,38.7,formerly smoked,0

        Generate more records following these rules:
        - id: Unique identifier between 10000 and 99999
        - gender: Male or Female
        - age: between 0.08 and 82
        - hypertension: 0 or 1
        - heart_disease: 0 or 1
        - ever_married: Yes or No
        - work_type: Private, Self-employed, Govt_job, children, Never_worked
        - Residence_type: Urban or Rural
        - avg_glucose_level: between 55 and 272
        - bmi: between 10 and 98
        - smoking_status: formerly smoked, never smoked, smokes, Unknown
        - stroke: 0 or 1 (with 1 being about 5% of cases)
        """,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.7,
        "use_cpu": False,
    },
    # Step 2: Prompt with Definitions
    "PROMPT_2": {
        "text": """
        Generate synthetic stroke data with these definitions:
        id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke

        Definitions:
        - id: Unique identifier for each record (10000-99999)
        - gender: Patient's biological sex (Male/Female)
        - age: Age in years (0.08-82)
        - hypertension: High blood pressure diagnosis (0=No, 1=Yes)
        - heart_disease: Heart disease diagnosis (0=No, 1=Yes)
        - ever_married: Marriage status (Yes/No)
        - work_type: Type of employment (Private/Self-employed/Govt_job/children/Never_worked)
        - Residence_type: Type of residence (Urban/Rural)
        - avg_glucose_level: Average glucose level (55-272)
        - bmi: Body Mass Index (10-98)
        - smoking_status: Smoking history (formerly smoked/never smoked/smokes/Unknown)
        - stroke: Stroke diagnosis (0=No, 1=Yes) - only about 5% should be Yes

        Examples:
        26727,Female,79,0,0,No,Private,Rural,88.92,22.9,never smoked,1
        11933,Female,79,0,0,Yes,Private,Rural,169.67,28.1,Unknown,1
        31697,Female,34,0,0,Yes,Private,Urban,76.42,27.6,smokes,0
        """,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.7,
        "use_cpu": False,
    },
    # Step 3: Prompt with Metadata
    "PROMPT_3": {
        "text": """
        Generate synthetic stroke data with these statistics:
        id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke

        Statistical properties:
        | Feature | Overall Mean | 0 Mean | 1 Mean | Min | Max | Std Dev |
        |---------|--------------|-------------|-------------|-----|-----|---------|
        | age | 43.2266 | 41.9715 | 67.7282 | 0.08 | 82.0 | 22.6126 |
        | hypertension | 0.0975 | 0.0889 | 0.2651 | 0 | 1 | 0.2966 |
        | heart_disease | 0.054 | 0.0471 | 0.1888 | 0 | 1 | 0.2261 |
        | avg_glucose_level | 106.1477 | 104.7955 | 132.5447 | 55.12 | 271.74 | 45.2836 |
        | bmi | 28.862 | 28.7991 | 30.0904 | 10.3 | 97.6 | 7.6996 |

        Key relationships:
        - Stroke (1) cases have higher avg age, hypertension rate, heart disease rate
        - Only 4.9% of records should have stroke=1

        Examples:
        47306,Male,58,0,0,No,Private,Rural,92.62,32,Unknown,1
        8580,Female,77,0,0,Yes,Self-employed,Rural,90,32,never smoked,1
        48017,Male,55,0,0,Yes,Private,Urban,62.56,28.6,never smoked,0
        """,
        "max_tokens": MAX_NEW_TOKENS // 2,  # Half tokens for PROMPT_3
        "temperature": 0.6,  # Lower temperature for more focused generation
        "use_cpu": True,  # Force CPU for PROMPT_3 which was successful with diabetes
    },
    # Step 4: Prompt with Rules - USING SPECIAL SETTINGS TO FIX ISSUES
    "PROMPT_4": {
        "text": """
        Generate synthetic stroke data following these rules:
        id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke

        Rules:
        1. Higher age (60+) increases stroke risk
        2. Hypertension (1) increases stroke risk
        3. Heart disease (1) increases stroke risk
        4. Higher glucose levels (>120) increase stroke risk
        5. BMI outside normal range increases risk
        6. Only 4.9% of records should have stroke=1
        7. Must follow these ranges:
           - id: Unique identifier between 10000 and 99999
           - gender: Male or Female only
           - age: 0.08-82 only
           - hypertension: 0 or 1 only
           - heart_disease: 0 or 1 only
           - ever_married: Yes or No only
           - work_type: Private, Self-employed, Govt_job, children, or Never_worked only
           - Residence_type: Urban or Rural only
           - avg_glucose_level: 55-272 only
           - bmi: 10-98 only
           - smoking_status: formerly smoked, never smoked, smokes, or Unknown only
           - stroke: 0 or 1 only (with 1 being 4.9% of cases)
        """,
        "max_tokens": MAX_NEW_TOKENS // 3,  # Even fewer tokens for PROMPT_4
        "temperature": 0.5,  # Lower temperature for more focused generation
        "use_cpu": True,  # Force CPU for PROMPT_4
    },
}


def initialize_model(model_name, device):
    """Initialize model based on its type with appropriate settings"""
    print(f"Initializing generator with {model_name} on device {device}")

    # Define model families (add more as needed)
    llama_models = ["llama", "llama2", "llama-2", "llama-3"]
    mistral_models = ["mistral", "mistralai", "mixtral"]
    gemma_models = ["gemma", "google/gemma"]
    phi_models = ["phi", "microsoft/phi"]

    model_config = {}

    # Default to half precision on GPU
    if device >= 0:
        model_config["torch_dtype"] = torch.float16

    # Add model-specific configurations
    model_name_lower = model_name.lower()

    is_llama = any(m in model_name_lower for m in llama_models)
    is_mistral = any(m in model_name_lower for m in mistral_models)
    is_gemma = any(m in model_name_lower for m in gemma_models)
    is_phi = any(m in model_name_lower for m in phi_models)

    # Set maximum token length based on model type
    if is_llama or is_mistral:
        model_config["max_length"] = 2048
    elif is_gemma:
        model_config["max_length"] = 2048
    elif is_phi:
        model_config["max_length"] = 1024
    else:  # Default for GPT-2 and others
        model_config["max_length"] = 1024

    # For large models, consider quantization
    if is_llama or is_mistral:
        # Uncomment the next line to enable 8-bit quantization for large models
        # model_config["load_in_8bit"] = True
        pass

    # Try to initialize with optimized settings
    try:
        # Try to initialize the model
        generator = pipeline(
            "text-generation", model=model_name, device=device, **model_config
        )
        print("Successfully initialized generator")
        return generator
    except Exception as e:
        print(f"Error initializing with default config: {e}")
        print("Trying with minimal configuration...")
        try:
            # Fall back to minimal configuration
            generator = pipeline(
                "text-generation",
                model=model_name,
                device=-1
                if device >= 0 and "out of memory" in str(e).lower()
                else device,  # Use CPU if GPU OOM
            )
            return generator
        except Exception as e:
            print(f"Error initializing with minimal config: {e}")
            raise


def get_model_specific_params(model_name, base_params):
    """Get model-specific generation parameters"""
    model_name_lower = model_name.lower()
    params = base_params.copy()

    # LLaMA family specific params
    if any(m in model_name_lower for m in ["llama", "llama2", "llama-2", "llama-3"]):
        # LLaMA often works better with these settings
        params["top_p"] = 0.9
        params["repetition_penalty"] = 1.1
        if "pad_token_id" in params:
            del params["pad_token_id"]  # LLaMA handles padding differently

    # Mistral family specific params
    elif any(m in model_name_lower for m in ["mistral", "mistralai", "mixtral"]):
        params["top_p"] = 0.92
        params["repetition_penalty"] = 1.05
        if "pad_token_id" in params:
            del params["pad_token_id"]

    # Gemma specific params
    elif any(m in model_name_lower for m in ["gemma", "google/gemma"]):
        params["top_p"] = 0.9
        params["repetition_penalty"] = 1.05
        if "pad_token_id" in params:
            del params["pad_token_id"]

    # Phi specific params
    elif any(m in model_name_lower for m in ["phi", "microsoft/phi"]):
        params["top_p"] = 0.9
        if "pad_token_id" in params:
            del params["pad_token_id"]

    # For all newer models, use new parameters if available
    if "gpt2" not in model_name_lower:
        # More modern models might prefer these over older parameters
        if "do_sample" not in params:
            params["do_sample"] = True

    return params


def validate_record(record):
    """Validate if a generated record matches expected format and constraints for stroke data"""
    try:
        parts = record.strip().split(",")
        if len(parts) != 12:  # Updated to 12 fields for stroke data
            return False

        # Validate each field
        (
            id_val,
            gender,
            age,
            hyp,
            heart,
            married,
            work_type,
            residence,
            glucose,
            bmi,
            smoking,
            stroke,
        ) = parts

        # Basic format checks
        try:
            id_int = int(id_val)
            if not (10000 <= id_int <= 99999):
                return False
        except ValueError:
            return False

        if gender not in ["Male", "Female"]:
            return False

        try:
            age_float = float(age)
            if not (0.08 <= age_float <= 82.0):
                return False
        except ValueError:
            return False

        if int(hyp) not in [0, 1]:
            return False

        if int(heart) not in [0, 1]:
            return False

        if married not in ["Yes", "No"]:
            return False

        if work_type not in [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked",
        ]:
            return False

        if residence not in ["Urban", "Rural"]:
            return False

        try:
            glucose_float = float(glucose)
            if not (55.0 <= glucose_float <= 272.0):
                return False
        except ValueError:
            return False

        try:
            bmi_float = float(bmi)
            if not (10.0 <= bmi_float <= 98.0):
                return False
        except ValueError:
            return False

        if smoking not in ["formerly smoked", "never smoked", "smokes", "Unknown"]:
            return False

        if int(stroke) not in [0, 1]:
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
            id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke

            For example, a record might look like:
            ID,Gender,Age,Hypertension,HeartDisease,EverMarried,WorkType,ResidenceType,AvgGlucoseLevel,BMI,SmokingStatus,Stroke
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
            id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke

            Each record should be on a new line and follow all the rules above.
            Remember that only about 5% of records should have stroke=1.
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
            id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke

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

        # Get model-specific parameters
        generation_params = {
            "max_new_tokens": approach["max_tokens"],
            "do_sample": True,
            "temperature": approach["temperature"],
            "num_return_sequences": 1,
        }
        # Add pad_token_id only for GPT-2
        if "gpt2" in MODEL_NAME.lower():
            generation_params["pad_token_id"] = 50256

        # Get model-specific parameters
        generation_params = get_model_specific_params(MODEL_NAME, generation_params)

        # Try multiple attempts with this approach
        for attempt in range(approach["attempts"]):
            if len(all_records) >= num_records:
                break

            try:
                # Clear memory
                clear_memory()

                # Generate text
                output = cpu_generator(modified_prompt, **generation_params)

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

        # Define fields and constraints for stroke data
        fields = [
            {"name": "id", "min": 10000, "max": 99999},
            {"name": "gender", "values": ["Male", "Female"]},
            {"name": "age", "min": 0.08, "max": 82.0},
            {"name": "hypertension", "values": [0, 1]},
            {"name": "heart_disease", "values": [0, 1]},
            {"name": "ever_married", "values": ["Yes", "No"]},
            {
                "name": "work_type",
                "values": [
                    "Private",
                    "Self-employed",
                    "Govt_job",
                    "children",
                    "Never_worked",
                ],
            },
            {"name": "Residence_type", "values": ["Urban", "Rural"]},
            {"name": "avg_glucose_level", "min": 55.0, "max": 272.0},
            {"name": "bmi", "min": 10.0, "max": 98.0},
            {
                "name": "smoking_status",
                "values": ["formerly smoked", "never smoked", "smokes", "Unknown"],
            },
            {
                "name": "stroke",
                "values": [0, 1],
                "distribution": [0.951, 0.049],
            },  # 4.9% should be 1
        ]

        # Generate records needed to complete the set
        needed_records = num_records - len(all_records)
        for record_num in range(needed_records):
            record_values = []

            # Decide stroke status first to ensure proper distribution
            stroke_value = (
                1 if record_num < int(needed_records * 0.049) else 0
            )  # Ensure ~4.9% are stroke=1

            # Generate each field
            for field in fields:
                try:
                    if field["name"] == "stroke":
                        # We've already decided stroke value
                        record_values.append(stroke_value)
                        continue

                    field_prompt = f"""
                    Based on the rules for stroke data:
                    1. Higher age (60+) increases stroke risk
                    2. Hypertension (1) increases stroke risk
                    3. Heart disease (1) increases stroke risk
                    4. Higher glucose levels (>120) increase stroke risk
                    5. BMI outside normal range increases risk

                    Generate a valid value for the {field['name']} field. The stroke value will be {stroke_value}.
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

                    # Set up generation parameters
                    field_params = {
                        "max_new_tokens": 20,  # Short generation for single value
                        "do_sample": True,
                        "temperature": 0.5,
                    }

                    # Add pad_token_id only for GPT-2
                    if "gpt2" in MODEL_NAME.lower():
                        field_params["pad_token_id"] = 50256

                    # Get model-specific parameters
                    field_params = get_model_specific_params(MODEL_NAME, field_params)

                    # Generate value
                    field_output = cpu_generator(field_prompt, **field_params)

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
                                "id",
                                "hypertension",
                                "heart_disease",
                                "stroke",
                            ]:
                                value = int(value)
                            elif field["name"] in ["age", "avg_glucose_level", "bmi"]:
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

                # Prepare generation parameters
                generation_params = {
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "temperature": temperature,
                    "num_return_sequences": batch_size,
                }

                # Add pad_token_id only for GPT-2
                if "gpt2" in MODEL_NAME.lower():
                    generation_params["pad_token_id"] = 50256

                # Get model-specific parameters
                generation_params = get_model_specific_params(
                    MODEL_NAME, generation_params
                )

                # Generate text with appropriate parameters
                try:
                    output = current_generator(prompt_text, **generation_params)
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
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    ]

    df = pd.DataFrame([r.split(",") for r in records], columns=columns)

    # Convert datatypes
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["hypertension"] = pd.to_numeric(df["hypertension"], errors="coerce")
    df["heart_disease"] = pd.to_numeric(df["heart_disease"], errors="coerce")
    df["avg_glucose_level"] = pd.to_numeric(df["avg_glucose_level"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")

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
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
        "stroke",
    ]:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True))

    # Calculate and print stroke distribution
    stroke_dist = df["stroke"].value_counts(normalize=True)
    print("\nStroke distribution:")
    print(f"0 (No Stroke): {stroke_dist.get(0, 0)*100:.2f}%")
    print(f"1 (Stroke): {stroke_dist.get(1, 0)*100:.2f}%")

    # Check if target ratio is close to expected (4.9%)
    stroke_pct = stroke_dist.get(1, 0) * 100
    if abs(stroke_pct - 4.9) > 2.0:  # Allow 2% deviation
        print(
            f"WARNING: Stroke percentage ({stroke_pct:.2f}%) deviates significantly from expected 4.9%"
        )


def main():
    # Set memory settings based on model
    adjust_memory_settings(MODEL_NAME)

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

    print(f"Starting {MODEL_NAME} stroke data generation on device {DEVICE}")
    print(f"Device is {'CUDA' if DEVICE >= 0 else 'CPU'}")
    print(f"Results will be saved to: {OUTPUT_DIR}")
    print(f"Logs will be saved to: {LOG_DIR}")

    try:
        # Initialize generator with model-specific configurations
        print(f"Initializing generator with {MODEL_NAME} on device {DEVICE}")
        generator = initialize_model(MODEL_NAME, DEVICE)
        print("Successfully initialized model!")

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
        # Authentication for Hugging Face (uncomment if needed)
        # from huggingface_hub import login
        # login("your_hf_token_here")  # Replace with your token if using gated models

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

        # Print model information
        print(f"\n=== Model Information ===")
        print(f"Model: {MODEL_NAME}")
        print(f"Model family: {MODEL_FAMILY}")
        print(f"Output directory: {OUTPUT_DIR}")

        # Initialize a basic progress file
        progress_file = os.path.join(OUTPUT_DIR, "progress.log")
        with open(progress_file, "w") as f:
            f.write(
                f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Using model: {MODEL_NAME}\n")

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
