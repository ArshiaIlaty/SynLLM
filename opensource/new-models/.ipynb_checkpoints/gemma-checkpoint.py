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
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig

"""
IMPORTANT: This code is optimized for NVIDIA A100 GPUs with MIG configuration.
It uses specific memory management techniques and model loading approaches
designed for enterprise-grade GPUs.
"""

# Add this to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Add explicit device selection for MIG environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Configuration - Using the correct Gemma model ID
MODEL_NAME = "google/gemma-2b"  # Using the 2B parameter model
OUTPUT_DIR = "model_gemma2b-100-prompt-in-5experiments"
MAX_NEW_TOKENS = 500  # Increased to accommodate multiple records
NUM_EXPERIMENTS = 5
BATCH_SIZE = 1  # Using single batch for MIG configuration
NUM_RECORDS = 100

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define base prompts dictionary - these will be transformed into chat format
BASE_PROMPTS = {
    # Step 1: Basic Prompt with Examples
    "PROMPT_1": """
    Generate realistic synthetic patient records for diabetes prediction following this format:
    - gender (String: 'Male' or 'Female')
    - age (Float: 0.0-100.0)
    - hypertension (Integer: 0 or 1)
    - heart_disease (Integer: 0 or 1)
    - smoking_history (String: never/former/current/not current)
    - bmi (Float: typical range 15.0-60.0)
    - HbA1c_level (Float: typical range 4.0-9.0)
    - blood_glucose_level (Integer: typical range 70-300)
    - diabetes (Integer: 0 or 1)

    Here are examples of records:
    Female,45.2,1,0,never,28.5,6.2,140,0
    Male,62.7,1,1,former,32.1,7.1,185,1
    Female,38.9,0,0,current,24.3,5.8,130,0
    Female,22.0,0,0,never,25.77,4.0,145,0
    Male,58.0,0,0,former,36.53,5.8,160,0
    Male,11.0,0,0,No Info,27.59,6.6,100,0
    """,
    
    # Step 2: Prompt with Definitions
    "PROMPT_2": """
    Generate realistic synthetic patient records for diabetes prediction. Here are the features with definitions:

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

    Here are examples of records:
    Female,45.2,1,0,never,28.5,6.2,140,0
    Male,62.7,1,1,former,32.1,7.1,185,1
    Female,38.9,0,0,current,24.3,5.8,130,0
    Female,22.0,0,0,never,25.77,4.0,145,0
    Male,58.0,0,0,former,36.53,5.8,160,0
    Male,11.0,0,0,No Info,27.59,6.6,100,0
    """,
    
    # Step 3: Prompt with Definitions and Metadata
    "PROMPT_3": """
    Generate realistic synthetic patient records for diabetes prediction. Here are the features with definitions and statistical metadata:

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

    Here are examples of records:
    Female,45.2,1,0,never,28.5,6.2,140,0
    Male,62.7,1,1,former,32.1,7.1,185,1
    Female,38.9,0,0,current,24.3,5.8,130,0
    Female,22.0,0,0,never,25.77,4.0,145,0
    Male,58.0,0,0,former,36.53,5.8,160,0
    Male,11.0,0,0,No Info,27.59,6.6,100,0
    """,
    
    # Step 4: Prompt with only Definitions and Metadata (No Examples)
    "PROMPT_4": """
    Generate realistic synthetic patient records for diabetes prediction. Here are the features with definitions and statistical metadata:

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
    """,
}

# Function to convert base prompts to chat format for Gemma
def get_chat_prompt(base_prompt, examples=None):
    """Convert base prompt to Gemma chat format with explicit instructions"""
    # Append examples if provided, otherwise use examples from the base prompt
    example_text = ""
    if examples and len(examples) > 0:
        example_text = "\n".join(examples[:10])  # Use up to 10 examples
    
    chat_prompt = f"""<s>[INST] You are an expert medical data generator.
ONLY generate numerical records in CSV format.
DO NOT output code, explanations, or comments.
ONLY output comma-separated records, one per line.

Each record MUST have 9 fields in exactly this order:
gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

Here is background information:
{base_prompt.strip()}

Generate 10 new records following the exact format shown in the examples.
Do not include any text, code, or explanation - ONLY the records. [/INST]

"""
    
    # Add starter examples to help the model understand the format
    chat_prompt += """Female,45.2,1,0,never,28.5,6.2,140,0
Male,62.7,1,1,former,32.1,7.1,185,1
Female,38.9,0,0,current,24.3,5.8,130,0
Male,51.2,1,0,former,29.3,6.1,145,0
Female,47.8,0,0,never,24.5,5.4,119,0
Male,63.4,1,1,former,33.2,7.2,190,1
Female,31.9,0,0,never,22.1,5.2,115,0
Male,55.6,1,0,current,30.5,6.4,160,1
Female,44.2,0,0,former,25.8,5.7,132,0
Male,68.0,1,1,former,34.7,7.5,200,1
</s>"""
    
    return chat_prompt


def check_hardware_compatibility():
    """Check if running on compatible hardware"""
    if not torch.cuda.is_available():
        print("WARNING: No CUDA device available. This code is optimized for NVIDIA A100 GPUs.")
        return False
        
    try:
        device_name = torch.cuda.get_device_name(0).lower()
        if "a100" in device_name:
            print(f"Compatible hardware detected: {device_name}")
            return True
        else:
            print(f"WARNING: Running on {device_name}. This code is optimized for NVIDIA A100 GPUs and may not work correctly.")
            return False
    except Exception as e:
        print(f"Error checking hardware compatibility: {e}")
        return False


def clean_gpu_memory():
    """Clean GPU memory and cache with robust error handling for MIG"""
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
        if not (0 <= float(age) <= 100):
            return False
        if int(hyp) not in [0, 1]:
            return False
        if int(heart) not in [0, 1]:
            return False
        if smoking not in ["never", "former", "current", "not current", "No Info"]:
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
    except Exception as e:
        # For debugging: print(f"Validation error: {e} for record: {record}")
        return False


def extract_records(text):
    """Extract valid records from generated text"""
    # Split into lines and find lines that might be records
    lines = text.split("\n")
    potential_records = [line.strip() for line in lines if "," in line]

    # Pre-process to remove common prefixes that might be added
    clean_records = []
    for record in potential_records:
        # Remove numbering (like "1. ", "2. ", etc.)
        record = re.sub(r'^\d+\.\s*', '', record)
        # Remove quotes if present
        record = record.strip('"\'')
        # Add to clean records
        clean_records.append(record)

    # Validate each potential record
    valid_records = [record for record in clean_records if validate_record(record)]
    
    # Print stats for debugging
    if len(valid_records) > 0:
        print(f"Successfully extracted {len(valid_records)} valid records from {len(potential_records)} potential records")
    elif len(potential_records) > 0:
        print(f"Found {len(potential_records)} potential records but none were valid")
        print(f"Example potential record: {potential_records[0]}")
    
    return valid_records


# def generate_dataset(
#     model, tokenizer, base_prompt, prompt_name, num_records=100, seed=None
# ):
#     """Generate dataset with optimized approach for MIG environment"""
#     if seed is not None:
#         set_seed(seed)

#     all_records = []
#     pbar = tqdm(total=num_records, desc="Generating records")
    
#     # Track unique records to avoid duplicates
#     unique_records = set()
    
#     # Start with the standard chat prompt
#     chat_prompt = get_chat_prompt(base_prompt)
    
#     attempts = 0
#     max_attempts = num_records * 3  # Allow more attempts than needed records
    
#     # Use these example records to avoid repeating the same initial examples
#     existing_examples = [
#         "Female,45.2,1,0,never,28.5,6.2,140,0",
#         "Male,62.7,1,1,former,32.1,7.1,185,1",
#         "Female,38.9,0,0,current,24.3,5.8,130,0",
#         "Male,51.2,1,0,former,29.3,6.1,145,0",
#         "Female,47.8,0,0,never,24.5,5.4,119,0",
#         "Male,63.4,1,1,former,33.2,7.2,190,1",
#         "Female,31.9,0,0,never,22.1,5.2,115,0",
#         "Male,55.6,1,0,current,30.5,6.4,160,1",
#         "Female,44.2,0,0,former,25.8,5.7,132,0",
#         "Male,68.0,1,1,former,34.7,7.5,200,1"
#     ]
    
#     # Add these to already seen records
#     for record in existing_examples:
#         unique_records.add(record)
    
#     while len(all_records) < num_records and attempts < max_attempts:
#         attempts += 1
        
#         try:
#             clean_gpu_memory()
            
#             # After first few failed attempts, use a more direct prompt
#             if attempts % 3 == 0:
#                 direct_prompt = f"""<s>[INST] Generate {min(20, num_records - len(all_records))} NEW and DIFFERENT synthetic patient records for diabetes.
# THEY MUST BE DIFFERENT from these examples:
# {existing_examples[0]}
# {existing_examples[1]}
# {existing_examples[9]}

# ONLY output records in this EXACT format:
# gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

# IMPORTANT: Create NEW values, not copies of the examples. [/INST]

# Female,42.3,0,0,never,26.8,5.5,125,0
# Male,59.1,1,1,former,31.7,6.9,175,1
# Female,35.6,0,0,current,23.9,5.6,128,0
# Male,53.2,1,0,former,33.3,6.3,147,0
# Female,29.8,0,0,never,22.1,5.1,112,0
# Male,71.4,1,1,former,35.6,7.4,195,1
# Female,33.9,0,0,current,24.7,5.3,122,0
# Male,67.2,1,1,former,36.8,7.6,201,1
# Female,50.3,1,0,never,29.4,6.0,139,0
# Male,44.7,0,0,former,26.9,5.8,131,0
# </s>"""
#                 chat_prompt = direct_prompt
            
#             # Tokenize prompt
#             inputs = tokenizer(chat_prompt, return_tensors="pt")
            
#             # Move to device safely
#             try:
#                 inputs = inputs.to("cuda:0")
#             except RuntimeError as e:
#                 print(f"Warning: Error moving inputs to GPU: {e}")
#                 print("Falling back to CPU")
#                 inputs = inputs
            
#             # Generate text with explicit device management
#             try:
#                 with torch.no_grad():
#                     outputs = model.generate(
#                         **inputs,
#                         max_new_tokens=MAX_NEW_TOKENS,
#                         do_sample=True,
#                         temperature=0.7 + (attempts % 10) * 0.03,  # Small temperature variations
#                         top_p=0.95,
#                         repetition_penalty=1.2,  # Increased to avoid repeats
#                         pad_token_id=tokenizer.pad_token_id,
#                         num_return_sequences=1,
#                     )
                
#                 # Decode output
#                 generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
#                 # Extract only the newly generated part
#                 if "[/INST]" in generated_text:
#                     response_text = generated_text.split("[/INST]")[1].strip()
#                 else:
#                     response_text = generated_text
                    
#                 # Debug - print a preview of the generated text
#                 print(f"\nGenerated text preview (first 300 chars):\n{response_text[:300]}...\n")
                
#                 # Extract records
#                 new_records = extract_records(response_text)
#                 print(f"Found {len(new_records)} potential records in attempt {attempts}")
                
#                 # Filter out duplicates
#                 truly_new_records = []
#                 for record in new_records:
#                     if record not in unique_records:
#                         unique_records.add(record)
#                         truly_new_records.append(record)
                
#                 print(f"After filtering duplicates: {len(truly_new_records)} new records")
                
#                 # Add valid records
#                 for record in truly_new_records:
#                     all_records.append(record)
#                     pbar.update(1)
#                     if len(all_records) >= num_records:
#                         break
                
#                 # If we've made progress, show some records
#                 if truly_new_records:
#                     print(f"Progress: {len(all_records)}/{num_records} records")
#                     print(f"Latest new record: {truly_new_records[0]}")
                    
#                     # Update our examples list with some new records to prevent repetition
#                     if len(truly_new_records) >= 3:
#                         existing_examples = existing_examples[3:] + truly_new_records[:3]
                
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print(f"OOM error in generation: {str(e)}")
#                     clean_gpu_memory()
#                     time.sleep(5)  # Wait for memory to be released
#                 else:
#                     print(f"Error in generation: {str(e)}")
#                     time.sleep(2)
            
#         except Exception as e:
#             print(f"Unexpected error: {str(e)}")
#             time.sleep(2)
    
#     pbar.close()
#     print(f"\nGeneration complete: collected {len(all_records)} records in {attempts} attempts")
    
#     return all_records

def generate_dataset(
    model, tokenizer, base_prompt, prompt_name, num_records=100, seed=None
):
    """Generate dataset with forced diversity to avoid repetition"""
    if seed is not None:
        set_seed(seed)

    all_records = []
    pbar = tqdm(total=num_records, desc="Generating records")
    
    # Track unique records to avoid duplicates
    unique_records = set()
    
    # Start with the standard chat prompt
    chat_prompt = get_chat_prompt(base_prompt)
    
    attempts = 0
    max_attempts = num_records * 3  # Allow more attempts than needed records
    
    # Use these example records to avoid repeating the same initial examples
    existing_examples = [
        "Female,45.2,1,0,never,28.5,6.2,140,0",
        "Male,62.7,1,1,former,32.1,7.1,185,1",
        "Female,38.9,0,0,current,24.3,5.8,130,0",
        "Male,51.2,1,0,former,29.3,6.1,145,0",
        "Female,47.8,0,0,never,24.5,5.4,119,0",
        "Male,63.4,1,1,former,33.2,7.2,190,1",
        "Female,31.9,0,0,never,22.1,5.2,115,0",
        "Male,55.6,1,0,current,30.5,6.4,160,1",
        "Female,44.2,0,0,former,25.8,5.7,132,0",
        "Male,68.0,1,1,former,34.7,7.5,200,1"
    ]
    
    # Add these to already seen records
    for record in existing_examples:
        unique_records.add(record)
    
    while len(all_records) < num_records and attempts < max_attempts:
        attempts += 1
        
        # Force diversity by constructing prompts with varied parameters
        gender = "Male" if attempts % 2 == 0 else "Female"
        age = 25 + (attempts % 50)
        is_hypertension = attempts % 5 == 0
        is_heart_disease = attempts % 10 == 0
        smoking_options = ["never", "former", "current", "not current"]
        smoking = smoking_options[attempts % 4]
        bmi = 20 + (attempts % 25)
        hba1c = 4.5 + (attempts * 0.1) % 4.0
        glucose = 100 + (attempts * 5) % 150
        is_diabetes = 1 if hba1c > 6.5 or glucose > 180 else 0
        
        seed_record = f"{gender},{age},{1 if is_hypertension else 0},{1 if is_heart_disease else 0},{smoking},{bmi:.1f},{hba1c:.1f},{glucose},{is_diabetes}"
        
        # Create a prompt that explicitly requests variation
        direct_prompt = f"""<s>[INST] Generate 10 synthetic patient records for diabetes that are COMPLETELY DIFFERENT from each other and from these examples:

DO NOT repeat any of these existing records:
{existing_examples[0]}
{existing_examples[1]}
{existing_examples[2]}

Generate records SIMILAR to this pattern but with DIFFERENT values:
{seed_record}

Each record MUST have 9 fields in this order:
gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

Make sure:
- gender: 'Male' or 'Female' 
- age: between 18-80
- hypertension: 0 or 1
- heart_disease: 0 or 1
- smoking_history: 'never', 'former', 'current', or 'not current'
- bmi: between 15-60
- HbA1c_level: between 4-9
- blood_glucose_level: between 70-300
- diabetes: 0 or 1

ONLY output the records. No explanations. [/INST]

"""
        chat_prompt = direct_prompt
        
        try:
            clean_gpu_memory()
            
            # Tokenize prompt
            inputs = tokenizer(chat_prompt, return_tensors="pt")
            
            # Move to device safely
            try:
                inputs = inputs.to("cuda:0")
            except RuntimeError as e:
                print(f"Warning: Error moving inputs to GPU: {e}")
                print("Falling back to CPU")
                inputs = inputs
            
            # Generate text with explicit device management
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=0.7 + (attempts % 10) * 0.05,  # More temperature variation
                        top_p=0.95,
                        repetition_penalty=1.3,  # Higher to avoid repetition
                        pad_token_id=tokenizer.pad_token_id,
                        num_return_sequences=1,
                    )
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the newly generated part
                if "[/INST]" in generated_text:
                    response_text = generated_text.split("[/INST]")[1].strip()
                else:
                    response_text = generated_text
                    
                # Debug - print a preview of the generated text
                print(f"\nGenerated text preview (first 300 chars):\n{response_text[:300]}...\n")
                
                # Extract records
                new_records = extract_records(response_text)
                print(f"Found {len(new_records)} potential records in attempt {attempts}")
                
                # Filter out duplicates
                truly_new_records = []
                for record in new_records:
                    if record not in unique_records:
                        unique_records.add(record)
                        truly_new_records.append(record)
                
                print(f"After filtering duplicates: {len(truly_new_records)} new records")
                
                # Add valid records
                for record in truly_new_records:
                    all_records.append(record)
                    pbar.update(1)
                    if len(all_records) >= num_records:
                        break
                
                # If we've made progress, show some records
                if truly_new_records:
                    print(f"Progress: {len(all_records)}/{num_records} records")
                    print(f"Latest new record: {truly_new_records[0]}")
                    
                    # Update our examples list with some new records to prevent repetition
                    if len(truly_new_records) >= 3:
                        existing_examples = existing_examples[3:] + truly_new_records[:3]
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in generation: {str(e)}")
                    clean_gpu_memory()
                    time.sleep(5)  # Wait for memory to be released
                else:
                    print(f"Error in generation: {str(e)}")
                    time.sleep(2)
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            time.sleep(2)
    
    pbar.close()
    print(f"\nGeneration complete: collected {len(all_records)} records in {attempts} attempts")
    
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

    # Safety check for empty records
    if not records:
        print(f"Warning: No records to save for {prompt_name} (Experiment {experiment_num})")
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
    """Get current system resource usage with error handling for MIG"""
    system_info = {}
    
    try:
        # Get CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        system_info.update({
            "cpu_percent": cpu_percent,
            "memory_total_gb": memory.total / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_percent": memory.percent,
        })
    except Exception as e:
        print(f"Warning: Error getting CPU info: {e}")
        system_info["cpu_error"] = str(e)

    # Add GPU information if available - with careful error handling for MIG
    try:
        if torch.cuda.is_available():
            # Add CUDA info from torch directly
            system_info.update({
                "cuda_available": True,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
            })
            
            # Try to get device name and memory info
            try:
                system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
                system_info["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
                system_info["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
            except Exception as e:
                system_info["cuda_device_info_error"] = str(e)
            
            # Try GPUtil as fallback
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info = []
                    for i, gpu in enumerate(gpus):
                        gpu_info.append({
                            "gpu_id": i,
                            "gpu_name": gpu.name,
                            "gpu_memory_total_mb": gpu.memoryTotal,
                            "gpu_memory_used_mb": gpu.memoryUsed,
                            "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            "gpu_temperature": gpu.temperature,
                        })
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
    print(f"Starting Gemma data generation experiment")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Check hardware compatibility
    check_hardware_compatibility()
    
    # Add token for Gemma access if needed - uncomment if you need to use a token
    # os.environ["HF_TOKEN"] = "your_huggingface_token"
    
    try:
        # Load model and tokenizer with 4-bit quantization
        print(f"Loading model {MODEL_NAME} on CUDA")
        
        # Try 4-bit quantization first (better for MIG constraints)
        try:
            print("Attempting to load with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            # Make sure tokenizer is loaded first
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with device_map explicit - not "auto" for MIG
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map={"": 0},  # Explicitly map to the first device
            )
            
            print("Successfully loaded with 4-bit quantization")
        
        except Exception as e:
            print(f"Error loading with 4-bit quantization: {e}")
            
            print("Falling back to standard loading...")
            # Standard loading approach
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map={"": 0},  # Explicit device mapping for MIG
            )
            
            print("Successfully loaded model with standard config")

        # Store all experiment metrics
        all_metrics = []

        # Run multiple experiments with different seeds
        base_seed = int(time.time())

        for experiment_num in range(1, NUM_EXPERIMENTS + 1):
            experiment_seed = base_seed + experiment_num
            print(
                f"\n=== Starting Experiment {experiment_num} (Seed: {experiment_seed}) ==="
            )

            for prompt_name, base_prompt in BASE_PROMPTS.items():
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
                    base_prompt,
                    prompt_name,
                    num_records=NUM_RECORDS,
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

                # Clean up between prompts
                clean_gpu_memory()
                time.sleep(3)  # Extra wait for memory to settle

            print(f"\n=== Completed Experiment {experiment_num} ===")

        # Save summary of all experiments
        summary_file = os.path.join(OUTPUT_DIR, "metrics", "experiments_summary.json")
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "total_experiments": NUM_EXPERIMENTS,
                    "model_name": MODEL_NAME,
                    "device": "cuda",
                    "batch_size": BATCH_SIZE,
                    "records_per_prompt": NUM_RECORDS,
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