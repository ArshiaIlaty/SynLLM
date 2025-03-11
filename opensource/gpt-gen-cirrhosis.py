import gc
import json
import os
import re
import time
import signal
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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Configuration
MODEL_NAME = "gpt2"
OUTPUT_DIR = "gpt2-cirrhosis-data-100-5experiments"
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
        print(f"Initial memory allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**2:.2f} MB")
        print(f"Initial memory reserved: {torch.cuda.memory_reserved(gpu_id) / 1024**2:.2f} MB")
        
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
    # Task 1: Basic Prompt with Examples
    "PROMPT_1": {
        "text": """
        Generate synthetic cirrhosis data in CSV format:
        ID,N_Days,Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage

        Examples:
        53, 1000, D, D-penicillamine, 24621, F, N, Y, N, N, 2.6, 309.5, 3.1, 94, 6456.2, 56.76, 108, 214, 11, 4.0
        242, 1810, C, D-penicillamine, 23585, F, N, Y, N, N, 1.9, 354, 2.97, 86, 1553, 196.85, 152, 277, 9.9, 3.0
        9, 2400, D, D-penicillamine, 15526, F, N, N, Y, N, 3.2, 562, 3.08, 79, 2276, 144.15, 88, 251, 11, 2.0
        272, 1525, C, D-penicillamine, 14025, F, N, N, N, N, 0.5, 226, 2.93, 22, 674, 58, 85, 153, 9.8, 1.0

        Generate more records following these rules:
        - ID: Unique identifier for each record
        - N_Days: Between 41 and 4795
        - Status: C or D
        - Drug: D-penicillamine or Placebo
        - Age: Between 9598 and 28650
        - Sex: M or F
        - Ascites: Y or N
        - Hepatomegaly: Y or N
        - Spiders: Y or N
        - Edema: N, S, or Y
        - Bilirubin: Between 0.3 and 28.0
        - Cholesterol: Between 120.0 and 1775.0
        - Albumin: Between 1.96 and 4.64
        - Copper: Between 4.0 and 588.0
        - Alk_Phos: Between 289.0 and 13862.4
        - SGOT: Between 26.35 and 457.25
        - Tryglicerides: Between 33.0 and 598.0
        - Platelets: Between 62.0 and 721.0
        - Prothrombin: Between 9.0 and 18.0
        - Stage: 4.0 (34.4%), 3.0 (38.5%), 2.0 (22.0%), or 1.0 (5.0%)
        """,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.7,
        "use_cpu": False
    },
    # Task 2: Prompt with Definitions
    "PROMPT_2": {
        "text": """
        Generate synthetic cirrhosis data with these definitions:
        ID,N_Days,Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage

        Definitions:
        - ID: Unique identifier for each record
        - N_Days: Measurement of n days (range: 41-4795)
        - Status: C or D
        - Drug: D-penicillamine or Placebo
        - Age: Measurement of age (range: 9598-28650)
        - Sex: M or F 
        - Ascites: Y or N
        - Hepatomegaly: Y or N
        - Spiders: Y or N
        - Edema: N, S, or Y
        - Bilirubin: Measurement of bilirubin (range: 0.3-28.0)
        - Cholesterol: Measurement of cholesterol (range: 120.0-1775.0)
        - Albumin: Minimum albu value (range: 1.96-4.64)
        - Copper: Measurement of copper (range: 4.0-588.0)
        - Alk_Phos: Measurement of alk phos (range: 289.0-13862.4)
        - SGOT: Measurement of sgot (range: 26.35-457.25)
        - Tryglicerides: Measurement of tryglicerides (range: 33.0-598.0)
        - Platelets: Measurement of platelets (range: 62.0-721.0)
        - Prothrombin: Measurement of prothrombin (range: 9.0-18.0)
        - Stage: Classification with '4.0' (34.4% of cases), '3.0' (38.5% of cases), '2.0' (22.0% of cases), '1.0' (5.0% of cases)

        Examples:
        54, 1434, D, D-penicillamine, 14317, F, Y, Y, Y, Y, 1.3, 288, 3.4, 262, 5487.2, 73.53, 125, 254, 11, 4.0
        385, 1635, C, D-penicillamine, 20089, F, N, Y, N, N, 0.7, 309.5, 2.93, 73, 1259, 114.7, 108, 209, 10.6, 3.0
        347, 2812, D, D-penicillamine, 18628, F, N, Y, N, N, 3.4, 309.5, 3.92, 73, 1259, 114.7, 108, 251, 9.3, 2.0
        272, 1525, C, D-penicillamine, 14025, F, N, N, N, N, 0.5, 226, 2.93, 22, 674, 58, 85, 153, 9.8, 1.0
        """,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.7, 
        "use_cpu": False
    },
    # Task 3: Prompt with Metadata
    "PROMPT_3": {
        "text": """
        Generate synthetic cirrhosis data with these statistics:
        ID,N_Days,Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage

        Statistical properties:
        | Feature | Overall Mean | 1.0 Mean | 2.0 Mean | 3.0 Mean | 4.0 Mean | Min | Max | Std Dev |
        |---------|--------------|-------------|-------------|-------------|-------------|-----|-----|---------|
        | N_Days | 1917.7823 | 2654.8095 | 2389.837 | 1996.9006 | 1420.25 | 41 | 4795 | 1104.673 |
        | Age | 18533.3517 | 17108.7143 | 18067.4348 | 17997.5155 | 19637.875 | 9598 | 28650 | 3815.8451 |
        | Bilirubin | 3.2208 | 1.3619 | 2.4533 | 2.823 | 4.4271 | 0.3 | 28.0 | 4.4075 |
        | Cholesterol | 350.2727 | 283.7143 | 338.462 | 385.4658 | 328.1771 | 120.0 | 1775.0 | 193.1239 |
        | Albumin | 3.4974 | 3.7052 | 3.6071 | 3.5821 | 3.3024 | 1.96 | 4.64 | 0.425 |
        | Copper | 91.2799 | 65.2381 | 69.4348 | 87.2174 | 113.5764 | 4.0 | 588.0 | 74.4855 |
        | Alk_Phos | 1799.145 | 1590.5905 | 1665.3022 | 1872.5118 | 1833.0417 | 289.0 | 13862.4 | 1875.122 |
        | SGOT | 120.5641 | 91.66 | 115.328 | 121.8752 | 126.6585 | 26.35 | 457.25 | 49.0851 |
        | Tryglicerides | 119.2679 | 97.381 | 111.9457 | 123.9565 | 121.8958 | 33.0 | 598.0 | 54.0507 |
        | Platelets | 256.866 | 289.8571 | 284.5543 | 265.0373 | 225.2292 | 62.0 | 721.0 | 97.0249 |
        | Prothrombin | 10.7311 | 10.7714 | 10.525 | 10.4745 | 11.1437 | 9.0 | 18.0 | 1.0196 |

        Key relationships:
        - Stage distribution: 4.0 (34.4%), 3.0 (38.5%), 2.0 (22.0%), 1.0 (5.0%)
        - Stage 4.0 typically has lower N_Days, higher Bilirubin, higher Copper
        - Stage 1.0 typically has higher N_Days, lower Bilirubin
        
        Examples:
        394, 1367, C, D-penicillamine, 20819, F, N, Y, N, S, 2, 309.5, 3.07, 73, 1259, 114.7, 108, 80, 12.1, 4.0
        77, 326, D, Placebo, 18199, F, N, Y, Y, S, 6.6, 244, 3.41, 199, 1819, 170.5, 91, 132, 12.1, 3.0
        89, 1741, D, D-penicillamine, 19155, F, N, Y, N, N, 2, 408, 3.65, 50, 1083, 110.05, 98, 200, 11.4, 2.0
        352, 2716, C, D-penicillamine, 19358, F, N, Y, N, N, 0.6, 309.5, 4.19, 73, 1259, 114.7, 108, 330, 9.9, 1.0
        """,
        "max_tokens": MAX_NEW_TOKENS // 2,  # Half tokens for PROMPT_3
        "temperature": 0.6,  # Lower temperature for more focused generation
        "use_cpu": True  # Force CPU for PROMPT_3 which was successful with diabetes
    },
    # Task 4: Prompt with Rules - NO EXAMPLES
    "PROMPT_4": {
        "text": """
        Generate synthetic cirrhosis data following these rules: ID,N_Days,Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage
            
        Rules:
        1. Generate Stage with proper distribution: 4.0 (34.4%), 3.0 (38.5%), 2.0 (22.0%), 1.0 (5.0%)
        2. Higher Stage (4.0) usually correlates with lower N_Days, higher Bilirubin, higher Copper
        3. Lower Stage (1.0) usually correlates with higher N_Days, lower Bilirubin
        4. Follow these constraints for categorical variables:
           - Status: C or D only
           - Drug: D-penicillamine or Placebo only
           - Sex: M or F only
           - Ascites: Y or N only
           - Hepatomegaly: Y or N only
           - Spiders: Y or N only
           - Edema: N, S, or Y only
        5. All numerical values must stay within the Min-Max range shown in the table
        """,
        "max_tokens": MAX_NEW_TOKENS // 3,  # Even fewer tokens for PROMPT_4
        "temperature": 0.5,  # Lower temperature for more focused generation
        "use_cpu": True  # Force CPU for PROMPT_4
    },
}


def validate_record(record):
    """Validate if a generated record matches expected format and constraints for cirrhosis data"""
    try:
        parts = record.strip().split(",")
        if len(parts) != 20:  # Should have 20 fields for cirrhosis data
            return False

        # Extract all parts for validation
        id_val, n_days, status, drug, age, sex, ascites, hepatomegaly, spiders, edema, \
        bilirubin, cholesterol, albumin, copper, alk_phos, sgot, tryglicerides, platelets, \
        prothrombin, stage = [p.strip() for p in parts]

        # Validate ID - should be a number
        try:
            id_int = int(id_val)
            if id_int <= 0:
                return False
        except ValueError:
            return False
            
        # Validate N_Days
        try:
            n_days_float = float(n_days)
            if not (41 <= n_days_float <= 4795):
                return False
        except ValueError:
            return False
            
        # Validate Status
        if status not in ["C", "D"]:
            return False
            
        # Validate Drug
        if drug not in ["D-penicillamine", "Placebo"]:
            return False
            
        # Validate Age
        try:
            age_float = float(age)
            if not (9598 <= age_float <= 28650):
                return False
        except ValueError:
            return False
            
        # Validate Sex
        if sex not in ["M", "F"]:
            return False
            
        # Validate Ascites
        if ascites not in ["Y", "N"]:
            return False
            
        # Validate Hepatomegaly
        if hepatomegaly not in ["Y", "N"]:
            return False
            
        # Validate Spiders
        if spiders not in ["Y", "N"]:
            return False
            
        # Validate Edema
        if edema not in ["N", "S", "Y"]:
            return False
            
        # Validate numerical values
        try:
            # Bilirubin
            bili_float = float(bilirubin)
            if not (0.3 <= bili_float <= 28.0):
                return False
                
            # Cholesterol
            chol_float = float(cholesterol)
            if not (120.0 <= chol_float <= 1775.0):
                return False
                
            # Albumin
            alb_float = float(albumin)
            if not (1.96 <= alb_float <= 4.64):
                return False
                
            # Copper
            copper_float = float(copper)
            if not (4.0 <= copper_float <= 588.0):
                return False
                
            # Alk_Phos
            alk_float = float(alk_phos)
            if not (289.0 <= alk_float <= 13862.4):
                return False
                
            # SGOT
            sgot_float = float(sgot)
            if not (26.35 <= sgot_float <= 457.25):
                return False
                
            # Tryglicerides
            trig_float = float(tryglicerides)
            if not (33.0 <= trig_float <= 598.0):
                return False
                
            # Platelets
            plat_float = float(platelets)
            if not (62.0 <= plat_float <= 721.0):
                return False
                
            # Prothrombin
            pro_float = float(prothrombin)
            if not (9.0 <= pro_float <= 18.0):
                return False
                
            # Stage
            if stage not in ["1.0", "2.0", "3.0", "4.0"]:
                return False
        except ValueError:
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


def generate_dataset_for_prompt4(generator, prompt_config, num_records=100, seed=None, experiment_num=1):
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
        f.write(f"PROMPT_4 generation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original parameters: max_tokens={max_tokens}, temperature={temperature}\n\n")
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
            ID,N_Days,Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage
            
            For example, a record might look like:
            ID,NDays,Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,AlkPhos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage
            """,
            "temperature": 0.5,
            "max_tokens": 500,
            "attempts": 5
        },
        # Approach 2: Add more specific instructions
        {
            "name": "Detailed instructions",
            "prompt_addition": """
            Please generate exactly 10 records with the following format:
            ID,N_Days,Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage
            
            Each record should be on a new line and follow all the rules above.
            Follow the Stage distribution: 4.0 (34.4%), 3.0 (38.5%), 2.0 (22.0%), 1.0 (5.0%)
            """,
            "temperature": 0.7,
            "max_tokens": 400,
            "attempts": 5
        },
        # Approach 3: Generate one record at a time with specific format
        {
            "name": "Single record generation",
            "prompt_addition": """
            Generate a single record in exactly this format:
            ID,N_Days,Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage
            
            Just output the record values with no additional text or explanation.
            """,
            "temperature": 0.6,
            "max_tokens": 100,
            "attempts": 15
        },
        # Approach 4: Use lower temperature
        {
            "name": "Lower temperature",
            "prompt_addition": "",  # No addition
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "attempts": 5
        },
        # Approach 5: Use higher temperature
        {
            "name": "Higher temperature",
            "prompt_addition": "",  # No addition
            "temperature": 0.9,
            "max_tokens": max_tokens,
            "attempts": 5
        }
    ]
    
    # Try each approach until we have enough records
    for approach in approaches:
        if len(all_records) >= num_records:
            break
            
        print(f"\nTrying PROMPT_4 approach: {approach['name']}")
        with open(log_file, "a") as f:
            f.write(f"\n\n=== Trying approach: {approach['name']} ===\n")
            f.write(f"Temperature: {approach['temperature']}, Max tokens: {approach['max_tokens']}\n")
            
        # Create modified prompt
        modified_prompt = prompt_text + approach['prompt_addition']
        
        # Try multiple attempts with this approach
        for attempt in range(approach['attempts']):
            if len(all_records) >= num_records:
                break
                
            try:
                # Clear memory
                clear_memory()
                
                # Generate text
                output = cpu_generator(
                    modified_prompt,
                    max_new_tokens=approach['max_tokens'],
                    do_sample=True,
                    temperature=approach['temperature'],
                    pad_token_id=50256,
                    num_return_sequences=1,
                )
                
                # Get generated text
                generated_text = output[0]["generated_text"] if isinstance(output, list) else output["generated_text"]
                
                # Log a sample
                with open(log_file, "a") as f:
                    f.write(f"\nAttempt {attempt+1}:\n")
                    f.write("Generated text sample:\n")
                    f.write(generated_text[:500] + "...\n" if len(generated_text) > 500 else generated_text)
                
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
        
        # Define fields and constraints for cirrhosis data
        fields = [
            {"name": "ID", "min": 1, "max": 500},
            {"name": "N_Days", "min": 41, "max": 4795, "mean": 1917.7823, "std": 1104.673},
            {"name": "Status", "values": ["C", "D"]},
            {"name": "Drug", "values": ["D-penicillamine", "Placebo"]},
            {"name": "Age", "min": 9598, "max": 28650, "mean": 18533.3517, "std": 3815.8451},
            {"name": "Sex", "values": ["M", "F"]},
            {"name": "Ascites", "values": ["Y", "N"]},
            {"name": "Hepatomegaly", "values": ["Y", "N"]},
            {"name": "Spiders", "values": ["Y", "N"]},
            {"name": "Edema", "values": ["N", "S", "Y"]},
            {"name": "Bilirubin", "min": 0.3, "max": 28.0, "mean": 3.2208, "std": 4.4075},
            {"name": "Cholesterol", "min": 120.0, "max": 1775.0, "mean": 350.2727, "std": 193.1239},
            {"name": "Albumin", "min": 1.96, "max": 4.64, "mean": 3.4974, "std": 0.425},
            {"name": "Copper", "min": 4.0, "max": 588.0, "mean": 91.2799, "std": 74.4855},
            {"name": "Alk_Phos", "min": 289.0, "max": 13862.4, "mean": 1799.145, "std": 1875.122},
            {"name": "SGOT", "min": 26.35, "max": 457.25, "mean": 120.5641, "std": 49.0851},
            {"name": "Tryglicerides", "min": 33.0, "max": 598.0, "mean": 119.2679, "std": 54.0507},
            {"name": "Platelets", "min": 62.0, "max": 721.0, "mean": 256.866, "std": 97.0249},
            {"name": "Prothrombin", "min": 9.0, "max": 18.0, "mean": 10.7311, "std": 1.0196},
            {"name": "Stage", "values": ["1.0", "2.0", "3.0", "4.0"], 
             "distribution": [0.05, 0.22, 0.385, 0.344]}  # Stage distribution
        ]
        
        # Stage-specific means for key features
        stage_means = {
            "1.0": {
                "N_Days": 2654.8095,
                "Age": 17108.7143,
                "Bilirubin": 1.3619,
                "Cholesterol": 283.7143,
                "Albumin": 3.7052,
                "Copper": 65.2381,
                "Platelets": 289.8571,
            },
            "2.0": {
                "N_Days": 2389.837,
                "Age": 18067.4348,
                "Bilirubin": 2.4533,
                "Cholesterol": 338.462,
                "Albumin": 3.6071,
                "Copper": 69.4348,
                "Platelets": 284.5543,
            },
            "3.0": {
                "N_Days": 1996.9006,
                "Age": 17997.5155,
                "Bilirubin": 2.823,
                "Cholesterol": 385.4658,
                "Albumin": 3.5821,
                "Copper": 87.2174,
                "Platelets": 265.0373,
            },
            "4.0": {
                "N_Days": 1420.25,
                "Age": 19637.875,
                "Bilirubin": 4.4271,
                "Cholesterol": 328.1771,
                "Albumin": 3.3024,
                "Copper": 113.5764,
                "Platelets": 225.2292,
            }
        }
        
        # Generate records needed to complete the set
        needed_records = num_records - len(all_records)
        
        # Pre-determine stage distribution to ensure proper ratios
        import numpy as np
        np.random.seed(seed + experiment_num) if seed else np.random.seed()
        
        # Generate stages based on required distribution
        stages = []
        stage_values = ["1.0", "2.0", "3.0", "4.0"]
        stage_probs = [0.05, 0.22, 0.385, 0.344]
        
        for _ in range(needed_records):
            stage = np.random.choice(stage_values, p=stage_probs)
            stages.append(stage)
        
        # Track used IDs to ensure uniqueness
        used_ids = set()
        
        for i in range(needed_records):
            record_values = []
            stage = stages[i]
            
            # Generate each field
            for field in fields:
                try:
                    if field["name"] == "Stage":
                        # Already predetermined
                        record_values.append(stage)
                        continue
                        
                    if field["name"] == "ID":
                        # Generate unique ID
                        while True:
                            id_val = np.random.randint(field["min"], field["max"])
                            if id_val not in used_ids:
                                used_ids.add(id_val)
                                record_values.append(str(id_val))
                                break
                        continue
                        
                    # For categorical fields
                    if "values" in field:
                        if field["name"] in ["Status", "Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]:
                            value = np.random.choice(field["values"])
                            record_values.append(value)
                            continue
                    
                    # For numerical fields, use stage-specific means when available
                    if field["name"] in stage_means[stage]:
                        # Generate value from normal distribution using stage-specific mean
                        stage_mean = stage_means[stage][field["name"]]
                        std = field["std"]
                        
                        # Generate value with stage-specific mean and global std
                        while True:
                            value = np.random.normal(stage_mean, std)
                            if field["min"] <= value <= field["max"]:
                                break
                        
                        # Format based on field type
                        if field["name"] in ["N_Days", "Age"]:
                            value = int(round(value))
                        elif field["name"] in ["Bilirubin", "Albumin", "Prothrombin"]:
                            value = round(value, 2)
                        else:
                            value = round(value, 1)
                            
                        record_values.append(str(value))
                    else:
                        # Regular numerical field without stage-specific mean
                        while True:
                            value = np.random.normal(field["mean"], field["std"])
                            if field["min"] <= value <= field["max"]:
                                break
                                
                        # Format based on field type
                        if field["name"] in ["N_Days", "Age"]:
                            value = int(round(value))
                        elif field["name"] in ["Bilirubin", "Albumin", "Prothrombin"]:
                            value = round(value, 2)
                        else:
                            value = round(value, 1)
                            
                        record_values.append(str(value))
                        
                except Exception as e:
                    print(f"Error generating {field['name']}: {e}")
                    # Use default values in case of error
                    if "values" in field:
                        value = field["values"][0]
                    else:
                        value = str(round(field["min"] + (field["max"] - field["min"])/2, 2))
                    record_values.append(value)
            
            # Create record string
            record = ", ".join(record_values)
            
            # Validate and add
            if validate_record(record):
                all_records.append(record)
                pbar.update(1)
                with open(log_file, "a") as f:
                    f.write(f"Generated step-by-step record: {record}\n")
            else:
                with open(log_file, "a") as f:
                    f.write(f"Invalid record generated: {record}\n")
    
    pbar.close()
    
    # Log completion
    with open(log_file, "a") as f:
        f.write(f"\nPROMPT_4 generation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total records generated: {len(all_records)}\n")
        if len(all_records) < num_records:
            f.write(f"WARNING: Only generated {len(all_records)}/{num_records} requested records\n")
    
    return all_records


def generate_dataset(generator, prompt_config, num_records=100, seed=None, experiment_num=1, prompt_name=""):
    """Generate dataset with improved handling and experiment-specific seed"""
    # Special case for PROMPT_4
    if prompt_name == "PROMPT_4":
        return generate_dataset_for_prompt4(generator, prompt_config, num_records, seed, experiment_num)
    
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
    log_file = os.path.join(LOG_DIR, f"generation_log_exp{experiment_num}_{prompt_name}.txt")
    with open(log_file, "w") as f:
        f.write(f"Generation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters: max_tokens={max_tokens}, temperature={temperature}, device={current_device}\n\n")

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
                        f.write(generated_text[:500] + "...\n" if len(generated_text) > 500 else generated_text)
                    
                    # Extract and validate records
                    new_records = extract_records(generated_text)
                    all_new_records.extend(new_records)
                    
                    # Log found records
                    with open(log_file, "a") as f:
                        f.write(f"\nFound {len(new_records)} valid records\n")
                        if new_records:
                            f.write("Sample records:\n")
                            for i, record in enumerate(new_records[:3]):  # Log just a few samples
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
                        f.write(f"\nNo valid records found, changing temperature for next attempt\n")
                    temperature = 0.8 if temperature < 0.7 else 0.6
                    time.sleep(1)  # Wait a moment before retrying
                    
            except TimeoutException:
                with open(log_file, "a") as f:
                    f.write(f"\n!!! Generation timed out after {TIMEOUT_SECONDS} seconds !!!\n")
                print(f"\nWarning: Generation timed out for {prompt_name}. Trying again with different parameters.")
                
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
        f.write(f"\nGeneration completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total records generated: {len(all_records)}\n")
    
    return all_records

def get_system_info():
    """Get current system resource usage - with better error handling for MIG"""
    system_info = {}
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        system_info.update({
            "cpu_percent": cpu_percent,
            "memory_total_gb": memory.total / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_percent": memory.percent,
        })
    except Exception as e:
        print(f"Warning: Error getting system info: {e}")
        system_info["system_error"] = str(e)

    # Add GPU information if available - with careful error handling
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                system_info.update({
                    "gpu_name": gpu.name,
                    "gpu_memory_total_mb": gpu.memoryTotal,
                    "gpu_memory_used_mb": gpu.memoryUsed,
                    "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_temperature": gpu.temperature,
                })
            else:
                system_info["gpu_info"] = "No GPUs found by GPUtil"
                
            # Add torch.cuda info as fallback
            system_info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            })
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
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Experiment {experiment_num}, {prompt_name}: Complete ({num_records} records in {(end_time - start_time):.1f}s)\n")

    return metrics


def save_and_validate_dataset(records, prompt_name, experiment_num):
    """Save dataset and perform basic validation with experiment number"""
    # Create experiment-specific directory
    experiment_dir = os.path.join(OUTPUT_DIR, f"experiment_{experiment_num}")
    os.makedirs(experiment_dir, exist_ok=True)

    output_path = os.path.join(experiment_dir, f"{prompt_name}_data.csv")

    # Create DataFrame for cirrhosis data
    columns = [
        "ID",
        "N_Days",
        "Status",
        "Drug",
        "Age",
        "Sex",
        "Ascites",
        "Hepatomegaly",
        "Spiders",
        "Edema",
        "Bilirubin",
        "Cholesterol",
        "Albumin",
        "Copper",
        "Alk_Phos",
        "SGOT",
        "Tryglicerides",
        "Platelets",
        "Prothrombin",
        "Stage"
    ]

    # Clean up strings and split by comma
    cleaned_records = []
    for record in records:
        # Replace any spaces after commas
        parts = [part.strip() for part in record.split(',')]
        cleaned_records.append(parts)

    df = pd.DataFrame(cleaned_records, columns=columns)

    # Convert datatypes
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df["N_Days"] = pd.to_numeric(df["N_Days"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Bilirubin"] = pd.to_numeric(df["Bilirubin"], errors="coerce")
    df["Cholesterol"] = pd.to_numeric(df["Cholesterol"], errors="coerce")
    df["Albumin"] = pd.to_numeric(df["Albumin"], errors="coerce")
    df["Copper"] = pd.to_numeric(df["Copper"], errors="coerce")
    df["Alk_Phos"] = pd.to_numeric(df["Alk_Phos"], errors="coerce")
    df["SGOT"] = pd.to_numeric(df["SGOT"], errors="coerce")
    df["Tryglicerides"] = pd.to_numeric(df["Tryglicerides"], errors="coerce")
    df["Platelets"] = pd.to_numeric(df["Platelets"], errors="coerce")
    df["Prothrombin"] = pd.to_numeric(df["Prothrombin"], errors="coerce")
    df["Stage"] = pd.to_numeric(df["Stage"], errors="coerce")

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print basic statistics
    print(f"\nDataset statistics for {prompt_name} (Experiment {experiment_num}):")
    print(f"Total records: {len(df)}")
    print("\nNumerical columns summary:")
    print(df.describe())
    print("\nCategorical columns value counts:")
    for col in [
        "Status",
        "Drug",
        "Sex",
        "Ascites",
        "Hepatomegaly",
        "Spiders",
        "Edema",
        "Stage",
    ]:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True))
    
    # Calculate and print Stage distribution
    stage_dist = df["Stage"].value_counts(normalize=True)
    print("\nStage distribution:")
    print(f"1.0: {stage_dist.get(1.0, 0)*100:.2f}%")
    print(f"2.0: {stage_dist.get(2.0, 0)*100:.2f}%")
    print(f"3.0: {stage_dist.get(3.0, 0)*100:.2f}%")
    print(f"4.0: {stage_dist.get(4.0, 0)*100:.2f}%")
    
    # Check if stage distribution is close to expected
    expected_dist = {1.0: 0.05, 2.0: 0.22, 3.0: 0.385, 4.0: 0.344}
    for stage, expected in expected_dist.items():
        actual = stage_dist.get(stage, 0)
        if abs(actual - expected) > 0.1:  # Allow 10% deviation
            print(f"WARNING: Stage {stage} percentage ({actual*100:.2f}%) deviates significantly from expected {expected*100:.2f}%")


def main():
    # Create initial status file
    status_file = os.path.join(OUTPUT_DIR, "experiment_status.txt")
    with open(status_file, "w") as f:
        f.write(f"Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Device: {'CUDA' if DEVICE >= 0 else 'CPU'}\n")
        f.write(f"Experiments: {NUM_EXPERIMENTS}\n")
        f.write("----------------------------------------\n")
    
    print(f"Starting GPT-2 cirrhosis data generation on device {DEVICE}")
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
                f.write(f"\nExperiment {experiment_num} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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
                        prompt_name=prompt_name
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
                    print(f"Error in experiment {experiment_num}, prompt {prompt_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Log the error
                    with open(status_file, "a") as f:
                        f.write(f"ERROR in {prompt_name}: {str(e)}\n")
                    # Continue to next prompt instead of exiting
                    continue

            print(f"\n=== Completed Experiment {experiment_num} ===")
            with open(status_file, "a") as f:
                f.write(f"Experiment {experiment_num} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Save summary of all experiments
        summary_file = os.path.join(OUTPUT_DIR, "metrics", "experiments_summary.json")
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "total_experiments": NUM_EXPERIMENTS,
                    "model_name": MODEL_NAME,
                    "device": DEVICE,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "metrics": all_metrics,
                },
                f,
                indent=4,
            )
        
        print(f"\n=== All experiments completed successfully ===")
        print(f"Results saved to {OUTPUT_DIR}")
        with open(status_file, "a") as f:
            f.write(f"\nAll experiments completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
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
            print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("CUDA not available")
        
        # Initialize a basic progress file
        progress_file = os.path.join(OUTPUT_DIR, "progress.log")
        with open(progress_file, "w") as f:
            f.write(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Run the main process
        main()
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting gracefully...")
        # Log the interruption
        with open(os.path.join(OUTPUT_DIR, "experiment_status.txt"), "a") as f:
            f.write(f"\nProcess interrupted by user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"\n\nUnhandled exception: {str(e)}")
        import traceback
        traceback.print_exc()
        # Log the error
        with open(os.path.join(OUTPUT_DIR, "experiment_status.txt"), "a") as f:
            f.write(f"\nUnhandled exception: {str(e)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    finally:
        print("\nExiting script")
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
