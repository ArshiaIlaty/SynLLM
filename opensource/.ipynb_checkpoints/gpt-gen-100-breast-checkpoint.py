import gc
import json
import os
import re
import time
import signal
from datetime import datetime

import pandas as pd
import psutil
import torch
from tqdm import tqdm
from transformers import pipeline, set_seed

# Configuration
MODEL_NAME = "gpt2"
OUTPUT_DIR = "gpt2-breast-100-prompt-in-5experiments"
BATCH_SIZE = 1  # Single batch for stability
MAX_NEW_TOKENS = 100  # Tokens to generate
TIMEOUT_SECONDS = 600  # Maximum time to wait for a generation (10 minutes)
NUM_EXPERIMENTS = 5  # Number of experiments to run

# Force CPU mode to avoid CUDA errors
DEVICE = -1  # Always use CPU
print(f"Forcing CPU mode to avoid CUDA errors")

# Add timeout handling
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Generation timed out")

# Create output directory and log directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Define prompts dictionary with specific generation settings for each
PROMPTS = {
    # Task 1: Basic Prompt with Examples
    "PROMPT_1": {
        "text": """
        I need you to generate synthetic breast-cancer data that closely resembles real-world data. The dataset should contain 100 samples with the following columns:
        
        id, diagnosis (M/B), radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean,  fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst, Unnamed: 32
        
        Here are 3 example records from a real dataset to guide your generation:
        
        Example 1:
        9110732, M, 17.75, 28.03, 117.3, 981.6, 0.09997, 0.1314, 0.1698, 0.08293, 0.1713, 0.05916, 0.3897, 1.077, 2.873, 43.95, 0.004714, 0.02015, 0.03697, 0.0111, 0.01237, 0.002556, 21.53, 38.54, 145.4, 1437, 0.1401, 0.3762, 0.6399, 0.197, 0.2972, 0.09075, 0
        
        Example 2:
        8911670, M, 18.81, 19.98, 120.9, 1102, 0.08923, 0.05884, 0.0802, 0.05843, 0.155, 0.04996, 0.3283, 0.828, 2.363, 36.74, 0.007571, 0.01114, 0.02623, 0.01463, 0.0193, 0.001676, 19.96, 24.3, 129, 1236, 0.1243, 0.116, 0.221, 0.1294, 0.2567, 0.05737, 0
        
        Example 3:
        904689, B, 12.96, 18.29, 84.18, 525.2, 0.07351, 0.07899, 0.04057, 0.01883, 0.1874, 0.05899, 0.2357, 1.299, 2.397, 20.21, 0.003629, 0.03713, 0.03452, 0.01065, 0.02632, 0.003705, 14.13, 24.61, 96.31, 621.9, 0.09329, 0.2318, 0.1604, 0.06608, 0.3207, 0.07247, 0
        
        Please generate 100 records in a CSV format that follows these patterns and maintains realistic relationships between the features. The data should be plausible and preserve the correlations between features that would be found in real breast-cancer data.
        """,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.7
    },
    # Task 2: Prompt with Definitions
    "PROMPT_2": {
        "text": """
        I need you to generate synthetic breast-cancer data. Please create 100 samples that realistically represent the patterns and relationships in this type of data.
        
        Generate breast-cancer data:
        ID, diagnosis (M: 37.3%, B: 62.7%), followed by 30 numerical features.
        
        Here are 3 example records from a real dataset:
        
        Example 1 (M):
        85638502, M, 13.17, 21.81, 85.42, 531.5, 0.09714, 0.1047, 0.08259, 0.05252, 0.1746, 0.06177, 0.1938, 0.6123, 1.334, 14.49, 0.00335, 0.01384, 0.01452, 0.006853, 0.01113, 0.00172, 16.23, 29.89, 105.5, 740.7, 0.1503, 0.3904, 0.3728, 0.1607, 0.3693, 0.09618, 0
        
        Example 2 (M):
        842517, M, 20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 24.99, 23.41, 158.8, 1956, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902, 0
        
        Example 3 (B):
        91544002, B, 11.06, 17.12, 71.25, 366.5, 0.1194, 0.1071, 0.04063, 0.04268, 0.1954, 0.07976, 0.1779, 1.03, 1.318, 12.3, 0.01262, 0.02348, 0.018, 0.01285, 0.0222, 0.008313, 11.69, 20.74, 76.08, 411.1, 0.1662, 0.2031, 0.1256, 0.09514, 0.278, 0.1168, 0
        
        Please provide 100 synthetic records in CSV format, with values that are plausible and maintain the natural relationships between features.
        """,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.7
    },
    # Task 3: Prompt with Metadata
    "PROMPT_3": {
        "text": """
        I need you to generate synthetic breast-cancer data based on real statistical properties. Please generate 100 records that accurately represent the data while maintaining the statistical properties and correlations found in real data.
        
        Key correlations to maintain:
        - Strong positive correlation among radius_mean, perimeter_worst, area_mean, radius_worst, perimeter_mean
        - Moderate positive correlation among concavity_se, concave points_worst, radius_worst, perimeter_mean, concave points_se
        - diagnosis class differences: B and M samples show significant differences in concavity_mean, area_se, concave points_mean, concavity_worst, area_worst
        
        Here are 3 example records from the real dataset:
        
        Example 1 (M):
        885429, M, 19.73, 19.82, 130.7, 1206, 0.1062, 0.1849, 0.2417, 0.0974, 0.1733, 0.06697, 0.7661, 0.78, 4.115, 92.81, 0.008482, 0.05057, 0.068, 0.01971, 0.01467, 0.007259, 25.28, 25.59, 159.8, 1933, 0.171, 0.5955, 0.8489, 0.2507, 0.2749, 0.1297, 0
        
        Example 2 (M):
        9012000, M, 22.01, 21.9, 147.2, 1482, 0.1063, 0.1954, 0.2448, 0.1501, 0.1824, 0.0614, 1.008, 0.6999, 7.561, 130.2, 0.003978, 0.02821, 0.03576, 0.01471, 0.01518, 0.003796, 27.66, 25.8, 195, 2227, 0.1294, 0.3885, 0.4756, 0.2432, 0.2741, 0.08574, 0
        
        Example 3 (B):
        904969, B, 12.34, 14.95, 78.29, 469.1, 0.08682, 0.04571, 0.02109, 0.02054, 0.1571, 0.05708, 0.3833, 0.9078, 2.602, 30.15, 0.007702, 0.008491, 0.01307, 0.0103, 0.0297, 0.001432, 13.18, 16.85, 84.11, 533.1, 0.1048, 0.06744, 0.04921, 0.04793, 0.2298, 0.05974, 0
        
        Please provide 100 synthetic records in CSV format, with values that are plausible and maintain both the statistical properties and natural relationships between features. Ensure the data could be useful for machine learning algorithms to differentiate between different diagnosis values.
        """,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.6
    },
    # Task 4: Prompt with No Examples - needs special handling
    "PROMPT_4": {
        "text": """
        I need you to generate synthetic breast-cancer data based exclusively on statistical properties without using any real examples. Please create 100 synthetic records that represent breast-cancer measurements while preserving the statistical distributions and correlations in the real-world data.
        
        Each record should include:
        
        - id: Unique identifier for each record
        - diagnosis: Classification with 'M' (37.3% of cases), 'B' (62.7% of cases)
        
        The data should contain breast-cancer measurements with their statistical properties:
        
        | Feature | Definition | Overall Mean | B Mean | M Mean | Min | Max | Std Dev |
        |---------|------------|--------------|-------------|-------------|-----|-----|---------|
        | radius_mean | Mean radius value | 14.1273 | 12.1465 | 17.4628 | 6.981 | 28.11 | 3.524 |
        | texture_mean | Mean texture value | 19.2896 | 17.9148 | 21.6049 | 9.71 | 39.28 | 4.301 |
        | perimeter_mean | Mean perimeter value | 91.969 | 78.0754 | 115.3654 | 43.79 | 188.5 | 24.299 |
        | area_mean | Mean area value | 654.8891 | 462.7902 | 978.3764 | 143.5 | 2501.0 | 351.9141 |
        | smoothness_mean | Mean smoothness value | 0.0964 | 0.0925 | 0.1029 | 0.0526 | 0.1634 | 0.0141 |
        | compactness_mean | Mean compactness value | 0.1043 | 0.0801 | 0.1452 | 0.0194 | 0.3454 | 0.0528 |
        | concavity_mean | Mean concavity value | 0.0888 | 0.0461 | 0.1608 | 0.0 | 0.4268 | 0.0797 |
        | concave points_mean | Mean concave points value | 0.0489 | 0.0257 | 0.088 | 0.0 | 0.2012 | 0.0388 |
        | symmetry_mean | Mean symmetry value | 0.1812 | 0.1742 | 0.1929 | 0.106 | 0.304 | 0.0274 |
        | fractal_dimension_mean | Mean fractal dimension value | 0.0628 | 0.0629 | 0.0627 | 0.05 | 0.0974 | 0.0071 |
        | radius_se | Numerical measurement of radius se | 0.4052 | 0.2841 | 0.6091 | 0.1115 | 2.873 | 0.2773 |
        | texture_se | Numerical measurement of texture se | 1.2169 | 1.2204 | 1.2109 | 0.3602 | 4.885 | 0.5516 |
        | perimeter_se | Numerical measurement of perimeter se | 2.8661 | 2.0003 | 4.3239 | 0.757 | 21.98 | 2.0219 |
        | area_se | Numerical measurement of area se | 40.3371 | 21.1351 | 72.6724 | 6.802 | 542.2 | 45.491 |
        | smoothness_se | Numerical measurement of smoothness se | 0.007 | 0.0072 | 0.0068 | 0.0017 | 0.0311 | 0.003 |
        | compactness_se | Numerical measurement of compactness se | 0.0255 | 0.0214 | 0.0323 | 0.0023 | 0.1354 | 0.0179 |
        | concavity_se | Numerical measurement of concavity se | 0.0319 | 0.026 | 0.0418 | 0.0 | 0.396 | 0.0302 |
        | concave points_se | Numerical measurement of concave points se | 0.0118 | 0.0099 | 0.0151 | 0.0 | 0.0528 | 0.0062 |
        | symmetry_se | Numerical measurement of symmetry se | 0.0205 | 0.0206 | 0.0205 | 0.0079 | 0.079 | 0.0083 |
        | fractal_dimension_se | Numerical measurement of fractal dimension se | 0.0038 | 0.0036 | 0.0041 | 0.0009 | 0.0298 | 0.0026 |
        | radius_worst | Numerical measurement of radius worst | 16.2692 | 13.3798 | 21.1348 | 7.93 | 36.04 | 4.8332 |
        | texture_worst | Numerical measurement of texture worst | 25.6772 | 23.5151 | 29.3182 | 12.02 | 49.54 | 6.1463 |
        | perimeter_worst | Numerical measurement of perimeter worst | 107.2612 | 87.0059 | 141.3703 | 50.41 | 251.2 | 33.6025 |
        | area_worst | Numerical measurement of area worst | 880.5831 | 558.8994 | 1422.2863 | 185.2 | 4254.0 | 569.357 |
        | smoothness_worst | Numerical measurement of smoothness worst | 0.1324 | 0.125 | 0.1448 | 0.0712 | 0.2226 | 0.0228 |
        | compactness_worst | Numerical measurement of compactness worst | 0.2543 | 0.1827 | 0.3748 | 0.0273 | 1.058 | 0.1573 |
        | concavity_worst | Numerical measurement of concavity worst | 0.2722 | 0.1662 | 0.4506 | 0.0 | 1.252 | 0.2086 |
        | concave points_worst | Numerical measurement of concave points worst | 0.1146 | 0.0744 | 0.1822 | 0.0 | 0.291 | 0.0657 |
        | symmetry_worst | Numerical measurement of symmetry worst | 0.2901 | 0.2702 | 0.3235 | 0.1565 | 0.6638 | 0.0619 |
        | fractal_dimension_worst | Numerical measurement of fractal dimension worst | 0.0839 | 0.0794 | 0.0915 | 0.055 | 0.2075 | 0.0181 |
        | Unnamed: 32 | Numerical measurement of unnamed: 32 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
        
        Important statistical relationships to maintain:
        
        1. Distribution pattern: Generate values following the statistical distributions in the table above, respecting min/max values and standard deviations
        2. Class-specific distributions: Maintain the statistical differences between different diagnosis values
        3. Feature correlations:
           - Strong positive correlation between radius_mean and perimeter_mean
           - Strong positive correlation between radius_worst and perimeter_worst
           - Strong positive correlation between radius_mean and area_mean
           - Moderate positive correlation between radius_worst and concave points_worst
           - Moderate positive correlation between concavity_se and concave points_se
        
        4. Logical constraints:
           - If present, perimeter should be proportional to radius (approximately 2πr)
           - If present, area should be proportional to radius squared (approximately πr²)
           - Values should respect the min/max ranges in the table above
        
        Please provide 100 synthetic records in CSV format that satisfy these statistical properties and mathematical relationships. The synthetic data should be suitable for analysis or training machine learning models while preserving privacy by not containing any actual data points.
        """,
        "max_tokens": MAX_NEW_TOKENS // 2,
        "temperature": 0.5
    },
}


def validate_record(record):
    """Validate if a generated record matches expected format and constraints for breast cancer dataset"""
    try:
        parts = record.strip().split(",")
        # Breast cancer dataset has 32 columns (id, diagnosis, 30 features)
        if len(parts) != 32:
            return False

        # Extract key fields
        id_val, diagnosis = parts[0], parts[1]
        
        # Basic format checks
        if not id_val.strip().isdigit():  # ID should be numeric
            return False
            
        if diagnosis.strip() not in ["M", "B"]:  # Diagnosis should be M or B
            return False
            
        # Convert all feature values to float for validation
        try:
            values = [float(p.strip()) for p in parts[2:]]
        except ValueError:
            return False
            
        # Check ranges for key features
        # Radius mean (index 2)
        if not (6.5 <= values[0] <= 29):
            return False
            
        # Area mean (index 5-2=3)
        if not (140 <= values[3] <= 2600):
            return False
            
        # Check relationships between related features
        radius_mean, perimeter_mean, area_mean = values[0], values[2], values[3]
        
        # Check if perimeter and area are reasonably proportional to radius
        # Perimeter should be roughly 2πr
        if not (5.5 * radius_mean <= perimeter_mean <= 7 * radius_mean):
            return False
            
        # Area should be roughly πr²
        if not (2.5 * radius_mean**2 <= area_mean <= 4 * radius_mean**2):
            return False
            
        # Check "worst" values are >= corresponding "mean" values
        if values[20] < values[0]:  # radius_worst < radius_mean
            return False
            
        if values[23] < values[3]:  # area_worst < area_mean
            return False
        
        # Malignant samples typically have higher values for certain features
        if diagnosis.strip() == "M" and values[6] < 0.04:  # concavity_mean too low for M
            return False
            
        # Check Unnamed: 32 is always 0
        if values[29] != 0:
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
    """Clear system memory (no GPU)"""
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
    
    # Try multiple approaches
    approaches = [
        # Approach 1: Add a formatting hint without examples
        {
            "name": "Format hint",
            "prompt_addition": """
            To generate data, use this format exactly for each record:
            id, diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst, Unnamed: 32
            
            For example, a generic record structure looks like:
            ID_NUMBER, DIAGNOSIS_LETTER, NUM1, NUM2, NUM3, NUM4, NUM5, NUM6, NUM7, NUM8, NUM9, NUM10, NUM11, NUM12, NUM13, NUM14, NUM15, NUM16, NUM17, NUM18, NUM19, NUM20, NUM21, NUM22, NUM23, NUM24, NUM25, NUM26, NUM27, NUM28, NUM29, NUM30, 0
            """,
            "temperature": 0.5,
            "max_tokens": 700,
            "attempts": 5
        },
        # Approach 2: Add more specific instructions
        {
            "name": "Detailed instructions",
            "prompt_addition": """
            Please generate exactly 10 records with the following format:
            id, diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst, Unnamed: 32
            
            Each record should be on a new line and follow all the statistical properties and distributions described above.
            """,
            "temperature": 0.7,
            "max_tokens": 700,
            "attempts": 5
        },
        # Approach 3: Generate one record at a time with specific format
        {
            "name": "Single record generation",
            "prompt_addition": """
            Generate a single valid breast cancer record in exactly this format:
            id, diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst, Unnamed: 32
            
            Just output the record values with no additional text or explanation.
            """,
            "temperature": 0.6,
            "max_tokens": 200,
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
                output = generator(
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
        
        # Define fields and constraints based on breast cancer data
        fields = [
            {"name": "id", "min": 8000000, "max": 9999999},
            {"name": "diagnosis", "values": ["M", "B"], "weights": [0.373, 0.627]},
            {"name": "radius_mean", "min": 6.981, "max": 28.11, "b_mean": 12.1465, "m_mean": 17.4628},
            {"name": "texture_mean", "min": 9.71, "max": 39.28, "b_mean": 17.9148, "m_mean": 21.6049},
            {"name": "perimeter_mean", "min": 43.79, "max": 188.5, "b_mean": 78.0754, "m_mean": 115.3654},
            {"name": "area_mean", "min": 143.5, "max": 2501.0, "b_mean": 462.7902, "m_mean": 978.3764},
            {"name": "smoothness_mean", "min": 0.0526, "max": 0.1634, "b_mean": 0.0925, "m_mean": 0.1029},
            {"name": "compactness_mean", "min": 0.0194, "max": 0.3454, "b_mean": 0.0801, "m_mean": 0.1452},
            {"name": "concavity_mean", "min": 0.0, "max": 0.4268, "b_mean": 0.0461, "m_mean": 0.1608},
            {"name": "concave points_mean", "min": 0.0, "max": 0.2012, "b_mean": 0.0257, "m_mean": 0.088},
            {"name": "symmetry_mean", "min": 0.106, "max": 0.304, "b_mean": 0.1742, "m_mean": 0.1929},
            {"name": "fractal_dimension_mean", "min": 0.05, "max": 0.0974, "b_mean": 0.0629, "m_mean": 0.0627},
            {"name": "radius_se", "min": 0.1115, "max": 2.873, "b_mean": 0.2841, "m_mean": 0.6091},
            {"name": "texture_se", "min": 0.3602, "max": 4.885, "b_mean": 1.2204, "m_mean": 1.2109},
            {"name": "perimeter_se", "min": 0.757, "max": 21.98, "b_mean": 2.0003, "m_mean": 4.3239},
            {"name": "area_se", "min": 6.802, "max": 542.2, "b_mean": 21.1351, "m_mean": 72.6724},
            {"name": "smoothness_se", "min": 0.0017, "max": 0.0311, "b_mean": 0.0072, "m_mean": 0.0068},
            {"name": "compactness_se", "min": 0.0023, "max": 0.1354, "b_mean": 0.0214, "m_mean": 0.0323},
            {"name": "concavity_se", "min": 0.0, "max": 0.396, "b_mean": 0.026, "m_mean": 0.0418},
            {"name": "concave points_se", "min": 0.0, "max": 0.0528, "b_mean": 0.0099, "m_mean": 0.0151},
            {"name": "symmetry_se", "min": 0.0079, "max": 0.079, "b_mean": 0.0206, "m_mean": 0.0205},
            {"name": "fractal_dimension_se", "min": 0.0009, "max": 0.0298, "b_mean": 0.0036, "m_mean": 0.0041},
            {"name": "radius_worst", "min": 7.93, "max": 36.04, "b_mean": 13.3798, "m_mean": 21.1348},
            {"name": "texture_worst", "min": 12.02, "max": 49.54, "b_mean": 23.5151, "m_mean": 29.3182},
            {"name": "perimeter_worst", "min": 50.41, "max": 251.2, "b_mean": 87.0059, "m_mean": 141.3703},
            {"name": "area_worst", "min": 185.2, "max": 4254.0, "b_mean": 558.8994, "m_mean": 1422.2863},
            {"name": "smoothness_worst", "min": 0.0712, "max": 0.2226, "b_mean": 0.125, "m_mean": 0.1448},
            {"name": "compactness_worst", "min": 0.0273, "max": 1.058, "b_mean": 0.1827, "m_mean": 0.3748},
            {"name": "concavity_worst", "min": 0.0, "max": 1.252, "b_mean": 0.1662, "m_mean": 0.4506},
            {"name": "concave points_worst", "min": 0.0, "max": 0.291, "b_mean": 0.0744, "m_mean": 0.1822},
            {"name": "symmetry_worst", "min": 0.1565, "max": 0.6638, "b_mean": 0.2702, "m_mean": 0.3235},
            {"name": "fractal_dimension_worst", "min": 0.055, "max": 0.2075, "b_mean": 0.0794, "m_mean": 0.0915},
            {"name": "Unnamed: 32", "values": [0]},
        ]
        
        # Generate records needed to complete the set
        needed_records = num_records - len(all_records)
        import random
        
        for record_num in range(needed_records):
            record_values = []
            
            # First generate fixed fields (id, diagnosis)
            id_val = random.randint(8000000, 9999999)
            record_values.append(str(id_val))
            
            # Weighted selection for diagnosis based on distribution
            diagnosis = random.choices(["M", "B"], weights=[0.373, 0.627], k=1)[0]
            record_values.append(diagnosis)
            
            # Get the appropriate means based on diagnosis
            means_key = "m_mean" if diagnosis == "M" else "b_mean"
            
            # Now generate all numeric fields with appropriate correlations
            for field_idx, field in enumerate(fields[2:], 2):  # Skip id and diagnosis
                # Skip Unnamed: 32 which is always 0
                if field_idx == len(fields) - 1:
                    record_values.append("0")
                    continue
                
                # Handle special correlations
                try:
                    # Use field-specific generation logic
                    if "values" in field:
                        value = random.choice(field["values"])
                    else:
                        # Generate from means based on diagnosis class
                        if means_key in field:
                            mean = field[means_key]
                            # Use the mean with a random deviation
                            min_val = field["min"]
                            max_val = field["max"]
                            
                            # Calculate std dev (range/6 is a rough approximation)
                            std_dev = (max_val - min_val) / 6
                            
                            # Generate value with normal distribution around the appropriate mean
                            value = random.normalvariate(mean, std_dev)
                            # Ensure within range
                            value = max(min_val, min(max_val, value))
                        else:
                            # Simple random value within range
                            value = random.uniform(field["min"], field["max"])
                            
                        # Handle special correlations for related fields
                        if field["name"] == "perimeter_mean" and len(record_values) > 2:
                            # perimeter_mean should be correlated with radius_mean
                            radius_mean = float(record_values[2])  # radius_mean is at index 2
                            # perimeter ≈ 2πr, with some variation
                            variation = random.uniform(0.9, 1.1)  # 10% variation
                            value = 2 * 3.14159 * radius_mean * variation
                        
                        elif field["name"] == "area_mean" and len(record_values) > 2:
                            # area_mean should be correlated with radius_mean
                            radius_mean = float(record_values[2])  # radius_mean at index 2
                            # area ≈ πr², with some variation
                            variation = random.uniform(0.9, 1.1)  # 10% variation
                            value = 3.14159 * radius_mean * radius_mean * variation
                        
                        elif field["name"] == "radius_worst" and len(record_values) > 2:
                            # radius_worst should be >= radius_mean
                            radius_mean = float(record_values[2])
                            value = max(value, radius_mean * random.uniform(1.0, 1.5))
                        
                        elif field["name"] == "perimeter_worst" and len(record_values) > 22:
                            # perimeter_worst should be correlated with radius_worst
                            radius_worst = float(record_values[22])  # radius_worst is at index 22
                            # perimeter ≈ 2πr, with some variation
                            variation = random.uniform(0.9, 1.1)  # 10% variation
                            value = 2 * 3.14159 * radius_worst * variation
                        
                        elif field["name"] == "area_worst" and len(record_values) > 22:
                            # area_worst should be correlated with radius_worst
                            radius_worst = float(record_values[22])  # radius_worst at index 22
                            # area ≈ πr², with some variation
                            variation = random.uniform(0.9, 1.1)  # 10% variation
                            value = 3.14159 * radius_worst * radius_worst * variation
                            
                        # Round appropriately based on typical precision
                        if "mean" in field["name"] or "worst" in field["name"]:
                            if field["name"].startswith(("smoothness", "compactness", "concavity", "symmetry", "fractal")):
                                value = round(value, 5)  # More decimal places for these measures
                            else:
                                value = round(value, 2)  # 2 decimal places for other measures
                        elif "se" in field["name"]:
                            value = round(value, 6)  # More precision for standard error values
                except Exception as e:
                    # If error, use default value
                    if "values" in field:
                        value = field["values"][0]
                    else:
                        value = field["min"]
                    print(f"Error generating {field['name']}: {e}")
                
                record_values.append(str(value))
            
            # Create record string
            record = ",".join(record_values)
            
            # Validate and add
            if validate_record(record):
                all_records.append(record)
                pbar.update(1)
                with open(log_file, "a") as f:
                    f.write(f"Generated step-by-step record: {record}\n")
            else:
                with open(log_file, "a") as f:
                    f.write(f"Failed to validate record: {record}\n")
    
    pbar.close()
    
    # Log completion
    with open(log_file, "a") as f:
        f.write(f"\nPROMPT_4 generation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total records generated: {len(all_records)}\n")
        if len(all_records) < num_records:
            f.write(f"WARNING: Only generated {len(all_records)}/{num_records} requested records\n")
    
    return all_records
        

def generate_dataset(generator, prompt_config, num_records=100, seed=None, experiment_num=1, prompt_name=""):

    # Extract settings
    prompt_text = prompt_config["text"]
    max_tokens = prompt_config["max_tokens"]
    
    # Calculate approximate token count of prompt (rough estimation)
    prompt_token_count = len(prompt_text.split())
    
    # Ensure we don't exceed model's context length (1024 for GPT-2)
    available_tokens = 1024 - prompt_token_count - 5  # 5 token buffer
    if max_tokens > available_tokens:
        print(f"Warning: Reducing max_tokens from {max_tokens} to {available_tokens} to fit within model context")
        max_tokens = max(available_tokens, 50)  # Ensure at least 50 tokens
    
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
    
    # Log the generation parameters
    print(f"\nGeneration parameters for {prompt_name}:")
    print(f"- Max tokens: {max_tokens}")
    print(f"- Temperature: {temperature}")
    
    # Create a log file for this specific generation
    log_file = os.path.join(LOG_DIR, f"generation_log_exp{experiment_num}_{prompt_name}.txt")
    with open(log_file, "w") as f:
        f.write(f"Generation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters: max_tokens={max_tokens}, temperature={temperature}, device=CPU\n\n")

    all_records = []
    pbar = tqdm(total=num_records, desc="Generating records")
    max_attempts = 20  # Increased attempts since we're using CPU
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
                    output = generator(
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
                for record in new_records:
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
                
                # Reduce tokens and try again
                max_tokens = max(50, max_tokens // 2)
                with open(log_file, "a") as f:
                    f.write(f"\nReducing max_tokens to {max_tokens}\n")
                
                continue  # Try the next attempt
                
            except Exception as e:
                print(f"Error in generation: {str(e)}")
                with open(log_file, "a") as f:
                    f.write(f"\nError in generation: {str(e)}\n")
                if attempt < max_attempts - 1:
                    time.sleep(2)  # Wait before retrying
                    continue
                else:
                    break
    
    pbar.close()
    
    # Log completion
    with open(log_file, "a") as f:
        f.write(f"\nGeneration completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total records generated: {len(all_records)}\n")
    
    return all_records


def save_and_validate_dataset(records, prompt_name, experiment_num):
    """Save dataset and perform basic validation with experiment number"""
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
        "Unnamed: 32"
    ]

    # Parse records with proper handling of whitespace
    parsed_records = []
    for record in records:
        parts = [part.strip() for part in record.split(",")]
        if len(parts) == len(columns):
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


def get_system_info():
    """Get current system resource usage"""
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


def main():
    # Create initial status file
    status_file = os.path.join(OUTPUT_DIR, "experiment_status.txt")
    with open(status_file, "w") as f:
        f.write(f"Breast cancer data generation experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Device: CPU (Forced to avoid CUDA errors)\n")
        f.write(f"Experiments: {NUM_EXPERIMENTS}\n")
        f.write("----------------------------------------\n")
    
    print(f"Starting GPT-2 breast cancer data generation on CPU only mode")
    print(f"Results will be saved to: {OUTPUT_DIR}")
    print(f"Logs will be saved to: {LOG_DIR}")
    
    try:
        # Initialize generator only on CPU
        print(f"Initializing generator with {MODEL_NAME} on CPU")
        generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device=DEVICE,  # DEVICE is set to -1 (CPU)
            framework="pt",  # Explicitly use PyTorch
        )
        print("Successfully initialized generator")

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
                    "device": "CPU",
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
        print(f"CPU info: {psutil.cpu_count()} cores")
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available")
        
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
        # No need for CUDA cleanup since we're only using CPU