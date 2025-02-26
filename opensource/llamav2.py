# from prompts import PROMPTS
import gc
import json
import os
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

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
OUTPUT_DIR = "model_comparison_llama"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 200
SAMPLES_PER_PROMPT = 100
BATCH_SIZE = 5

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


class ExperimentMonitor:
    """Monitor and log experiment metrics"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.start_time = time.time()
        self.metrics = {
            "timestamps": [],
            "gpu_memory": [],
            "cpu_memory": [],
            "generation_speeds": [],
        }

        # Create experiment directory with timestamp
        self.experiment_dir = os.path.join(
            output_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

    def log_memory(self):
        """Log current memory usage"""
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            gpu_memory = {
                "total": gpu.memoryTotal,
                "used": gpu.memoryUsed,
                "free": gpu.memoryFree,
            }
        else:
            gpu_memory = None

        cpu_memory = {
            "total": psutil.virtual_memory().total / (1024**3),
            "used": psutil.virtual_memory().used / (1024**3),
            "free": psutil.virtual_memory().free / (1024**3),
        }

        self.metrics["timestamps"].append(time.time() - self.start_time)
        self.metrics["gpu_memory"].append(gpu_memory)
        self.metrics["cpu_memory"].append(cpu_memory)

    def log_generation_speed(self, num_records, time_taken):
        """Log generation speed"""
        self.metrics["generation_speeds"].append(
            {
                "records": num_records,
                "time": time_taken,
                "records_per_second": num_records / time_taken,
            }
        )

    def save_metrics(self, experiment_info):
        """Save all metrics to file"""
        metrics_file = os.path.join(self.experiment_dir, "experiment_metrics.json")

        # Combine metrics with experiment info
        full_metrics = {
            "experiment_info": experiment_info,
            "total_duration": time.time() - self.start_time,
            "metrics": self.metrics,
        }

        with open(metrics_file, "w") as f:
            json.dump(full_metrics, f, indent=2)


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
        if not (70 <= int(float(glucose)) <= 300):
            return False
        if int(diabetes) not in [0, 1]:
            return False

        return True
    except:
        return False


def extract_records(text):
    """Extract valid records from generated text"""
    lines = text.split("\n")
    potential_records = [line.strip() for line in lines if "," in line]
    valid_records = [record for record in potential_records if validate_record(record)]
    return valid_records


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def save_and_analyze_dataset(records, prompt_name, experiment_dir):
    """Save and analyze the generated dataset"""
    output_path = os.path.join(experiment_dir, f"llama_{prompt_name}_data.csv")

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
    numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    binary_cols = ["hypertension", "heart_disease", "diabetes"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print analysis
    print(f"\nDataset analysis for {prompt_name}:")
    print(f"Total records: {len(df)}")
    print("\nNumerical columns summary:")
    print(df[numeric_cols].describe())
    print("\nValue counts:")
    for col in ["gender", "smoking_history"] + binary_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True))


# Define prompts
PROMPTS = {
    "PROMPT_1": """<s>[INST] You are a medical data generator. Generate synthetic diabetes patient data following this format exactly:
gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

Requirements:
- gender: only "Male" or "Female"
- age: between 18 and 80
- hypertension: only 0 or 1
- heart_disease: only 0 or 1
- smoking_history: only "never", "former", "current", or "not current"
- bmi: between 15 and 60
- HbA1c_level: between 4 and 9
- blood_glucose_level: between 70 and 300
- diabetes: only 0 or 1

Examples:
Female,45.2,0,0,never,28.5,5.7,155,0
Male,62.7,1,1,former,32.1,6.8,185,1

Generate 5 new records: [/INST]</s>""",
    "PROMPT_2": """<s>[INST] You are a medical data generator. Generate synthetic diabetes patient data following these medical rules:
[Same format as PROMPT_1 with medical rules]
[/INST]</s>""",
    "PROMPT_3": """<s>[INST] You are a medical data generator. Generate synthetic diabetes patient data with statistical properties:
[Same format as PROMPT_1 with statistical properties]
[/INST]</s>""",
}


def generate_batch(model, tokenizer, prompt, num_records, monitor):
    """Generate a batch of records with monitoring"""
    records = []
    batch_start_time = time.time()
    pbar = tqdm(total=num_records, desc="Generating records")

    while len(records) < num_records:
        try:
            # Log memory before generation
            monitor.log_memory()

            # Generate
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Process outputs
            generated_text = tokenizer.decode(
                outputs[0].cpu(), skip_special_tokens=True
            )
            del outputs
            clear_gpu_memory()

            new_records = extract_records(generated_text)

            # Add valid records
            for record in new_records:
                if len(records) < num_records:
                    records.append(record)
                    pbar.update(1)
                else:
                    break

        except RuntimeError as e:
            if "out of memory" in str(e):
                clear_gpu_memory()
                print("OOM error, clearing memory and continuing...")
                continue
            else:
                raise e

    generation_time = time.time() - batch_start_time
    monitor.log_generation_speed(len(records), generation_time)
    pbar.close()

    return records


def main():
    # Initialize experiment monitor
    monitor = ExperimentMonitor(OUTPUT_DIR)

    print(f"Loading LLaMA model on {DEVICE}")

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=quantization_config, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Generate for each prompt
        for prompt_name, prompt_text in PROMPTS.items():
            print(f"\n=== Generating for: {prompt_name} ===")
            records = generate_batch(
                model, tokenizer, prompt_text, SAMPLES_PER_PROMPT, monitor
            )
            save_and_analyze_dataset(records, prompt_name, monitor.experiment_dir)
            clear_gpu_memory()

        # Save all metrics
        monitor.save_metrics(
            {
                "model_name": MODEL_NAME,
                "device": DEVICE,
                "samples_per_prompt": SAMPLES_PER_PROMPT,
                "quantization": "4-bit",
                "batch_size": BATCH_SIZE,
                "torch_version": torch.__version__,
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else None
                ),
                "gpu_info": (
                    GPUtil.getGPUs()[0].name if torch.cuda.is_available() else None
                ),
            }
        )

    finally:
        clear_gpu_memory()
        print("\nExperiment completed")


if __name__ == "__main__":
    main()
