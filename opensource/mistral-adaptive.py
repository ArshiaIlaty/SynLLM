import gc
import os
import time
from datetime import datetime

import GPUtil
import pandas as pd
import psutil
import torch
from prompts import PROMPTS
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class MemoryManager:
    def __init__(self, min_gpu_memory_free=4000):  # Increased from 2000 to 4000 MB
        self.min_gpu_memory_free = min_gpu_memory_free
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.using_gpu = self.device == "cuda"
        self.fallback_to_cpu = False

    def get_memory_status(self):
        """Get current memory status"""
        if self.using_gpu:
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

        return {"gpu": gpu_memory, "cpu": cpu_memory}

    def should_use_gpu(self):
        """Check if GPU should be used based on memory availability"""
        if not self.using_gpu:
            return False

        gpu = GPUtil.getGPUs()[0]
        free_memory = gpu.memoryFree
        print(f"\nAvailable GPU memory: {free_memory}MB")
        return free_memory >= self.min_gpu_memory_free

    def clear_memory(self):
        """Clear memory cache more aggressively"""
        if self.using_gpu:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def log_memory_status(self, operation):
        """Log memory status for debugging"""
        status = self.get_memory_status()
        print(f"\nMemory status after {operation}:")
        if status["gpu"]:
            print(
                f"GPU - Free: {status['gpu']['free']:.2f}MB, Used: {status['gpu']['used']:.2f}MB"
            )
        print(
            f"CPU - Free: {status['cpu']['free']:.2f}GB, Used: {status['cpu']['used']:.2f}GB"
        )


class AdaptiveGenerator:
    def __init__(self, model_name, output_dir, samples_per_prompt=100):
        self.model_name = model_name
        self.output_dir = output_dir
        self.samples_per_prompt = samples_per_prompt
        self.memory_manager = MemoryManager()
        self.max_new_tokens = 1000

        # Create output directory with timestamp
        self.experiment_dir = os.path.join(
            output_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

    @staticmethod
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

    @staticmethod
    def extract_records(text):
        """Extract valid records from generated text"""
        lines = text.split("\n")
        potential_records = [line.strip() for line in lines if "," in line]
        valid_records = [
            record
            for record in potential_records
            if AdaptiveGenerator.validate_record(record)
        ]
        return valid_records

    def load_model(self):
        """Load model with adaptive device selection"""
        use_gpu = self.memory_manager.should_use_gpu()
        device = "cuda" if use_gpu else "cpu"
        print(f"\nLoading model on {device}")

        try:
            if use_gpu:
                # Updated quantization config
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                model = model.to(device)  # Ensure model is on correct device
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="cpu", torch_dtype=torch.float32
                )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            return model, tokenizer, device

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.memory_manager.log_memory_status("model loading failure")
            raise

    def generate_text(self, model, tokenizer, prompt, device):
        """Generate text with improved error handling and memory management"""
        max_attempts = 5  # Increased from 3 to 5
        for attempt in range(max_attempts):
            try:
                self.memory_manager.clear_memory()

                # Split long prompts into smaller chunks if needed
                if len(prompt) > 1000:  # If prompt is very long
                    print("\nLong prompt detected, processing in chunks...")

                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                timeout = 120  # Increased timeout from 60 to 120 seconds
                start_time = time.time()

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2,  # Added to prevent repetitions
                    )

                    if time.time() - start_time > timeout:
                        raise TimeoutError("Generation took too long")

                    outputs = outputs.cpu()
                    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    del outputs, inputs
                    self.memory_manager.clear_memory()
                    return text

            except (RuntimeError, TimeoutError) as e:
                print(f"\nError in attempt {attempt + 1}/{max_attempts}: {str(e)}")
                self.memory_manager.clear_memory()
                if device == "cuda":
                    print("Switching to CPU for this generation")
                    device = "cpu"
                    model = model.to("cpu")
                time.sleep(5)  # Added cool-down period
                continue

    def generate_dataset(self, prompt_name, prompt_text):
        """Generate complete dataset with adaptive handling and strict sample control"""
        try:
            model, tokenizer, device = self.load_model()
            records = []
            pbar = tqdm(total=self.samples_per_prompt, desc=f"Generating {prompt_name}")

            max_consecutive_failures = 5
            consecutive_failures = 0
            start_time = time.time()
            timeout = 3600  # 1 hour timeout for entire dataset generation

            while len(records) < self.samples_per_prompt:
                try:
                    if time.time() - start_time > timeout:
                        print(f"\nTimeout reached for {prompt_name}")
                        break

                    generated_text = self.generate_text(
                        model, tokenizer, prompt_text, device
                    )
                    new_records = self.extract_records(generated_text)

                    if not new_records:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            print(f"\nToo many consecutive failures for {prompt_name}")
                            break
                    else:
                        consecutive_failures = 0

                    # Only add records up to the desired sample size
                    remaining = self.samples_per_prompt - len(records)
                    records.extend(new_records[:remaining])
                    pbar.update(min(len(new_records), remaining))

                    # Break if we have enough samples
                    if len(records) >= self.samples_per_prompt:
                        break

                except Exception as e:
                    print(f"\nError in generation: {str(e)}")
                    self.memory_manager.log_memory_status("generation error")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"\nToo many consecutive failures for {prompt_name}")
                        break
                    continue

            pbar.close()

            # Return whatever records we managed to generate
            return records[: self.samples_per_prompt]

        except Exception as e:
            print(f"\nFatal error in dataset generation: {str(e)}")
            self.memory_manager.log_memory_status("fatal error")
            raise

    @staticmethod
    def save_and_analyze_dataset(records, prompt_name, experiment_dir):
        """Save and analyze the generated dataset"""
        output_path = os.path.join(experiment_dir, f"mistral_{prompt_name}_data.csv")

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
        print(f"\nSaved dataset to: {output_path}")

        # Print analysis
        print(f"\nDataset analysis for {prompt_name}:")
        print(f"Total records: {len(df)}")
        print("\nNumerical columns summary:")
        print(df[numeric_cols].describe())
        print("\nValue counts:")
        for col in ["gender", "smoking_history"] + binary_cols:
            print(f"\n{col}:")
            print(df[col].value_counts(normalize=True))

        # Check medical consistency
        medical_checks = {
            "diabetes_hba1c": len(df[(df["diabetes"] == 1) & (df["HbA1c_level"] > 6.5)])
            / len(df[df["diabetes"] == 1]),
            "diabetes_glucose": len(
                df[(df["diabetes"] == 1) & (df["blood_glucose_level"] > 180)]
            )
            / len(df[df["diabetes"] == 1]),
            "hypertension_age": len(df[(df["hypertension"] == 1) & (df["age"] > 50)])
            / len(df[df["hypertension"] == 1]),
        }
        print("\nMedical consistency checks (proportions):")
        print(medical_checks)


def main():
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    OUTPUT_DIR = "mistral"
    SAMPLES_PER_PROMPT = 100

    generator = AdaptiveGenerator(MODEL_NAME, OUTPUT_DIR, SAMPLES_PER_PROMPT)

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n=== Processing {prompt_name} ===")
        try:
            records = generator.generate_dataset(prompt_name, prompt_text)
            AdaptiveGenerator.save_and_analyze_dataset(
                records, prompt_name, generator.experiment_dir
            )
            generator.memory_manager.log_memory_status(f"completed {prompt_name}")
        except Exception as e:
            print(f"Error processing {prompt_name}: {str(e)}")
            continue


if __name__ == "__main__":
    main()

# from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
# import torch
# import os
# import pandas as pd
# from tqdm import tqdm
# import time
# import gc
# from prompts import PROMPTS

# # Memory optimization settings
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# # Configuration
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
# OUTPUT_DIR = 'model_comparison'
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MAX_NEW_TOKENS = 200
# SAMPLES_PER_PROMPT = 100  # Reduced samples
# BATCH_SIZE = 1  # Minimal batch size

# def clear_gpu_memory():
#     """Clear GPU memory cache"""
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         gc.collect()

# def load_model_with_minimal_memory():
#     """Load model with minimal memory usage"""
#     try:
#         # Configure 8-bit quantization
#         quantization_config = BitsAndBytesConfig(
#             load_in_8bit=True,
#             llm_int8_threshold=6.0,
#             llm_int8_has_fp16_weight=True,
#             bnb_8bit_compute_dtype=torch.float16
#         )

#         # Clear any existing cache
#         clear_gpu_memory()

#         # Load model with minimal memory footprint
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_NAME,
#             quantization_config=quantization_config,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True
#         )

#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         return model, tokenizer

#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         print("\nAvailable GPU memory:")
#         print(torch.cuda.memory_summary())
#         raise

# def generate_with_memory_constraints(model, tokenizer, prompt, max_attempts=3):
#     """Generate text with memory constraints"""
#     for attempt in range(max_attempts):
#         try:
#             # Clear memory before generation
#             clear_gpu_memory()

#             # Generate with minimal memory usage
#             with torch.no_grad():
#                 inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=MAX_NEW_TOKENS,
#                     num_return_sequences=1,
#                     temperature=0.7,
#                     do_sample=True,
#                     pad_token_id=tokenizer.eos_token_id
#                 )

#                 # Immediately move to CPU and clear GPU
#                 generated_text = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
#                 del outputs
#                 clear_gpu_memory()

#                 return generated_text

#         except RuntimeError as e:
#             if "out of memory" in str(e) and attempt < max_attempts - 1:
#                 print(f"OOM error, attempt {attempt + 1}/{max_attempts}")
#                 clear_gpu_memory()
#                 continue
#             else:
#                 raise e

# def generate_batch(model, tokenizer, prompt, num_records):
#     """Generate records with memory-efficient batching"""
#     records = []
#     pbar = tqdm(total=num_records, desc="Generating records")

#     while len(records) < num_records:
#         try:
#             # Generate one record at a time
#             generated_text = generate_with_memory_constraints(model, tokenizer, prompt)
#             new_records = extract_records(generated_text)

#             # Add valid records
#             for record in new_records:
#                 if len(records) < num_records:
#                     records.append(record)
#                     pbar.update(1)
#                 else:
#                     break

#         except Exception as e:
#             print(f"Error in generation: {str(e)}")
#             continue

#     pbar.close()
#     return records

# def main():
#     print(f"Loading Mistral model on {DEVICE} with minimal memory usage")

#     try:
#         # Load model with memory optimization
#         model, tokenizer = load_model_with_minimal_memory()

#         # Generate for each prompt
#         for prompt_name, prompt_text in PROMPTS.items():
#             print(f"\n=== Generating for: {prompt_name} ===")
#             try:
#                 records = generate_batch(model, tokenizer, prompt_text, SAMPLES_PER_PROMPT)
#                 save_and_analyze_dataset(records, prompt_name)
#                 clear_gpu_memory()  # Clear after each prompt
#             except Exception as e:
#                 print(f"Error processing {prompt_name}: {str(e)}")
#                 continue

#     finally:
#         clear_gpu_memory()
#         print("\nGeneration completed")

# if __name__ == "__main__":
#     main()
