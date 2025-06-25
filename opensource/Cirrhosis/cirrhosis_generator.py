import argparse
import gc
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils import is_bitsandbytes_available

# Restrict to first MIG instance
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Added from diabetes code

# Configuration
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RECORDS = 100

# Chat style mappings - Updated based on diabetes code
CHAT_STYLE_MODELS = {
    "gpt2": "plain",
    "mistral": "chatml",
    "mixtral": "chatml",
    "llama2": "llama",
    "llama3": "llama",
    "gemma": "openai",
    "yi": "im_start",
    "zephyr": "zephyr",
    "qwen": "qwen",
    "qwen2": "chatml",  # Updated to match diabetes code
    "internlm": "internlm",
    "beluga": "beluga",
    "openchat": "openchat",
    "nous": "nous",
    "mpt": "mpt",
    "t5": "t5",  # Added T5 format
}

SYSTEM_PROMPT = """You are a synthetic medical data generator. Generate realistic patient records for liver cirrhosis research."""

EXAMPLE_RECORDS = [
    "1191,C,D-penicillamine,58,F,N,Y,Y,N,1.1,261,3.48,54,1636,113.52,119,221,10.6,3",
    "400,D,D-penicillamine,70,F,N,N,Y,S,12.58,200.5,2.74,140,1058,138.7,246,214,12.1,4",
    "4556,C,Placebo,54,F,N,Y,Y,N,0.8,156,3.05,54,5882,71.0,183,416,10.9,3",
]


def detect_chat_style(model_name):
    # Special case for T5 models
    if "t5" in model_name.lower():
        return "t5"

    for key, style in CHAT_STYLE_MODELS.items():
        if key in model_name.lower():
            return style
    return "plain"


def build_prompt(model_name, system_prompt, user_prompt, tokenizer):
    style = detect_chat_style(model_name)

    if style == "t5":
        # T5 uses a simple text input format
        return tokenizer(f"{system_prompt}\n\n{user_prompt}", return_tensors="pt")
    elif style == "llama":
        return tokenizer(
            f"<s>[INST] {system_prompt}\n{user_prompt} [/INST]", return_tensors="pt"
        )
    elif style == "chatml":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True)
    elif style == "openai":
        messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")
    elif style in ["im_start", "internlm", "nous"]:
        prompt_str = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer(prompt_str, return_tensors="pt")
    elif style in ["zephyr", "qwen", "qwen2"]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")
    elif style == "phi":
        return tokenizer(
            f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n",
            return_tensors="pt",
        )
    elif style == "openchat":
        return tokenizer(
            f"GPT4 System: {system_prompt}\nHuman: {user_prompt}\nAssistant:",
            return_tensors="pt",
        )
    elif style == "mpt":
        return tokenizer(
            f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:",
            return_tensors="pt",
        )
    else:
        return tokenizer(f"{system_prompt}\n\n{user_prompt}", return_tensors="pt")


def extract_records(text):
    print("=== DEBUG: GENERATED TEXT ===")
    print(text[:500])
    print("...\n=== END DEBUG ===")

    # Define valid values for categorical fields with more flexibility
    VALID_STATUS = {"c", "cl", "d"}  # Main values
    # Allow flexibility with more drugs
    VALID_DRUG = {
        "d-penicillamine",
        "placebo",
        "rifaximin",
        "lactulose",
        "spirinolactone",
        "spironolactone",
        "furosemide",
        "frusemide",
        "sodium oxychloride",
        "terlipressin",
        "propranolol",
        "prednisone",
        "prednisolone",
        "ribavirin",
        "sulfasalazine",
        "sucralfate",
        "sodium polystyrene sulfonate",
        "folic acid",
        "pantoprazole",
        "lisinopril",
    }
    VALID_SEX = {"m", "f"}
    VALID_YN = {"y", "n"}
    VALID_EDEMA = {"y", "n", "s"}
    VALID_STAGE = {"1", "2", "3", "4", "3.0", "3.5", "4.0"}

    records = []
    skipped = 0
    bad_lines = []

    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+[\.,]\s*", "", line)  # Remove numbering if present

        # Skip empty lines or lines with explanatory text
        if not line or "note:" in line.lower() or "these records" in line.lower():
            continue

        # Skip field headers
        if (
            "id" in line.lower()
            and "n_days" in line.lower()
            and "status" in line.lower()
        ):
            continue

        parts = [p.strip() for p in line.split(",")]

        # Skip lines with wrong number of fields
        if len(parts) < 19:  # Allow both with and without ID
            if line:  # Only log non-empty lines
                bad_lines.append(f"Wrong field count: {line}")
                skipped += 1
            continue

        # Handle both with and without ID
        if len(parts) == 20:  # Has ID field
            # Skip the ID field
            (
                _,
                n_days,
                status,
                drug,
                age,
                sex,
                ascites,
                hepatomegaly,
                spiders,
                edema,
                bilirubin,
                cholesterol,
                albumin,
                copper,
                alk_phos,
                sgot,
                tryglicerides,
                platelets,
                prothrombin,
                stage,
            ) = parts
        else:  # No ID field (19 fields)
            (
                n_days,
                status,
                drug,
                age,
                sex,
                ascites,
                hepatomegaly,
                spiders,
                edema,
                bilirubin,
                cholesterol,
                albumin,
                copper,
                alk_phos,
                sgot,
                tryglicerides,
                platelets,
                prothrombin,
                stage,
            ) = parts

        # Validate categorical fields with more flexibility
        status = status.lower()
        if status not in VALID_STATUS:
            print(f"[WARNING] Invalid status: {status}, allowing anyway")
            # We'll allow it with a warning instead of skipping

        drug = drug.lower()
        if drug not in VALID_DRUG:
            print(f"[WARNING] New drug: {drug}, allowing it")
            # Allow new drugs with a warning

        sex = sex.lower()
        if sex not in VALID_SEX:
            bad_lines.append(f"Invalid sex: {line}")
            skipped += 1
            continue

        ascites = ascites.lower()
        if ascites not in VALID_YN:
            if ascites == "a":  # Handle "A" for Accumulation
                ascites = "y"
            else:
                bad_lines.append(f"Invalid ascites: {line}")
                skipped += 1
                continue

        hepatomegaly = hepatomegaly.lower()
        if hepatomegaly not in VALID_YN:
            if hepatomegaly == "a":  # Handle "A" for Accumulation
                hepatomegaly = "y"
            else:
                bad_lines.append(f"Invalid hepatomegaly: {line}")
                skipped += 1
                continue

        spiders = spiders.lower()
        if spiders not in VALID_YN:
            if spiders == "a":  # Handle "A" for Present
                spiders = "y"
            else:
                bad_lines.append(f"Invalid spiders: {line}")
                skipped += 1
                continue

        edema = edema.lower()
        if edema not in VALID_EDEMA:
            if edema == "a":  # Handle "A" for Present
                edema = "y"
            else:
                bad_lines.append(f"Invalid edema: {line}")
                skipped += 1
                continue

        # Validate stage with more flexibility
        if stage not in VALID_STAGE:
            if stage.lower() in ["a", "b", "c", "d"]:
                # Convert letter stages to numbers
                stage_map = {"a": "1", "b": "2", "c": "3", "d": "4"}
                stage = stage_map[stage.lower()]
            else:
                try:
                    stage_float = float(stage)
                    if 1 <= stage_float <= 4:
                        stage = str(stage_float)
                    else:
                        bad_lines.append(f"Invalid stage range: {line}")
                        skipped += 1
                        continue
                except ValueError:
                    bad_lines.append(f"Invalid stage: {line}")
                    skipped += 1
                    continue

        # More flexible numeric field validation
        try:
            # Convert to appropriate types to validate
            n_days = float(n_days)

            # Fix the age issue - ensure it's within realistic bounds
            try:
                age_val = float(age)
                # Cap age to realistic values (18-100)
                if age_val > 100:
                    print(f"[WARNING] Unrealistic age value: {age_val}, capping to 100")
                    age = "100"
                elif age_val < 18:
                    print(f"[WARNING] Unrealistic age value: {age_val}, setting to 18")
                    age = "18"
            except ValueError:
                bad_lines.append(f"Invalid age: {line}")
                skipped += 1
                continue

            bilirubin = float(bilirubin)
            cholesterol = float(cholesterol)
            albumin = float(albumin)
            copper = float(copper)
            alk_phos = float(alk_phos)
            sgot = float(sgot)
            tryglicerides = float(tryglicerides)
            platelets = float(platelets)
            prothrombin = float(prothrombin)
        except ValueError as e:
            bad_lines.append(f"Non-numeric fields: {line} - {e}")
            skipped += 1
            continue

        # Format fields with proper capitalization
        status = status.upper()
        # Capitalize drug name properly
        drug_words = drug.split()
        drug = " ".join(word.capitalize() for word in drug_words)
        sex = sex.upper()
        ascites = ascites.upper()
        hepatomegaly = hepatomegaly.upper()
        spiders = spiders.upper()
        edema = edema.upper()

        # Create the record with all processed values
        record = [
            str(n_days),
            status,
            drug,
            str(age),
            sex,
            ascites,
            hepatomegaly,
            spiders,
            edema,
            str(bilirubin),
            str(cholesterol),
            str(albumin),
            str(copper),
            str(alk_phos),
            str(sgot),
            str(tryglicerides),
            str(platelets),
            str(prothrombin),
            stage,
        ]
        records.append(record)

    if bad_lines:
        Path("reports").mkdir(parents=True, exist_ok=True)
        with open("reports/rejected_records.txt", "a") as bad:
            bad.write("\n".join(bad_lines) + "\n")

    print(f"[INFO] Extracted {len(records)} valid records, skipped {skipped}")

    # Debug output
    if records:
        print(f"[DEBUG] First few status values: {[r[1] for r in records[:5]]}")
        print(f"[DEBUG] First few drug values: {[r[2] for r in records[:5]]}")
        print(
            f"[DEBUG] First few age values: {[r[3] for r in records[:5]]}"
        )  # Debug age values
        print(f"[DEBUG] First few edema values: {[r[8] for r in records[:5]]}")
        print(f"[DEBUG] First few stage values: {[r[18] for r in records[:5]]}")

    return records


def log_system_metrics(model_name, prompt_name, start_time, end_time):
    gpu_info_before = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        .decode()
        .strip()
    )

    gpu_info_after = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        .decode()
        .strip()
    )

    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    report = {
        "model": model_name,
        "prompt": prompt_name,
        "duration_sec": end_time - start_time,
        "cpu_percent": cpu,
        "ram_percent": ram,
        "gpu_mem_start_MB": gpu_info_before,
        "gpu_mem_end_MB": gpu_info_after,
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/usage.jsonl", "a") as f:
        f.write(json.dumps(report) + "\n")


def log_generation_error(model_name, prompt_name, error):
    error_log = {"model": model_name, "prompt": prompt_name, "error": str(error)}
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/errors.jsonl", "a") as errfile:
        errfile.write(json.dumps(error_log) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of records to generate in each batch",
    )
    args = parser.parse_args()

    model_name = args.model_name
    safe_model_name = model_name.replace("/", "_")
    prompt_name = Path(args.prompt_file).stem
    output_dir = Path("bash") / safe_model_name / f"records_{prompt_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cirrhosis_records.csv"

    # Special handling for GPT-2
    batch_size = args.batch_size
    if "gpt2" in model_name.lower():
        batch_size = 5  # Reduced batch size for GPT-2
        print(f"[INFO] Using reduced batch size {batch_size} for GPT-2 model")

    print(f"Loading model {model_name} on {DEVICE}")

    # Clean GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)  # Explicitly set device
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU memory: {total_mem:.2f} GB")

    # Determine attention implementation - same as diabetes code
    if "phi" in model_name.lower() or "gpt2" in model_name.lower():
        attn_backend = "eager"
    elif "t5" in model_name.lower() or "bart" in model_name.lower():
        attn_backend = None
    else:
        attn_backend = "flash_attention_2"

    # Start building kwargs - same as diabetes code
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # MPT does not support this kwarg
    if attn_backend and "mpt" not in model_name.lower():
        model_kwargs["attn_implementation"] = attn_backend

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        if "llama" in model_name.lower() and hasattr(config, "rope_scaling"):
            current_scaling = config.rope_scaling or {}
            config.rope_scaling = {
                "name": "dynamic",
                "factor": current_scaling.get("factor", 8.0),
            }

        # Determine the correct model class based on the model type
        if "t5" in model_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                config=config,
                quantization_config=quantization_config,
                **model_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                quantization_config=quantization_config,
                **model_kwargs,
            )
    except Exception as e:
        print(
            f"⚠️ Could not load model with quantization ({e}), falling back to basic loading."
        )
        # Determine the correct model class based on the model type
        if "t5" in model_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    try:
        from transformers.models.gpt_neox.tokenization_gpt_neox import GPTNeoXTokenizer
    except ImportError:
        print(
            "Could not import GPTNeoXTokenizer explicitly, falling back to AutoTokenizer."
        )

    # Tokenizer setup - same as diabetes code
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.prompt_file, "r") as f:
        user_prompt = f.read()

    all_records = []
    start_time = time.time()

    print("Starting generation...")

    while len(all_records) < NUM_RECORDS:
        needed = min(batch_size, NUM_RECORDS - len(all_records))
        print(
            f"Generating batch of {needed} records... ({len(all_records)}/{NUM_RECORDS} total)"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        chat_input = build_prompt(model_name, SYSTEM_PROMPT, user_prompt, tokenizer).to(
            DEVICE
        )

        try:
            # Calculate max tokens based on model context size
            max_context = getattr(tokenizer, "model_max_length", 2048)
            max_tokens = min(1000, max_context - chat_input["input_ids"].shape[-1] - 10)

            print(f"Generating with max_new_tokens={max_tokens}")

            # Different generation approach for T5 vs causal models
            if "t5" in model_name.lower():
                outputs = model.generate(
                    input_ids=chat_input["input_ids"],
                    attention_mask=chat_input.get("attention_mask"),
                    max_length=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            else:
                outputs = model.generate(
                    input_ids=chat_input["input_ids"],
                    attention_mask=chat_input.get("attention_mask"),
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if (
                "n_days" in generated_text.lower()
                and "status" in generated_text.lower()
            ):
                print("[DEBUG] Model likely returned schema, not data.")

            with open(output_dir / "raw_output.txt", "a") as f:
                f.write(generated_text + "\n\n")

            new_records = extract_records(generated_text)
            print(f"Found {len(new_records)} valid records in this batch")
            all_records.extend(new_records)
            time.sleep(2)
        except RuntimeError as e:
            print(f"[ERROR] CUDA error during generation: {e}")
            log_generation_error(model_name, prompt_name, e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(5)  # Wait a bit before retrying
            continue
        except Exception as e:
            log_generation_error(model_name, prompt_name, e)
            print(f"[ERROR] Unexpected error: {e}")
            time.sleep(2)
            continue

    end_time = time.time()
    log_system_metrics(model_name, prompt_name, start_time, end_time)

    all_records = all_records[:NUM_RECORDS]

    if all_records:
        # Define columns based on cirrhosis dataset (without ID - we'll add it later)
        columns = [
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
            "Stage",
        ]

        df = pd.DataFrame(all_records, columns=columns)

        # Convert numeric columns
        numeric_columns = [
            "N_Days",
            "Age",
            "Bilirubin",
            "Cholesterol",
            "Albumin",
            "Copper",
            "Alk_Phos",
            "SGOT",
            "Tryglicerides",
            "Platelets",
            "Prothrombin",
            "Stage",
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Apply additional validation/normalization
            if col == "Age":
                # Ensure age is within realistic bounds
                df[col] = df[col].clip(18, 100)
            elif col == "Stage":
                # Ensure stage is between 1 and 4
                df[col] = df[col].clip(1, 4)

        # Debug output before saving
        print("\nDataFrame head before saving:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())

        # Generate a random ID for each record
        df["ID"] = range(1, len(df) + 1)

        # Reorder columns to match the original dataset
        final_columns = ["ID"] + columns
        df = df[final_columns]

        df.to_csv(output_path, index=False)

        print(f"\nSuccessfully generated {len(all_records)} records!")
        print(f"Data saved to {output_path}")
        print("\nData statistics:")
        print(f"Status distribution: {df['Status'].value_counts(normalize=True)}")
        print(f"Drug distribution: {df['Drug'].value_counts(normalize=True)}")
        print(f"Sex distribution: {df['Sex'].value_counts(normalize=True)}")
        print(f"Stage distribution: {df['Stage'].value_counts(normalize=True)}")
        print(
            f"Age range: {df['Age'].min()} to {df['Age'].max()}, mean: {df['Age'].mean():.2f}"
        )
        print(
            f"Bilirubin range: {df['Bilirubin'].min()} to {df['Bilirubin'].max()}, mean: {df['Bilirubin'].mean():.2f}"
        )
    else:
        print("Failed to generate any valid records.")


if __name__ == "__main__":
    main()
