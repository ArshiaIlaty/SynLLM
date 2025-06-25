import argparse
import gc
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
from llama_cpp import Llama

# Configuration
NUM_RECORDS = 100
SYSTEM_PROMPT = """You are a synthetic medical data generator. Generate realistic patient records for diabetes research."""

EXAMPLE_RECORDS = [
    "Female,45.2,1,0,never,28.5,6.2,140,0",
    "Male,62.7,1,1,former,32.1,7.1,185,1",
    "Female,38.9,0,0,current,24.3,5.8,130,0",
    "Male,70.0,1,1,current,30.2,8.0,210,1",
    "Female,29.4,0,0,never,23.0,5.1,110,0",
]


def build_prompt(system_prompt, user_prompt):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"


def extract_records(text):
    print("=== DEBUG: GENERATED TEXT ===")
    print(text[:500])
    print("...\n=== END DEBUG ===")

    VALID_SMOKING = {"never", "former", "current", "not current", "no info", "unknown"}
    yes_no_map = {"yes": "1", "no": "0"}

    records = []
    skipped = 0
    bad_lines = []

    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+\.\s*", "", line)
        if not line or line.count(",") != 8:
            bad_lines.append(line)
            skipped += 1
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 9:
            bad_lines.append(line)
            skipped += 1
            continue

        gender, age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes = parts

        if gender.lower() not in ["male", "female"]:
            bad_lines.append(line)
            skipped += 1
            continue

        smoking = smoking.lower()
        if smoking not in VALID_SMOKING:
            bad_lines.append(line)
            smoking = "unknown"

        hyp = yes_no_map.get(hyp.lower(), hyp)
        heart = yes_no_map.get(heart.lower(), heart)
        diabetes = yes_no_map.get(diabetes.lower(), diabetes)

        records.append(
            [gender.title(), age, hyp, heart, smoking, bmi, hba1c, glucose, diabetes]
        )

    if bad_lines:
        Path("reports").mkdir(parents=True, exist_ok=True)
        with open("reports/rejected_records.txt", "a") as bad:
            bad.write("\n".join(bad_lines) + "\n")

    print(f"[INFO] Extracted {len(records)} valid records, skipped {skipped}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the GGUF model file"
    )
    parser.add_argument("--prompt_file", type=str, required=True)
    args = parser.parse_args()

    output_dir = (
        Path("bash")
        / Path(args.model_path).stem
        / f"records_{Path(args.prompt_file).stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "diabetes_records.csv"

    print(f"Loading GGUF model from {args.model_path}...")
    model = Llama(model_path=args.model_path, n_ctx=4096, n_threads=os.cpu_count())

    with open(args.prompt_file, "r") as f:
        user_prompt = f.read()

    all_records = []
    batch_size = 20
    start_time = time.time()

    while len(all_records) < NUM_RECORDS:
        needed = min(batch_size, NUM_RECORDS - len(all_records))
        print(
            f"Generating batch of {needed} records... ({len(all_records)}/{NUM_RECORDS} total)"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # No torch needed for llama-cpp

        gc.collect()

        prompt = build_prompt(SYSTEM_PROMPT, user_prompt)
        output = model(prompt, temperature=0.7, top_p=0.9, max_tokens=2000)
        generated_text = output["choices"][0]["text"]

        with open(output_dir / "raw_output.txt", "a") as f:
            f.write(generated_text + "\n\n")

        new_records = extract_records(generated_text)
        all_records.extend(new_records)
        time.sleep(2)

    end_time = time.time()
    print(f"[INFO] Finished generation in {end_time - start_time:.2f} seconds")

    if all_records:
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
        df = pd.DataFrame(all_records[:NUM_RECORDS], columns=columns)
        for col in columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_csv(output_path, index=False)
        print(f"\nSuccessfully generated {len(all_records)} records!")
        print(f"Data saved to {output_path}")
    else:
        print("Failed to generate any valid records.")


if __name__ == "__main__":
    main()
