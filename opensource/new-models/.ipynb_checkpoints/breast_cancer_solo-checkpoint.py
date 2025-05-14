import gc
import os
import re
import time
import argparse

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "opensource/new-models/breast_cancer_synthetic_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RECORDS = 100

CHAT_STYLE_MODELS = {
    "mistral": "chatml",
    "llama": "instruct",
    "gemma": "openai",
    "yi": "im_start",
    "mixtral": "instruct"
}

SYSTEM_PROMPT = """You are a synthetic medical data generator. Generate realistic patient records for breast cancer research."""


def generate_chat_prompt(batch_size=100):
    return f"""I need you to generate synthetic breast-cancer data that closely resembles real-world data. The dataset should contain {batch_size} samples with the following columns:

id, diagnosis (M/B), radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean,  fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst, Unnamed: 32

Here are 3 example records from a real dataset to guide your generation:

Example 1:
9110732, M, 17.75, 28.03, 117.3, 981.6, 0.09997, 0.1314, 0.1698, 0.08293, 0.1713, 0.05916, 0.3897, 1.077, 2.873, 43.95, 0.004714, 0.02015, 0.03697, 0.0111, 0.01237, 0.002556, 21.53, 38.54, 145.4, 1437, 0.1401, 0.3762, 0.6399, 0.197, 0.2972, 0.09075, 0

Example 2:
8911670, M, 18.81, 19.98, 120.9, 1102, 0.08923, 0.05884, 0.0802, 0.05843, 0.155, 0.04996, 0.3283, 0.828, 2.363, 36.74, 0.007571, 0.01114, 0.02623, 0.01463, 0.0193, 0.001676, 19.96, 24.3, 129, 1236, 0.1243, 0.116, 0.221, 0.1294, 0.2567, 0.05737, 0

Example 3:
904689, B, 12.96, 18.29, 84.18, 525.2, 0.07351, 0.07899, 0.04057, 0.01883, 0.1874, 0.05899, 0.2357, 1.299, 2.397, 20.21, 0.003629, 0.03713, 0.03452, 0.01065, 0.02632, 0.003705, 14.13, 24.61, 96.31, 621.9, 0.09329, 0.2318, 0.1604, 0.06608, 0.3207, 0.07247, 0

Please generate {batch_size} records in a CSV format that follows these patterns and maintains realistic relationships between the features. The data should be plausible and preserve the correlations between features that would be found in real breast-cancer data."""


def extract_records(text):
    print("=== DEBUG: GENERATED TEXT ===")
    print(text[:500])
    print("...\n=== END DEBUG ===")

    records = []
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"^\d+\.\s*", "", line)
        if not line or line.count(",") < 30:
            continue
        records.append(line)
    return records


def detect_chat_style(model_name):
    for key, style in CHAT_STYLE_MODELS.items():
        if key in model_name.lower():
            return style
    return "plain"


def build_prompt(model_name, system_prompt, user_prompt, tokenizer):
    style = detect_chat_style(model_name)
    if style == "instruct":
        prompt = f"<s>[INST] {system_prompt}\n{user_prompt} [/INST]"
        return tokenizer(prompt, return_tensors="pt")
    elif style == "openai":
        if "gemma" in model_name.lower():
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(prompt_str, return_tensors="pt")
    elif style == "im_start":
        prompt_str = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer(prompt_str, return_tensors="pt")
    else:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        return tokenizer(prompt, return_tensors="pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    model_name = args.model_name

    print(f"Loading model {model_name} on {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            max_memory={0: "8GiB"},
        )
    except Exception as e:
        print(f"Could not load in 4-bit: {e}, falling back to full precision.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_records = []
    batch_size = 25

    while len(all_records) < NUM_RECORDS:
        records_needed = min(batch_size, NUM_RECORDS - len(all_records))
        print(f"Generating batch of {records_needed}... ({len(all_records)}/{NUM_RECORDS})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        chat_input = build_prompt(
            model_name,
            SYSTEM_PROMPT,
            generate_chat_prompt(records_needed),
            tokenizer
        ).to(DEVICE)

        try:
            outputs = model.generate(
                input_ids=chat_input["input_ids"],
                attention_mask=chat_input.get("attention_mask"),
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.strip()
            new_records = extract_records(response)
            print(f"Found {len(new_records)} valid records.")
            all_records.extend(new_records)
            time.sleep(2)
        except Exception as e:
            print(f"Generation error: {str(e)}")
            time.sleep(5)

    all_records = all_records[:NUM_RECORDS]
    if all_records:
        output_path = os.path.join(OUTPUT_DIR, "breast_cancer_records.csv")
        df = pd.DataFrame([r.split(",") for r in all_records])
        df.to_csv(output_path, index=False, header=False)
        print(f"\n✅ Saved {len(all_records)} records to {output_path}")
    else:
        print("❌ No valid records generated.")


if __name__ == "__main__":
    main()
