# from llama_cpp import Llama

# llm = Llama(
#     model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # Download from Hugging Face
#     n_ctx=2048,
#     n_gpu_layers=20
# )

# response = llm.create_chat_completion(
#     messages=[{"role": "user", "content": "Generate CSV..."}],
#     temperature=0.7
# )

import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # or "meta-llama/Llama-2-13b-hf"
OUTPUT_DIR = "model_comparison"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 200
set_seed(42)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define prompts dictionary - Adjusted for LLaMA's style
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
    "PROMPT_2": """<s>[INST] You are a medical data generator. Generate synthetic diabetes patient data following these medical relationships:

Format:
gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

Medical rules to follow:
1. HbA1c > 6.5 indicates high diabetes risk
2. Blood glucose > 180 indicates high diabetes risk
3. BMI > 30 increases diabetes risk
4. Age > 45 increases hypertension risk
5. Hypertension increases heart disease risk

Value constraints:
[Same as PROMPT_1]

Examples:
Female,45.2,0,0,never,28.5,5.7,155,0
Male,62.7,1,1,former,32.1,6.8,185,1

Generate 5 medically consistent records: [/INST]</s>""",
    "PROMPT_3": """<s>[INST] You are a medical data generator. Generate synthetic diabetes patient data matching these statistical properties:

Format:
gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

Statistical distribution:
- gender: Male (48%), Female (52%)
- age: Mean=41.8, range 18-80
- hypertension: No (85%), Yes (15%)
- heart_disease: No (92%), Yes (8%)
- smoking_history: never (60%), former (22%), current (15%), not current (3%)
- bmi: Mean=27.3, range 15-60
- HbA1c_level: Mean=5.7, range 4-9
- blood_glucose_level: Mean=138, range 70-300
- diabetes: No (88%), Yes (12%)

Generate 5 records matching these distributions: [/INST]</s>""",
    "PROMPT_4": """<s>[INST] You are a medical data generator. Generate synthetic diabetes patient data following this schema and relationships:

Format:
gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

Clinical relationships to maintain:
1. For diabetes positive cases (diabetes=1):
   - HbA1c_level should typically be > 6.5
   - blood_glucose_level should typically be > 180
   - BMI often > 30
2. For hypertension cases (hypertension=1):
   - Age typically > 50
   - Often accompanied by heart_disease=1
3. For heart_disease cases (heart_disease=1):
   - Usually occurs with hypertension=1
   - More common in smokers (smoking_history="current" or "former")
   - Age typically > 60

Value constraints:
[Same as previous prompts]

Generate 5 clinically consistent records: [/INST]</s>""",
}


def generate_batch(model, tokenizer, prompt, num_records=1000):
    """Generate a batch of records"""
    records = []
    pbar = tqdm(total=num_records, desc="Generating records")

    while len(records) < num_records:
        try:
            # Prepare input
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode and extract records
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_records = extract_records(generated_text)

            # Add valid records
            for record in new_records:
                if len(records) < num_records:
                    records.append(record)
                    pbar.update(1)
                else:
                    break

        except Exception as e:
            print(f"Error in generation: {str(e)}")
            continue

        # Update prompt with successful records
        if len(new_records) > 0:
            prompt = prompt.replace(
                "[/INST]</s>", "\n" + "\n".join(new_records[-2:]) + "\n[/INST]</s>"
            )

    pbar.close()
    return records


def main():
    print(f"Loading LLaMA model on {DEVICE}")

    # Load model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        token=True,  # You'll need to add your HF token here
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Generate for each prompt
    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n=== Generating for: {prompt_name} ===")
        records = generate_batch(model, tokenizer, prompt_text)
        save_and_analyze_dataset(records, prompt_name)
        print(f"Completed generation for {prompt_name}")


if __name__ == "__main__":
    main()
