system_prompt = """You are a medical data synthesis expert. Generate 1000 synthetic diabetes patient records in CSV format with these columns:
gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes

Rules:
1. Maintain clinical plausibility:
   - HbA1c > 6.5% or blood_glucose > 200 mg/dL should strongly correlate with diabetes=1
   - BMI < 18.5 should be rare (<5% of cases)
   - Hypertension/heart disease should increase with age

2. Formatting:
   - First line: header
   - Subsequent lines: comma-separated values
   - No markdown formatting
   - Float values for numerical columns

3. Diversity:
   - Include all smoking_history categories
   - 45-55% female patients
   - Age range: 0.1-80 years

Examples:
Female,54.0,0,0,No Info,27.32,6.6,80,0
Male,50.0,1,0,current,36.8,7.2,260,1
"""
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Generate 1000 synthetic records following the rules.",
        },
    ],
    temperature=0.7,  # Balances creativity vs consistency
    max_tokens=4000,
)

import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=4000,
    temperature=0.7,
    system=system_prompt,
    messages=[{"role": "user", "content": "Generate 1000 records"}],
)


def parse_response(response):
    raw_text = response.choices[0].message.content
    # Remove headers/markdown if present
    clean_text = raw_text.replace("```csv", "").replace("```", "").strip()
    # Split into lines
    lines = [line for line in clean_text.split("\n") if line.count(",") == 8]

    with open("synthetic_data.csv", "w") as f:
        f.write(
            "gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes\n"
        )
        f.write("\n".join(lines))
