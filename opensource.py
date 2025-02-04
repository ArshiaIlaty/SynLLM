from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="gpt2-medium")
set_seed(42)

prompt = """Generate synthetic diabetes data in CSV format:
gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes
Female,54.0,0,0,No Info,27.32,6.6,80,0
Male,50.0,1,0,current,36.8,7.2,260,1
"""

output = generator(prompt, max_length=1500, num_return_sequences=1, temperature=0.7)

print(output[0]["generated_text"])
