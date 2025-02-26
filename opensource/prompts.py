# Define your prompts
PROMPTS = {
    # Step 1: Basic Prompt with Examples
    "PROMPT_1": """
Generate 100 realistic synthetic patient records for diabetes prediction following this format:
- gender (String: 'Male' or 'Female')
- age (Float: 0.0-100.0)
- hypertension (Integer: 0 or 1)
- heart_disease (Integer: 0 or 1)
- smoking_history (String: categories will be provided in examples)
- bmi (Float: typical range 15.0-60.0)
- HbA1c_level (Float: typical range 4.0-9.0)
- blood_glucose_level (Integer: typical range 70-300)
- diabetes (Integer: 0 or 1)

Examples will be provided in the following format:
1. Female,45.2,1,0,never,28.5,6.2,140,0
2. Male,62.7,1,1,former,32.1,7.1,185,1
3. Female,38.9,0,0,current,24.3,5.8,130,0
4. Female,22.0,0,0,never,25.77,4.0,145,0
5. Male,58.0,0,0,former,36.53,5.8,160,0
6. Male,11.0,0,0,No Info,27.59,6.6,100,0

Generate 100 comma-separated records, one per line, maintaining realistic correlations between features.
""",
    # Step 2: Prompt with Definitions
    "PROMPT_2": """
Generate 100 realistic synthetic patient records for diabetes prediction. Here are the features with definitions:

Features:
1. gender: Patient's gender (Male/Female)
2. age: Patient's age in years (Float: 0.0-100.0)
3. hypertension: Whether patient has hypertension (0: No, 1: Yes)
4. heart_disease: Whether patient has heart disease (0: No, 1: Yes)
5. smoking_history: Patient's smoking history (never/former/current/not current)
6. bmi: Body Mass Index, measure of body fat based on weight and height (Float: 15.0-60.0)
7. HbA1c_level: Hemoglobin A1c level, measure of average blood sugar over past 3 months (Float: 4.0-9.0)
8. blood_glucose_level: Current blood glucose level in mg/dL (Integer: 70-300)
9. diabetes: Whether patient has diabetes (0: No, 1: Yes)

Examples from real data:
1. Female,45.2,1,0,never,28.5,6.2,140,0
2. Male,62.7,1,1,former,32.1,7.1,185,1
3. Female,38.9,0,0,current,24.3,5.8,130,0
4. Female,22.0,0,0,never,25.77,4.0,145,0
5. Male,58.0,0,0,former,36.53,5.8,160,0
6. Male,11.0,0,0,No Info,27.59,6.6,100,0

Generate 100 comma-separated records, one per line, maintaining realistic correlations between features.
""",
    # Step 3: Prompt with Definitions and Metadata
    "PROMPT_3": """
Generate 100 realistic synthetic patient records for diabetes prediction. Here are the features with definitions and statistical metadata:

Features and Statistics:
1. gender
   - Distribution: Male: 48%, Female: 52%

2. age
   - Mean: 41.8, Std: 15.2
   - Range: 18.0-80.0
   - Distribution: Slightly right-skewed

3. hypertension
   - Distribution: No (0): 85%, Yes (1): 15%
   - Correlates with age and BMI

4. heart_disease
   - Distribution: No (0): 92%, Yes (1): 8%
   - Correlates with age and hypertension

5. smoking_history
   - Categories: never (60%), former (22%), current (15%), not current (3%)

6. bmi
   - Mean: 27.3, Std: 6.4
   - Range: 15.0-60.0
   - Distribution: Right-skewed

7. HbA1c_level
   - Mean: 5.7, Std: 0.9
   - Range: 4.0-9.0
   - Distribution: Right-skewed
   - Strong correlation with diabetes status

8. blood_glucose_level
   - Mean: 138.0, Std: 40.5
   - Range: 70-300
   - Distribution: Right-skewed
   - Strong correlation with HbA1c_level

9. diabetes
   - Distribution: No (0): 88%, Yes (1): 12%
   - Correlates strongly with HbA1c_level and blood_glucose_level

Examples from real data:
1. Female,45.2,1,0,never,28.5,6.2,140,0
2. Male,62.7,1,1,former,32.1,7.1,185,1
3. Female,38.9,0,0,current,24.3,5.8,130,0
4. Female,22.0,0,0,never,25.77,4.0,145,0
5. Male,58.0,0,0,former,36.53,5.8,160,0
6. Male,11.0,0,0,No Info,27.59,6.6,100,0

Generate 100 comma-separated records, one per line, maintaining realistic correlations between features.
""",
    # Step 4: Prompt with only Definitions and Metadata (No Examples)
    "PROMPT_4": """
Generate 100 realistic synthetic patient records for diabetes prediction. Here are the features with definitions and statistical metadata:

Features and Statistics:
[Generate 100 realistic synthetic patient records for diabetes prediction. Here are the features with definitions and statistical metadata:

Features and Statistics:
1. gender
   - Distribution: Male: 48%, Female: 52%

2. age
   - Mean: 41.8, Std: 15.2
   - Range: 18.0-80.0
   - Distribution: Slightly right-skewed

3. hypertension
   - Distribution: No (0): 85%, Yes (1): 15%
   - Correlates with age and BMI

4. heart_disease
   - Distribution: No (0): 92%, Yes (1): 8%
   - Correlates with age and hypertension

5. smoking_history
   - Categories: never (60%), former (22%), current (15%), not current (3%)

6. bmi
   - Mean: 27.3, Std: 6.4
   - Range: 15.0-60.0
   - Distribution: Right-skewed

7. HbA1c_level
   - Mean: 5.7, Std: 0.9
   - Range: 4.0-9.0
   - Distribution: Right-skewed
   - Strong correlation with diabetes status

8. blood_glucose_level
   - Mean: 138.0, Std: 40.5
   - Range: 70-300
   - Distribution: Right-skewed
   - Strong correlation with HbA1c_level

9. diabetes
   - Distribution: No (0): 88%, Yes (1): 12%
   - Correlates strongly with HbA1c_level and blood_glucose_level

Important correlations to maintain:
1. Higher age correlates with increased hypertension and heart disease risk
2. Higher BMI correlates with increased diabetes risk
3. HbA1c_level strongly correlates with diabetes status
4. blood_glucose_level correlates with HbA1c_level and diabetes status
5. Hypertension and heart_disease are more common in older ages

Generate 100 comma-separated records, one per line, maintaining these relationships and statistical properties.
""",
}

# # Define prompts dictionary - Adjusted for LLaMA's style
# PROMPTS = {
#     "PROMPT_1": """<s>[INST] You are a medical data generator. Generate synthetic diabetes patient data following this format exactly:
# gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

# Requirements:
# - gender: only "Male" or "Female"
# - age: between 18 and 80
# - hypertension: only 0 or 1
# - heart_disease: only 0 or 1
# - smoking_history: only "never", "former", "current", or "not current"
# - bmi: between 15 and 60
# - HbA1c_level: between 4 and 9
# - blood_glucose_level: between 70 and 300
# - diabetes: only 0 or 1

# Examples:
# Female,45.2,0,0,never,28.5,5.7,155,0
# Male,62.7,1,1,former,32.1,6.8,185,1

# Generate 5 new records: [/INST]</s>""",

#     "PROMPT_2": """<s>[INST] You are a medical data generator. Generate synthetic diabetes patient data following these medical relationships:

# Format:
# gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

# Medical rules to follow:
# 1. HbA1c > 6.5 indicates high diabetes risk
# 2. Blood glucose > 180 indicates high diabetes risk
# 3. BMI > 30 increases diabetes risk
# 4. Age > 45 increases hypertension risk
# 5. Hypertension increases heart disease risk

# Value constraints:
# [Same as PROMPT_1]

# Examples:
# Female,45.2,0,0,never,28.5,5.7,155,0
# Male,62.7,1,1,former,32.1,6.8,185,1

# Generate 5 medically consistent records: [/INST]</s>""",

#     "PROMPT_3": """<s>[INST] You are a medical data generator. Generate synthetic diabetes patient data matching these statistical properties:

# Format:
# gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

# Statistical distribution:
# - gender: Male (48%), Female (52%)
# - age: Mean=41.8, range 18-80
# - hypertension: No (85%), Yes (15%)
# - heart_disease: No (92%), Yes (8%)
# - smoking_history: never (60%), former (22%), current (15%), not current (3%)
# - bmi: Mean=27.3, range 15-60
# - HbA1c_level: Mean=5.7, range 4-9
# - blood_glucose_level: Mean=138, range 70-300
# - diabetes: No (88%), Yes (12%)

# Generate 5 records matching these distributions: [/INST]</s>""",

#     "PROMPT_4": """<s>[INST] You are a medical data generator. Generate synthetic diabetes patient data following this schema and relationships:

# Format:
# gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

# Clinical relationships to maintain:
# 1. For diabetes positive cases (diabetes=1):
#    - HbA1c_level should typically be > 6.5
#    - blood_glucose_level should typically be > 180
#    - BMI often > 30
# 2. For hypertension cases (hypertension=1):
#    - Age typically > 50
#    - Often accompanied by heart_disease=1
# 3. For heart_disease cases (heart_disease=1):
#    - Usually occurs with hypertension=1
#    - More common in smokers (smoking_history="current" or "former")
#    - Age typically > 60

# Value constraints:
# [Same as previous prompts]

# Generate 5 clinically consistent records: [/INST]</s>"""}
