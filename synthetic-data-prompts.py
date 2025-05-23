# Step 1: Basic Prompt with Examples
PROMPT_1 = """
Generate 1000 realistic synthetic patient records for diabetes prediction following this format:
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

Generate 1000 comma-separated records, one per line, maintaining realistic correlations between features.
"""

# Step 2: Prompt with Definitions
PROMPT_2 = """
Generate 1000 realistic synthetic patient records for diabetes prediction. Here are the features with definitions:

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

Generate 1000 comma-separated records, one per line, maintaining realistic correlations between features.
"""

# Step 3: Prompt with Definitions and Metadata
PROMPT_3 = """
Generate 1000 realistic synthetic patient records for diabetes prediction. Here are the features with definitions and statistical metadata:

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

Generate 1000 comma-separated records, one per line, maintaining realistic correlations between features.
"""

# Step 4: Prompt with only Definitions and Metadata (No Examples)
PROMPT_4 = """
Generate 1000 realistic synthetic patient records for diabetes prediction. Here are the features with definitions and statistical metadata:

Features and Statistics:
[Same as PROMPT_3 but without examples]

Important correlations to maintain:
1. Higher age correlates with increased hypertension and heart disease risk
2. Higher BMI correlates with increased diabetes risk
3. HbA1c_level strongly correlates with diabetes status
4. blood_glucose_level correlates with HbA1c_level and diabetes status
5. Hypertension and heart_disease are more common in older ages

Generate 1000 comma-separated records, one per line, maintaining these relationships and statistical properties.
"""
