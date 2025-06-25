# import pandas as pd

# # Load the dataset (adjust filename as needed)
# df = pd.read_csv("processed.cleveland.data", header=None)

# # Save as CSV
# df.to_csv("processed_cleveland.csv", index=False)

# print("Conversion complete! Saved as processed_cleveland.csv")

# import os
# import pandas as pd
# import zipfile

# # Define the paths
# zip_path = "/mnt/data/thyroid+disease.zip"
# extract_folder = "/mnt/data/thyroid_disease"

# Step 1: Extract the ZIP file
# with zipfile.ZipFile(zip_path, "r") as zip_ref:
#     zip_ref.extractall(extract_folder)

# Step 2: Convert all .data and .txt files to CSV
# for file in os.listdir(extract_folder):
#     if file.endswith((".data", ".txt")):
#         file_path = os.path.join(extract_folder, file)

#         # Read the dataset (without headers, assuming space or comma separation)
#         try:
#             df = pd.read_csv(file_path, header=None, delim_whitespace=True)
#         except:
#             df = pd.read_csv(file_path, header=None)

#         # Save as CSV
#         csv_file = file_path.rsplit(".", 1)[0] + ".csv"
#         df.to_csv(csv_file, index=False)
#         print(f"Converted {file} to {csv_file}")

# print("Conversion completed for all files!")

import pandas as pd

# Define column names
column_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Define categorical mappings
category_mappings = {
    "sex": {0: "Female", 1: "Male"},
    "fbs": {0: "False", 1: "True"},
    "exang": {0: "No", 1: "Yes"},
    "restecg": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy",
    },
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"},
    "num": {
        0: "No Disease",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Very Severe",
    },  # Adjusting target labels
}

# Load the dataset (assuming it's a CSV with no header)
file_path = "processed_cleveland.csv"  # Change this path accordingly
df = pd.read_csv(file_path, header=None, names=column_names)

# Convert categorical values
for col, mapping in category_mappings.items():
    df[col] = df[col].map(mapping)

# Save cleaned dataset
cleaned_file_path = "cleveland_cleaned.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved at: {cleaned_file_path}")
