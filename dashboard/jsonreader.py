# import pandas as pd; df = pd.read_json("/home/jovyan/SynLLM/gpt2-100-prompt-in-5experiments/metrics/experiments_summary.json"); print(df)

import os

import pandas as pd

# Directory where JSON files are stored
json_dir = "/home/jovyan/SynLLM/gpt2-100-prompt-in-5experiments/metrics/"

# List all JSON files in the directory
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

# Ensure there are JSON files available
if not json_files:
    print("No JSON files found in the directory.")
    exit()

# Display options for the user
print("Available JSON files:")
for idx, file in enumerate(json_files, start=1):
    print(f"{idx}. {file}")

# Get user input
while True:
    try:
        choice = int(input("Select a file number to open: "))
        if 1 <= choice <= len(json_files):
            selected_file = json_files[choice - 1]
            break
        else:
            print("Invalid choice. Please enter a number from the list.")
    except ValueError:
        print("Please enter a valid number.")

# Full path of the selected JSON file
file_path = os.path.join(json_dir, selected_file)

# Read and display JSON file
try:
    df = pd.read_json(file_path)
    print(f"\nContents of {selected_file}:")
    print(df)
except ValueError as e:
    print(f"Error reading {selected_file}: {e}")
