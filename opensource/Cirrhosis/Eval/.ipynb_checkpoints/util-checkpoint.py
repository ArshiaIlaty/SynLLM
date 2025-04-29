import os
import pandas as pd
import argparse
import traceback
from pathlib import Path

# Import the CirrhosisDataEvaluator class
from fixed_evaluator import CirrhosisDataEvaluator

# Define the correct paths
REAL_DATA_PATH = "/home/jovyan/SynLLM/datasets/cirrhosis.csv"
BASE_SYNTHETIC_PATH = "/home/jovyan/SynLLM/opensource/Cirrhosis/bash"
PROMPTS = ["prompt1", "prompt2", "prompt3", "prompt4"]
RECORDS_DIRNAME = "records_prompt"
CSV_FILENAME = "cirrhosis_records.csv"

def load_real_data(data_path):
    """Load the real dataset."""
    print(f"Loading real dataset from {data_path}...")
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find real data file at '{data_path}'.")
    
    # Load the data
    try:
        real_data = pd.read_csv(data_path)
        print(f"Successfully loaded real data with {len(real_data)} rows and {len(real_data.columns)} columns")
        return real_data
    except Exception as e:
        print(f"Error loading real data: {str(e)}")
        raise

def load_synthetic_data(file_path):
    """Load a synthetic dataset from CSV."""
    try:
        # Try to load the data with different potential formats
        try:
            data = pd.read_csv(file_path)
        except:
            # Try with different encoding if default fails
            try:
                data = pd.read_csv(file_path, encoding='latin1')
            except:
                # Try with error handling for quoted data
                data = pd.read_csv(file_path, on_bad_lines='skip', escapechar='\\')
            
        print(f"Successfully loaded synthetic data from {file_path} with {len(data)} rows")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def evaluate_model_prompt(real_data, synthetic_data, model_name, prompt_name):
    """Evaluate a model-prompt combination."""
    print(f">>> Evaluating {model_name} × {prompt_name}...")
    
    try:
        # Create evaluator
        evaluator = CirrhosisDataEvaluator(real_data, synthetic_data)
        
        # Run evaluation
        results = evaluator.evaluate_flat(model_name=model_name, prompt_name=prompt_name)
        
        # Return results
        return results
    except Exception as e:
        error_message = str(e)
        print(f"❌ Evaluation failed for {model_name} × {prompt_name}: {error_message}")
        return {"model": model_name, "prompt": prompt_name, "error": error_message}

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate synthetic cirrhosis datasets")
    parser.add_argument("--data-dir", type=str, default=BASE_SYNTHETIC_PATH, 
                        help="Directory containing synthetic datasets")
    parser.add_argument("--real-data", type=str, default=REAL_DATA_PATH, 
                        help="Path to real dataset")
    parser.add_argument("--output", type=str, default="cirrhosis_evaluation_summary.csv", 
                        help="Output CSV file")
    parser.add_argument("--records-dir", type=str, default=RECORDS_DIRNAME,
                        help="Name of the records directory prefix")
    parser.add_argument("--csv-name", type=str, default=CSV_FILENAME,
                        help="Name of the CSV file in each prompt directory")
    args = parser.parse_args()
    
    # Load real data
    try:
        real_data = load_real_data(args.real_data)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please specify the correct path to the real data file using --real-data")
        return
    
    # Find all synthetic datasets
    all_results = []
    data_dir = Path(args.data_dir)
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        print("Please specify the correct path to the data directory using --data-dir")
        return
    
    print(f"Looking for synthetic datasets in {data_dir}...")
    
    # Go through all model directories
    model_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(model_dirs)} model directories")
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"Processing model: {model_name}")
        
        # Check each prompt
        for prompt_name in PROMPTS:
            # Construct the path to the records directory for this prompt
            records_dir = model_dir / f"{args.records_dir}{prompt_name.replace('prompt', '')}"
            file_path = records_dir / args.csv_name
            
            # Check if directory exists
            if not records_dir.exists():
                print(f"⚠️ Skipping: {model_name} × {prompt_name} (directory not found at {records_dir})")
                continue
                
            # Check if file exists
            if not file_path.exists():
                print(f"⚠️ Skipping: {model_name} × {prompt_name} (file not found at {file_path})")
                continue
            
            # Load synthetic data
            synthetic_data = load_synthetic_data(file_path)
            
            if synthetic_data is None:
                print(f"⚠️ Skipping: {model_name} × {prompt_name} (failed to load)")
                continue
                
            # For debugging
            print(f"Synthetic data columns: {synthetic_data.columns.tolist()}")
            print(f"Real data columns: {real_data.columns.tolist()}")
            
            # Before evaluation, ensure required columns exist
            required_columns = [
                "ID", "N_Days", "Status", "Drug", "Age", "Sex", "Ascites", 
                "Hepatomegaly", "Spiders", "Edema", "Bilirubin", "Cholesterol", 
                "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", 
                "Platelets", "Prothrombin", "Stage"
            ]
            
            for col in required_columns:
                if col not in synthetic_data.columns:
                    print(f"⚠️ Warning: Column '{col}' missing in {model_name} × {prompt_name}")
            
            # Run evaluation
            try:
                results = evaluate_model_prompt(real_data, synthetic_data, model_name, prompt_name)
                all_results.append(results)
            except Exception as e:
                print(f"❌ Evaluation completely failed: {traceback.format_exc()}")
                all_results.append({
                    "model": model_name, 
                    "prompt": prompt_name, 
                    "error": str(e)
                })
    
    # Save results
    if all_results:
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save to CSV
        results_df.to_csv(args.output, index=False)
        print(f"✅ Evaluation summary saved to {args.output}")
    else:
        print("❌ No results to save")

if __name__ == "__main__":
    main()