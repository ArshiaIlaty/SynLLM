import os
import pandas as pd
import numpy as np
import warnings
import json
from pathlib import Path

# Import your evaluator class (assuming it's in the same directory)
# For example, if you're working with the stroke data:
# from stroke_evaluator import StrokeDataEvaluator
# If you're working with cirrhosis data:
# from cirrhosis_evaluator import CirrhosisDataEvaluator
# If you're working with diabetes data:
from diabetes_evaluator import DiabetesDataEvaluator

# Suppress warnings
warnings.filterwarnings("ignore")

def evaluate_experiments(
    original_data_path, 
    experiments_root_dir, 
    output_dir=None,
    evaluator_class=DiabetesDataEvaluator, # Change this to your actual evaluator class
    file_pattern="PROMPT_*.csv"
):
    """
    Evaluate multiple experiments and their synthetic datasets
    
    Parameters:
    -----------
    original_data_path : str
        Path to the original dataset (e.g., 'diabetes_prediction_dataset.csv')
    experiments_root_dir : str
        Path to the root directory containing experiment folders
    output_dir : str, optional
        Directory to save evaluation results, defaults to 'evaluation_results_diabetes'
    evaluator_class : class
        The evaluator class to use (DiabetesDataEvaluator)
    file_pattern : str, optional
        Pattern to match synthetic data files, defaults to 'PROMPT_*.csv'
    """
    
    # """
    # Evaluate multiple experiments and their synthetic datasets
    
    # Parameters:
    # -----------
    # original_data_path : str
    #     Path to the original dataset (e.g., 'stroke.csv')
    # experiments_root_dir : str
    #     Path to the root directory containing experiment folders
    # output_dir : str, optional
    #     Directory to save evaluation results, defaults to 'evaluation_results'
    # evaluator_class : class
    #     The evaluator class to use (e.g., StrokeDataEvaluator)
    # file_pattern : str, optional
    #     Pattern to match synthetic data files, defaults to 'prompt_*.csv'
    # """
    # Create output directory if it doesn't exist
    if output_dir is None:
        # output_dir = "evaluation_results_stroke"
        output_dir = "diabetes_evaluation_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original dataset
    print(f"Loading original dataset from {original_data_path}...")
    try:
        real_data = pd.read_csv(original_data_path)
        print(f"Original data shape: {real_data.shape}")
    except Exception as e:
        print(f"Error loading original data: {e}")
        return
    
    # Find all experiment directories
    experiment_dirs = [d for d in os.listdir(experiments_root_dir) 
                      if os.path.isdir(os.path.join(experiments_root_dir, d))]
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    # Process each experiment
    for exp_dir in sorted(experiment_dirs):
        exp_path = os.path.join(experiments_root_dir, exp_dir)
        print(f"\n{'='*50}")
        print(f"Processing experiment: {exp_dir}")
        
        # Create experiment output directory
        exp_output_dir = os.path.join(output_dir, exp_dir)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Find all synthetic data files in this experiment
        synthetic_files = []
        for root, _, files in os.walk(exp_path):
            for file in files:
                if file.endswith('.csv') and (file_pattern == "*" or file.startswith('PROMPT_')):
                    synthetic_files.append(os.path.join(root, file))
        
        print(f"Found {len(synthetic_files)} synthetic datasets in experiment {exp_dir}")
        
        # Skip if no files found
        if not synthetic_files:
            print(f"No synthetic datasets found in {exp_dir}, skipping...")
            continue

        # Sort files by prompt number
        try:
            synthetic_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x).split('_')[1]))))
        except:
            print("Could not sort files by prompt number, using default order")
        
        # Process each synthetic dataset
        for synth_file_path in sorted(synthetic_files):
            synth_file_name = os.path.basename(synth_file_path)
            print(f"\n--- Evaluating {synth_file_name} ---")
            
            try:
                # Load synthetic data
                synthetic_data = pd.read_csv(synth_file_path)
                print(f"Synthetic data shape: {synthetic_data.shape}")
                
                # Skip if synthetic data is empty
                if synthetic_data.empty:
                    print(f"Synthetic dataset {synth_file_name} is empty, skipping...")
                    continue
                
                # Initialize evaluator
                print("Initializing evaluator...")
                evaluator = evaluator_class(real_data, synthetic_data)
                
                # Run evaluation
                print("Running evaluation...")
                evaluation_results = evaluator.evaluate_all()
                
                # Save results in JSON format for easier reading
                results_file = os.path.join(exp_output_dir, f"results_{synth_file_name.replace('.csv', '.json')}")
                
                # Convert any non-serializable objects to strings
                def convert_to_serializable(obj):
                    if isinstance(obj, (np.integer, np.floating, np.bool_)):
                        return obj.item()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict()
                    elif isinstance(obj, pd.Series):
                        return obj.to_dict()
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(i) for i in obj]
                    elif isinstance(obj, tuple):
                        return tuple(convert_to_serializable(i) for i in obj)
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    return str(obj)
                
                # Convert results to serializable format
                serializable_results = convert_to_serializable(evaluation_results)
                
                # Save to JSON
                with open(results_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                
                # Also save a plain text version for readability
                text_file = os.path.join(exp_output_dir, f"results_{synth_file_name.replace('.csv', '.txt')}")
                with open(text_file, 'w') as f:
                    f.write(f"Evaluation Results for {synth_file_name}:\n")
                    for metric, results in evaluation_results.items():
                        f.write(f"\n{metric.upper()}:\n")
                        f.write(str(results))
                        f.write("\n")
                
                print(f"Results saved to {results_file} and {text_file}")
                
            except Exception as e:
                print(f"Error processing {synth_file_name}: {str(e)}")
                continue
    
    print("\nEvaluation complete!")

# def evaluate_all_datasets(datasets_config):
#     """
#     Run evaluations on multiple datasets

#     Parameters:
#     -----------
#     datasets_config : list of dict
#         List of configurations for each dataset:
#         [
#             {
#                 'name': 'diabetes',
#                 'original_data': '/path/to/diabetes.csv',
#                 'experiments_dir': '/path/to/diabetes_experiments',
#                 'evaluator_class': DiabetesDataEvaluator,
#                 'file_pattern': 'PROMPT_*.csv'
#             },
#             ...
#         ]
#     """
#     for config in datasets_config:
#         print(f"\n\n{'='*70}")
#         print(f"PROCESSING {config['name'].upper()} DATASET")
#         print(f"{'='*70}")
        
#         output_dir = f"evaluation_results_{config['name']}"
        
#         evaluate_experiments(
#             original_data_path=config['original_data'],
#             experiments_root_dir=config['experiments_dir'],
#             output_dir=output_dir,
#             evaluator_class=config['evaluator_class'],
#             file_pattern=config.get('file_pattern', 'PROMPT_*.csv')
#         )

if __name__ == "__main__":
    # Example usage
    original_data = "/home/jovyan/SynLLM/datasets/diabetes_prediction_dataset.csv"  # Change to your actual original dataset
    experiments_dir = "/home/jovyan/SynLLM/evaluation/gpt2-diabetes-data-100-5experiments"  # Directory containing experiment folders
    
    # Choose the appropriate evaluator class based on your dataset
    # evaluator_class = StrokeDataEvaluator  # For stroke data
    # evaluator_class = CirrhosisDataEvaluator  # For cirrhosis data
    # evaluator_class = DiabetesDataEvaluator  # For diabetes data
    
    evaluate_experiments(
        original_data_path=original_data,
        experiments_root_dir=experiments_dir,
        evaluator_class=DiabetesDataEvaluator,  # Change this to match your dataset
        file_pattern="PROMPT_*.csv"  # Change if your files follow a different pattern
    )

    # Alternative: To evaluate multiple datasets, use the following:
    """
    datasets_config = [
        {
            'name': 'diabetes',
            'original_data': '/home/jovyan/SynLLM/datasets/diabetes_prediction_dataset.csv',
            'experiments_dir': '/home/jovyan/SynLLM/evaluation/gpt2-diabetes-data-100-5experiments',
            'evaluator_class': DiabetesDataEvaluator,
            'file_pattern': 'PROMPT_*.csv'
        },
        {
            'name': 'stroke',
            'original_data': '/home/jovyan/SynLLM/datasets/stroke.csv',
            'experiments_dir': '/home/jovyan/SynLLM/evaluation/gpt2-stroke-data-experiments',
            'evaluator_class': StrokeDataEvaluator,
            'file_pattern': 'PROMPT_*.csv'
        }
    ]
    
    evaluate_all_datasets(datasets_config)
    """
