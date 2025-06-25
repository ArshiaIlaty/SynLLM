import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

# Set up logging
log_dir = "opensource/Diabetes/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(
    log_dir, f"diabetes_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

# Log GPU information
logging.info(
    f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}"
)
logging.info(f"Available GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    logging.info(f"Current GPU: {torch.cuda.current_device()}")
    logging.info(f"GPU Name: {torch.cuda.get_device_name()}")

# Add the Eval directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "Eval"))
from fixed_evaluator import DiabetesDataEvaluator

# Import functions from solo-prompt.py
sys.path.append(os.path.dirname(__file__))
from solo_prompt import SYSTEM_PROMPT, generate_synthetic_data

# Constants
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
REAL_DATA_PATH = (
    "/home/ailaty3088@id.sdsu.edu/SynLLM/datasets/diabetes_prediction_dataset_900K.csv"
)
PROMPTS_DIR = "opensource/Diabetes/prompts"
REPORTS_DIR = "opensource/Diabetes/reports"


def load_real_data(data_path: str) -> pd.DataFrame:
    """Load the real dataset."""
    return pd.read_csv(data_path)


def run_synthetic_generation(prompt_file: str, iteration: int) -> str:
    """Generate synthetic data using the specified prompt."""
    output_dir = f"bash/{MODEL_NAME.replace('/', '_')}/records_prompt{Path(prompt_file).stem[-1]}_iter{iteration}"
    output_file = f"{output_dir}/diabetes_records.csv"

    logging.info(f"Checking output directory: {output_dir}")
    logging.info(f"Full output file path: {output_file}")

    # Check if data already exists for this iteration
    if os.path.exists(output_file):
        logging.info(f"Found existing data for {output_file}, skipping generation...")
        return output_file

    # Create directory with verbose output
    logging.info(f"Creating directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(output_dir):
        raise RuntimeError(f"Failed to create directory: {output_dir}")

    try:
        # Read the prompt file
        with open(prompt_file, "r") as f:
            prompt = f.read()

        # Generate synthetic data
        logging.info(f"Generating synthetic data using model {MODEL_NAME}")
        synthetic_data = generate_synthetic_data(prompt, MODEL_NAME)

        # Save the generated data
        synthetic_data.to_csv(output_file, index=False)
        logging.info(f"Successfully generated data at: {output_file}")

        return output_file

    except Exception as e:
        logging.error(f"Error generating synthetic data: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to generate synthetic data: {str(e)}")


def evaluate_synthetic_data(
    real_data: pd.DataFrame, synthetic_data_path: str, prompt_name: str, iteration: int
) -> Dict:
    """Evaluate synthetic data using the fixed evaluator."""
    synthetic_data = pd.read_csv(synthetic_data_path)
    evaluator = DiabetesDataEvaluator(real_data, synthetic_data)

    # Get all evaluation metrics
    results = evaluator.evaluate_flat(
        model_name=MODEL_NAME, prompt_name=f"{prompt_name}_iter{iteration}"
    )

    # Log evaluation results
    logging.info("\nEvaluation Results:")
    for metric, value in results.items():
        logging.info(f"{metric}: {value}")

    return results


def update_prompt_with_metrics(prompt_file: str, metrics: Dict, iteration: int) -> str:
    """Update the prompt with evaluation metrics."""
    with open(prompt_file, "r") as f:
        prompt = f.read()

    # Add metrics summary to the prompt
    metrics_summary = f"\n\n{'='*50}\n"
    metrics_summary += f"Metrics for Iteration {iteration}\n"
    metrics_summary += f"{'='*50}\n"

    # Calculate overall scores
    distribution_similarity = metrics.get("distribution_metrics_age_wasserstein", 0)
    privacy_score = metrics.get("privacy_assessment_overall_privacy_score", 0)
    data_quality_score = metrics.get("utility_metrics_TSTR_accuracy", 0)

    # Get MIA metrics
    mia_vulnerability = metrics.get("mia_vulnerability_score", 0)
    mia_black_box_auc = metrics.get("mia_black_box_auc", 0)
    mia_white_box_auc = metrics.get("mia_white_box_auc", 0)

    # Format each metric with proper rounding and handling of missing values
    metrics_summary += f"- Distribution similarity: {distribution_similarity:.3f}\n"
    metrics_summary += f"- Privacy score: {privacy_score:.3f}\n"
    metrics_summary += f"- Data quality score: {data_quality_score:.3f}\n"
    metrics_summary += f"- MIA vulnerability score: {mia_vulnerability:.3f}\n"
    metrics_summary += f"  - Black-box attack AUC: {mia_black_box_auc:.3f}\n"
    metrics_summary += f"  - White-box attack AUC: {mia_white_box_auc:.3f}\n"

    # Add specific improvement suggestions based on metrics
    metrics_summary += "\nPlease improve the synthetic data generation by:\n"

    if distribution_similarity < 0.7:
        metrics_summary += (
            "- Ensuring better alignment with the real data distributions\n"
        )

    if privacy_score < 0.8 or mia_vulnerability > 0.7:
        metrics_summary += (
            "- Generating more diverse and unique records to improve privacy\n"
        )
        if mia_vulnerability > 0.7:
            metrics_summary += (
                "- Making the synthetic data less distinguishable from real data\n"
            )

    if data_quality_score < 0.8:
        metrics_summary += "- Maintaining better data quality and consistency\n"

    # Save updated prompt
    updated_prompt_file = f"{PROMPTS_DIR}/prompt3_iter{iteration+1}.txt"
    with open(updated_prompt_file, "w") as f:
        f.write(prompt + metrics_summary)

    # Also save a copy with iteration number for history
    history_prompt_file = f"{PROMPTS_DIR}/prompt3_history_iter{iteration}.txt"
    with open(history_prompt_file, "w") as f:
        f.write(prompt + metrics_summary)

    return updated_prompt_file


def save_prompt_history(prompt_path: str, iteration: int):
    """Save a copy of the original prompt with iteration number."""
    with open(prompt_path, "r") as f:
        prompt = f.read()

    history_prompt_file = f"{PROMPTS_DIR}/prompt3_history_iter{iteration}.txt"
    with open(history_prompt_file, "w") as f:
        f.write(prompt)


def save_iteration_results(results: List[Dict], iteration: int):
    """Save iteration results to a JSON file."""
    results_file = f"{REPORTS_DIR}/iteration_{iteration}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


def main():
    logging.info("Starting diabetes evaluation process")
    logging.info(f"Results will be saved in: {REPORTS_DIR}")
    logging.info(f"Logs will be saved in: {log_dir}")

    # Create necessary directories
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load real data
    logging.info(f"Loading real data from: {REAL_DATA_PATH}")
    real_data = load_real_data(REAL_DATA_PATH)

    # Focus only on prompt 3
    prompt_file = "prompt3.txt"
    prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
    prompt_name = Path(prompt_file).stem

    # Run iterations
    for iteration in range(10):
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting iteration {iteration}")
        logging.info(f"{'='*50}")

        try:
            # Save original prompt for this iteration
            if iteration == 0:
                save_prompt_history(prompt_path, iteration)
            else:
                # For subsequent iterations, use the updated prompt from previous iteration
                prompt_path = f"{PROMPTS_DIR}/prompt3_iter{iteration}.txt"
                if not os.path.exists(prompt_path):
                    logging.error(f"Error: Updated prompt not found at {prompt_path}")
                    break
                logging.info(
                    f"Using updated prompt from iteration {iteration-1}: {prompt_path}"
                )
                save_prompt_history(prompt_path, iteration)

            # Check for existing data first
            synthetic_data_path = f"bash/{MODEL_NAME.replace('/', '_')}/records_prompt3_iter{iteration}/diabetes_records.csv"

            if not os.path.exists(synthetic_data_path):
                logging.info(f"Generating synthetic data for iteration {iteration}...")
                synthetic_data_path = run_synthetic_generation(prompt_path, iteration)
            else:
                logging.info(f"Using existing data from: {synthetic_data_path}")

            # Verify the data file exists
            if not os.path.exists(synthetic_data_path):
                raise RuntimeError(
                    f"Data file not found after generation: {synthetic_data_path}"
                )

            # Evaluate synthetic data
            logging.info("Evaluating synthetic data...")
            metrics = evaluate_synthetic_data(
                real_data, synthetic_data_path, prompt_name, iteration
            )

            # Save iteration results
            save_iteration_results([metrics], iteration)

            # Update prompt with metrics for next iteration (except for last iteration)
            if iteration < 9:
                logging.info("Updating prompt with metrics for next iteration...")
                updated_prompt = update_prompt_with_metrics(
                    prompt_path, metrics, iteration
                )
                logging.info(f"Updated prompt saved to: {updated_prompt}")

        except Exception as e:
            logging.error(
                f"Error during iteration {iteration}: {str(e)}", exc_info=True
            )
            raise

    logging.info("\nAll iterations complete!")
    logging.info("Results have been saved in the reports directory.")
    logging.info("Final synthetic data can be found in:")
    for i in range(10):
        logging.info(
            f"- Iteration {i}: bash/{MODEL_NAME.replace('/', '_')}/records_prompt3_iter{i}/diabetes_records.csv"
        )
    logging.info("\nPrompt history can be found in:")
    for i in range(10):
        logging.info(f"- Iteration {i}: {PROMPTS_DIR}/prompt3_history_iter{i}.txt")


if __name__ == "__main__":
    main()
