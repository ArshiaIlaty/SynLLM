import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add the Eval directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "Eval"))
from fixed_evaluator import DiabetesDataEvaluator

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

    # Print evaluation results for debugging
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

    return results


def update_prompt_with_metrics(prompt_file: str, metrics: Dict, iteration: int) -> str:
    """Update the prompt with evaluation metrics."""
    with open(prompt_file, "r") as f:
        prompt = f.read()

    # Add metrics summary to the prompt
    metrics_summary = "\n\nBased on the previous generation's evaluation:\n"

    # Calculate overall scores
    distribution_similarity = metrics.get("distribution_metrics_age_wasserstein", 0)
    privacy_score = metrics.get("privacy_assessment_overall_privacy_score", 0)
    data_quality_score = metrics.get("utility_metrics_TSTR_accuracy", 0)

    # Format each metric with proper rounding and handling of missing values
    metrics_summary += f"- Distribution similarity: {distribution_similarity:.3f}\n"
    metrics_summary += f"- Privacy score: {privacy_score:.3f}\n"
    metrics_summary += f"- Data quality score: {data_quality_score:.3f}\n"

    # Add specific improvement suggestions based on metrics
    metrics_summary += "\nPlease improve the synthetic data generation by:\n"

    if distribution_similarity < 0.7:
        metrics_summary += (
            "- Ensuring better alignment with the real data distributions\n"
        )

    if privacy_score < 0.8:
        metrics_summary += (
            "- Generating more diverse and unique records to improve privacy\n"
        )

    if data_quality_score < 0.8:
        metrics_summary += "- Maintaining better data quality and consistency\n"

    # Save updated prompt
    updated_prompt_file = (
        f"{PROMPTS_DIR}/prompt{Path(prompt_file).stem[-1]}_iter{iteration+1}.txt"
    )
    with open(updated_prompt_file, "w") as f:
        f.write(prompt + metrics_summary)

    return updated_prompt_file


def save_iteration_results(results: List[Dict], iteration: int):
    """Save iteration results to a JSON file."""
    results_file = f"{REPORTS_DIR}/iteration_{iteration}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


def main():
    # Create necessary directories
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load real data
    real_data = load_real_data(REAL_DATA_PATH)

    # Process existing prompts 1 and 2
    for prompt_num in [1, 2]:
        prompt_file = f"prompt{prompt_num}.txt"
        prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
        prompt_name = Path(prompt_file).stem

        print(f"\nProcessing {prompt_name}...")

        # Use existing generated data - fix path construction
        synthetic_data_path = f"bash/{MODEL_NAME.replace('/', '_')}/records_prompt{prompt_num}_iter0/diabetes_records.csv"

        if not os.path.exists(synthetic_data_path):
            print(f"Warning: No existing data found at {synthetic_data_path}")
            continue

        print(f"Using existing data from: {synthetic_data_path}")

        # Evaluate synthetic data
        print("Evaluating synthetic data...")
        metrics = evaluate_synthetic_data(
            real_data, synthetic_data_path, prompt_name, 0
        )

        # Update prompt with metrics for next iteration
        print("Updating prompt with metrics for iteration 1...")
        updated_prompt = update_prompt_with_metrics(prompt_path, metrics, 0)

        print(
            f"\nEvaluation complete for {prompt_name}. Updated prompt saved to: {updated_prompt}"
        )

    print(
        "\nAll evaluations complete. You can now review the updated prompts for iteration 1."
    )


if __name__ == "__main__":
    main()
