import json
import os
import subprocess
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
ITERATIONS = 2  # Number of iterations for the prompt engineering loop


def load_real_data(data_path: str) -> pd.DataFrame:
    """Load the real dataset."""
    return pd.read_csv(data_path)


def run_synthetic_generation(prompt_file: str, iteration: int) -> str:
    """Run the solo-prompt.py script to generate synthetic data."""
    output_dir = f"bash/{MODEL_NAME.replace('/', '_')}/records_prompt{Path(prompt_file).stem[-1]}_iter{iteration}"
    output_file = f"{output_dir}/diabetes_records.csv"

    # Check if data already exists for this iteration
    if os.path.exists(output_file):
        print(f"Found existing data for {output_file}, skipping generation...")
        return output_file

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python",
        "opensource/Diabetes/solo-prompt.py",
        "--model_name",
        MODEL_NAME,
        "--prompt_file",
        prompt_file,
        "--output_dir",
        output_dir,  # Pass the output directory to solo-prompt.py
    ]

    subprocess.run(cmd, check=True)
    return output_file


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

    # Format each metric with proper rounding and handling of missing values
    if "distribution_similarity" in metrics:
        dist_sim = metrics["distribution_similarity"]
        if isinstance(dist_sim, (int, float)):
            metrics_summary += f"- Distribution similarity: {dist_sim:.3f}\n"
        else:
            metrics_summary += f"- Distribution similarity: {dist_sim}\n"

    if "privacy_score" in metrics:
        privacy = metrics["privacy_score"]
        if isinstance(privacy, (int, float)):
            metrics_summary += f"- Privacy score: {privacy:.3f}\n"
        else:
            metrics_summary += f"- Privacy score: {privacy}\n"

    if "data_quality_score" in metrics:
        quality = metrics["data_quality_score"]
        if isinstance(quality, (int, float)):
            metrics_summary += f"- Data quality score: {quality:.3f}\n"
        else:
            metrics_summary += f"- Data quality score: {quality}\n"

    # Add specific improvement suggestions based on metrics
    metrics_summary += "\nPlease improve the synthetic data generation by:\n"

    if "distribution_similarity" in metrics and isinstance(
        metrics["distribution_similarity"], (int, float)
    ):
        if metrics["distribution_similarity"] < 0.7:
            metrics_summary += (
                "- Ensuring better alignment with the real data distributions\n"
            )

    if "privacy_score" in metrics and isinstance(
        metrics["privacy_score"], (int, float)
    ):
        if metrics["privacy_score"] < 0.8:
            metrics_summary += (
                "- Generating more diverse and unique records to improve privacy\n"
            )

    if "data_quality_score" in metrics and isinstance(
        metrics["data_quality_score"], (int, float)
    ):
        if metrics["data_quality_score"] < 0.8:
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

        # Use existing generated data
        synthetic_data_path = f"bash/{MODEL_NAME.replace('/', '_')}/records_prompt{prompt_num}/diabetes_records.csv"

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
