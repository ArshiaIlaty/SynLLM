import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Regular expression pattern to extract numpy values from text format
np_pattern = re.compile(r"np\.float64\(([^)]+)\)")


def extract_value(value_str):
    """Extract numerical values from string that might contain np.float64 notation"""
    if isinstance(value_str, str):
        # Check if it's an np.float64 representation
        match = np_pattern.search(value_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return value_str
        # Try to convert to float directly
        try:
            return float(value_str)
        except ValueError:
            return value_str
    return value_str


def parse_results_file(file_path):
    """
    Parse evaluation results from either JSON or text format
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Determine file type
    if file_path.endswith(".json"):
        # Load JSON file
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith(".txt"):
        # Parse text file
        results = {}
        current_section = None

        with open(file_path, "r") as f:
            content = f.read()

            # Split the content by the headers (in uppercase followed by colon)
            sections = re.split(r"\n([A-Z_]+):\n", content)

            # First element is the intro text, skip it
            if not sections[0].strip().upper().endswith("RESULTS"):
                sections = sections[1:]

            # Process sections in pairs (name, content)
            for i in range(0, len(sections), 2):
                if i + 1 < len(sections):
                    section_name = sections[i].strip()
                    section_content = sections[i + 1].strip()

                    # Try to evaluate the section content to convert it to a Python object
                    try:
                        # Replace np.float64 with actual values
                        section_content = np_pattern.sub(r"\1", section_content)
                        # Convert 'nan' to actual NaN
                        section_content = section_content.replace("nan", 'float("nan")')
                        # Try to evaluate
                        section_data = eval(section_content)
                        results[section_name] = section_data
                    except:
                        # If evaluation fails, just store the raw text
                        results[section_name] = section_content

        return results
    else:
        print(f"Unsupported file type: {file_path}")
        return None


def extract_key_metrics(results, dataset_type="diabetes"):
    """
    Extract key metrics from evaluation results for comparison

    Parameters:
    -----------
    results : dict
        Parsed evaluation results
    dataset_type : str
        Type of dataset (diabetes, stroke, or cirrhosis)

    Returns:
    --------
    dict
        Dictionary of key metrics
    """
    metrics = {}

    # Check if results is None
    if results is None:
        return {"error": "No results data available"}

    # Common metrics regardless of dataset type
    try:
        # Get correlation matrix distance (synthetic vs real)
        if "feature_correlations" in results:
            if (
                isinstance(results["feature_correlations"], dict)
                and "correlation_matrix_distance" in results["feature_correlations"]
            ):
                metrics["correlation_distance"] = extract_value(
                    results["feature_correlations"]["correlation_matrix_distance"]
                )

        # Get privacy risk
        if "privacy_assessment" in results:
            privacy_metrics = results["privacy_assessment"]
            violation_rates = []
            for key, value in privacy_metrics.items():
                if isinstance(value, dict) and "violation_rate" in value:
                    violation_rates.append(extract_value(value["violation_rate"]))
            if violation_rates:
                metrics["avg_privacy_violation"] = np.mean(violation_rates)

        # Extract distribution metrics (average across numerical features)
        if "distribution_metrics" in results:
            dist_metrics = results["distribution_metrics"]
            wasserstein_values = []
            jensen_shannon_values = []
            kl_divergence_values = []

            for feature, values in dist_metrics.items():
                if isinstance(values, dict):
                    if "wasserstein" in values:
                        wasserstein_values.append(extract_value(values["wasserstein"]))
                    if "jensen_shannon" in values:
                        jensen_shannon_values.append(
                            extract_value(values["jensen_shannon"])
                        )
                    if "kl_divergence" in values:
                        kl_divergence_values.append(
                            extract_value(values["kl_divergence"])
                        )

            if wasserstein_values:
                metrics["avg_wasserstein"] = np.mean(
                    [x for x in wasserstein_values if not np.isnan(x)]
                )
            if jensen_shannon_values:
                metrics["avg_jensen_shannon"] = np.mean(
                    [x for x in jensen_shannon_values if not np.isnan(x)]
                )
            if kl_divergence_values:
                metrics["avg_kl_divergence"] = np.mean(
                    [x for x in kl_divergence_values if not np.isnan(x)]
                )

        # Extract numerical statistics
        if "numerical_statistics" in results:
            num_stats = results["numerical_statistics"]
            mean_errors = []
            std_diffs = []
            mean_diffs = []

            for feature, values in num_stats.items():
                if isinstance(values, dict):
                    if "relative_mean_error" in values:
                        mean_errors.append(extract_value(values["relative_mean_error"]))
                    if "std_difference" in values:
                        std_diffs.append(extract_value(values["std_difference"]))
                    if "mean_difference" in values:
                        mean_diffs.append(extract_value(values["mean_difference"]))

            if mean_errors:
                metrics["avg_relative_mean_error"] = np.mean(
                    [x for x in mean_errors if not np.isnan(x)]
                )
            if std_diffs:
                metrics["avg_std_difference"] = np.mean(
                    [x for x in std_diffs if not np.isnan(x)]
                )
            if mean_diffs:
                metrics["avg_mean_difference"] = np.mean(
                    [x for x in mean_diffs if not np.isnan(x)]
                )

        # Extract statistical test results if available
        if "advanced_metrics" in results and isinstance(
            results["advanced_metrics"], dict
        ):
            adv_metrics = results["advanced_metrics"]
            if "statistical_tests" in adv_metrics and isinstance(
                adv_metrics["statistical_tests"], dict
            ):
                stat_tests = adv_metrics["statistical_tests"]
                ks_pvalues = []
                ad_stats = []

                for feature, values in stat_tests.items():
                    if isinstance(values, dict):
                        if "ks_pvalue" in values:
                            val = extract_value(values["ks_pvalue"])
                            if val is not None and not isinstance(val, str):
                                ks_pvalues.append(val)
                        if "anderson_darling_statistic" in values:
                            val = extract_value(values["anderson_darling_statistic"])
                            if val is not None and not isinstance(val, str):
                                ad_stats.append(val)

                if ks_pvalues:
                    metrics["avg_ks_pvalue"] = np.mean(
                        [x for x in ks_pvalues if not np.isnan(x)]
                    )
                if ad_stats:
                    metrics["avg_anderson_darling"] = np.mean(
                        [x for x in ad_stats if not np.isnan(x)]
                    )
    except Exception as e:
        metrics["error"] = f"Error extracting common metrics: {str(e)}"

    # Dataset-specific metrics
    try:
        if dataset_type == "diabetes":
            # Extract diabetes-specific metrics
            if "medical_consistency" in results:
                med_consistency = results["medical_consistency"]
                if isinstance(med_consistency, dict):
                    if "hba1c_diabetes_relationship" in med_consistency:
                        hba1c_rel = med_consistency["hba1c_diabetes_relationship"]
                        if (
                            isinstance(hba1c_rel, dict)
                            and "real_difference" in hba1c_rel
                            and "synthetic_difference" in hba1c_rel
                        ):
                            real_diff = extract_value(hba1c_rel["real_difference"])
                            synth_diff = extract_value(
                                hba1c_rel["synthetic_difference"]
                            )
                            if real_diff != 0:
                                metrics["hba1c_diff_preservation"] = 1 - abs(
                                    real_diff - synth_diff
                                ) / abs(real_diff)
                            metrics["hba1c_real_diff"] = real_diff
                            metrics["hba1c_synth_diff"] = synth_diff

                    if "glucose_diabetes_relationship" in med_consistency:
                        glucose_rel = med_consistency["glucose_diabetes_relationship"]
                        if (
                            isinstance(glucose_rel, dict)
                            and "real_difference" in glucose_rel
                            and "synthetic_difference" in glucose_rel
                        ):
                            real_diff = extract_value(glucose_rel["real_difference"])
                            synth_diff = extract_value(
                                glucose_rel["synthetic_difference"]
                            )
                            if real_diff != 0:
                                metrics["glucose_diff_preservation"] = 1 - abs(
                                    real_diff - synth_diff
                                ) / abs(real_diff)
                            metrics["glucose_real_diff"] = real_diff
                            metrics["glucose_synth_diff"] = synth_diff

        elif dataset_type == "stroke":
            # Extract stroke-specific metrics
            if "medical_consistency" in results:
                med_consistency = results["medical_consistency"]
                if isinstance(med_consistency, dict):
                    if "glucose_stroke_relationship" in med_consistency:
                        glucose_rel = med_consistency["glucose_stroke_relationship"]
                        if (
                            isinstance(glucose_rel, dict)
                            and "difference_in_difference" in glucose_rel
                        ):
                            metrics["glucose_stroke_diff"] = extract_value(
                                glucose_rel["difference_in_difference"]
                            )

                        # Add more granular metrics
                        if (
                            isinstance(glucose_rel, dict)
                            and "real_means" in glucose_rel
                            and "synthetic_means" in glucose_rel
                        ):
                            real_means = glucose_rel["real_means"]
                            synth_means = glucose_rel["synthetic_means"]

                            if isinstance(real_means, dict) and isinstance(
                                synth_means, dict
                            ):
                                # Get the values for stroke=0 and stroke=1
                                if 0 in real_means and 1 in real_means:
                                    metrics["glucose_real_no_stroke"] = extract_value(
                                        real_means[0]
                                    )
                                    metrics["glucose_real_stroke"] = extract_value(
                                        real_means[1]
                                    )
                                if 0 in synth_means and 1 in synth_means:
                                    metrics["glucose_synth_no_stroke"] = extract_value(
                                        synth_means[0]
                                    )
                                    metrics["glucose_synth_stroke"] = extract_value(
                                        synth_means[1]
                                    )

                    if "age_stroke_relationship" in med_consistency:
                        age_rel = med_consistency["age_stroke_relationship"]
                        if (
                            isinstance(age_rel, dict)
                            and "difference_in_difference" in age_rel
                        ):
                            metrics["age_stroke_diff"] = extract_value(
                                age_rel["difference_in_difference"]
                            )

                        # Add more granular metrics
                        if (
                            isinstance(age_rel, dict)
                            and "real_means" in age_rel
                            and "synthetic_means" in age_rel
                        ):
                            real_means = age_rel["real_means"]
                            synth_means = age_rel["synthetic_means"]

                            if isinstance(real_means, dict) and isinstance(
                                synth_means, dict
                            ):
                                # Get the values for stroke=0 and stroke=1
                                if 0 in real_means and 1 in real_means:
                                    metrics["age_real_no_stroke"] = extract_value(
                                        real_means[0]
                                    )
                                    metrics["age_real_stroke"] = extract_value(
                                        real_means[1]
                                    )
                                if 0 in synth_means and 1 in synth_means:
                                    metrics["age_synth_no_stroke"] = extract_value(
                                        synth_means[0]
                                    )
                                    metrics["age_synth_stroke"] = extract_value(
                                        synth_means[1]
                                    )

            # Extract stroke prediction metrics
            if "advanced_metrics" in results and isinstance(
                results["advanced_metrics"], dict
            ):
                adv_metrics = results["advanced_metrics"]
                if "stroke_prediction_metrics" in adv_metrics and isinstance(
                    adv_metrics["stroke_prediction_metrics"], dict
                ):
                    stroke_pred = adv_metrics["stroke_prediction_metrics"]
                    if "risk_factor_correlations" in stroke_pred and isinstance(
                        stroke_pred["risk_factor_correlations"], dict
                    ):
                        risk_corr = stroke_pred["risk_factor_correlations"]
                        if "absolute_differences" in risk_corr and isinstance(
                            risk_corr["absolute_differences"], dict
                        ):
                            abs_diffs = risk_corr["absolute_differences"]
                            metrics["avg_risk_factor_diff"] = np.mean(
                                [
                                    extract_value(v)
                                    for v in abs_diffs.values()
                                    if not np.isnan(extract_value(v))
                                ]
                            )

                            # Extract individual risk factor differences
                            for factor, diff in abs_diffs.items():
                                metrics[f"risk_diff_{factor}"] = extract_value(diff)

        elif dataset_type == "cirrhosis":
            # Extract cirrhosis-specific metrics
            if "medical_consistency" in results:
                med_consistency = results["medical_consistency"]
                if isinstance(med_consistency, dict):
                    if "bilirubin_stage_relationship" in med_consistency:
                        bili_rel = med_consistency["bilirubin_stage_relationship"]
                        if isinstance(bili_rel, dict):
                            metrics["bilirubin_stage_preservation"] = 1  # Placeholder

                    if "albumin_stage_relationship" in med_consistency:
                        alb_rel = med_consistency["albumin_stage_relationship"]
                        if isinstance(alb_rel, dict):
                            metrics["albumin_stage_preservation"] = 1  # Placeholder
    except Exception as e:
        metrics["error_specific"] = f"Error extracting {dataset_type} metrics: {str(e)}"

    # Force numeric conversion for all metrics
    for key in list(metrics.keys()):
        if key not in ["error", "error_specific"]:  # Skip error messages
            try:
                # First ensure we have a number, not a string representation
                value = metrics[key]
                if isinstance(value, str):
                    # Try to extract a number from the string
                    numeric_match = re.search(r"[-+]?\d*\.\d+|\d+", value)
                    if numeric_match:
                        metrics[key] = float(numeric_match.group(0))
                    else:
                        # If no numeric part, remove non-numeric metric
                        del metrics[key]
                elif value is None or np.isnan(value):
                    # Remove None or NaN values
                    del metrics[key]
            except:
                # If conversion fails, keep original value
                pass

    # Debug: print the extracted metrics
    print(f"Extracted {len(metrics)} metrics: {list(metrics.keys())}")

    return metrics


def extract_prompt_number(filename):
    """Extract prompt number from filename with better error handling"""
    # Try different pattern matches
    patterns = [
        r"prompt[_-](\d+)",  # matches prompt_1 or prompt-1
        r"PROMPT[_-](\d+)",  # matches PROMPT_1 or PROMPT-1
        r"p(\d+)",  # matches p1
        r"(\d+)\.(?:json|txt)",  # matches 1.json or 1.txt
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # If no pattern matches, use a simple number extraction
    numbers = re.findall(r"\d+", filename)
    if numbers:
        return int(numbers[0])

    # If all else fails, return a default value
    print(f"Could not extract prompt number from {filename}, using 999")
    return 999


def list_files_in_dir(directory):
    """Debug function to list files in a directory"""
    print(f"Files in {directory}:")
    if os.path.exists(directory):
        for file in os.listdir(directory):
            print(f"  - {file}")
    else:
        print(f"  Directory does not exist!")


def compare_within_experiment(
    results_dir, experiment_name, dataset_type="diabetes", output_format="dataframe"
):
    """
    Compare results across different prompts within the same experiment

    Parameters:
    -----------
    results_dir : str
        Directory containing the evaluation results
    experiment_name : str
        Name of the experiment to analyze
    dataset_type : str
        Type of dataset (diabetes, stroke, or cirrhosis)
    output_format : str
        Format of the output ('dataframe', 'plot', or 'both')

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with comparison results, or None if output_format is 'plot'
    """
    # Create full path to experiment directory
    exp_dir = os.path.join(results_dir, experiment_name)

    if not os.path.exists(exp_dir):
        print(f"Experiment directory not found: {exp_dir}")
        # Debug: List contents of the parent directory
        list_files_in_dir(results_dir)
        return None

    # Find all result files, preferring JSON over TXT
    result_files = []
    processed_prompts = set()

    # First pass: collect JSON files
    for file in os.listdir(exp_dir):
        if (file.startswith("results_") or "PROMPT" in file.upper()) and file.endswith(
            ".json"
        ):
            prompt_num = extract_prompt_number(file)
            if prompt_num not in processed_prompts:
                result_files.append(file)
                processed_prompts.add(prompt_num)

    # If we didn't find enough JSON files, add TXT files for missing prompts
    for file in os.listdir(exp_dir):
        if (file.startswith("results_") or "PROMPT" in file.upper()) and file.endswith(
            ".txt"
        ):
            prompt_num = extract_prompt_number(file)
            if prompt_num not in processed_prompts:
                result_files.append(file)
                processed_prompts.add(prompt_num)

    if not result_files:
        print(f"No result files found in {exp_dir}")
        # Debug: List contents of the experiment directory
        list_files_in_dir(exp_dir)
        return None

    print(f"Found {len(result_files)} result files: {result_files}")

    # Sort result files by prompt number
    try:
        result_files.sort(key=extract_prompt_number)
    except Exception as e:
        print(f"Error sorting files: {e}")
        # Continue with unsorted files

    # Parse each result file and extract key metrics
    all_metrics = {}
    for file in result_files:
        try:
            prompt_num = extract_prompt_number(file)
            file_path = os.path.join(exp_dir, file)
            print(f"Processing {file_path}")
            results = parse_results_file(file_path)
            metrics = extract_key_metrics(results, dataset_type)

            # Force numeric conversion for key metrics
            for key in list(metrics.keys()):
                if key not in ["error", "error_specific"]:  # Skip error messages
                    try:
                        metrics[key] = float(metrics[key])
                    except:
                        pass  # Keep as is if conversion fails

            all_metrics[f"Prompt {prompt_num}"] = metrics
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    if not all_metrics:
        print("No metrics could be extracted from any files.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics).T

    # Debug output to see what data we're working with
    print("\nExtracted metrics dataframe:")
    print(df.dtypes)  # Show data types
    print(df.head())  # Show the first few rows

    # Handle output format
    if output_format in ["plot", "both"]:
        # Create directory for plots
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Create a plot for each metric
        for metric in df.columns:
            if (
                not df[metric].isnull().all()
                and not df[metric].apply(lambda x: isinstance(x, str)).any()
            ):
                try:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=df.index, y=metric, data=df)
                    plt.title(f"{metric} across Prompts in {experiment_name}")
                    plt.ylabel(metric)
                    plt.tight_layout()
                    plot_path = os.path.join(
                        plots_dir, f"{experiment_name}_{metric}_comparison.png"
                    )
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Saved plot to {plot_path}")
                except Exception as e:
                    print(f"Error creating plot for {metric}: {e}")

        # Create a heatmap of all metrics
        try:
            # Filter out columns with non-numeric data or all NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if not df[col].isnull().all()]

            if numeric_cols:  # Only create heatmap if we have numeric columns
                numeric_df = df[numeric_cols]
                plt.figure(figsize=(12, 8))
                sns.heatmap(numeric_df, annot=True, cmap="viridis", fmt=".3f")
                plt.title(f"Metrics Heatmap for {experiment_name}")
                plt.tight_layout()
                heatmap_path = os.path.join(
                    plots_dir, f"{experiment_name}_metrics_heatmap.png"
                )
                plt.savefig(heatmap_path)
                plt.close()
                print(f"Saved heatmap to {heatmap_path}")
            else:
                print("No numeric data columns available for heatmap.")
        except Exception as e:
            print(f"Error creating heatmap: {e}")

        # Fallback for creating simple plots if regular ones failed
        if len(os.listdir(plots_dir)) == 0:  # If plots directory is empty
            print("Using fallback visualization method...")
            for col in df.columns:
                try:
                    # Try to convert to numeric, ignoring errors
                    numeric_data = pd.to_numeric(df[col], errors="coerce")
                    if not numeric_data.isna().all():
                        plt.figure(figsize=(10, 6))
                        numeric_data.plot(kind="bar")
                        plt.title(f"{col} across Prompts in {experiment_name}")
                        plt.ylabel(col)
                        plt.tight_layout()
                        plot_path = os.path.join(
                            plots_dir, f"{experiment_name}_{col}_comparison.png"
                        )
                        plt.savefig(plot_path)
                        plt.close()
                        print(f"Created fallback plot for {col}")
                except Exception as e:
                    print(f"Fallback plotting failed for {col}: {e}")

        print(f"Plots saved to {plots_dir}")

    if output_format in ["dataframe", "both"]:
        return df
    return None


def compare_across_datasets(
    results_dirs, prompt_num, dataset_types, output_format="dataframe"
):
    """
    Compare results for the same prompt number across different datasets

    Parameters:
    -----------
    results_dirs : dict
        Dictionary mapping dataset names to their result directories
    prompt_num : int or str
        Prompt number to compare
    dataset_types : dict
        Dictionary mapping dataset names to their types (diabetes, stroke, or cirrhosis)
    output_format : str
        Format of the output ('dataframe', 'plot', or 'both')

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with comparison results, or None if output_format is 'plot'
    """
    # Normalize prompt_num to string
    prompt_num = str(prompt_num)

    # Find and parse result files for each dataset
    all_metrics = {}
    for dataset_name, results_dir in results_dirs.items():
        if not os.path.exists(results_dir):
            print(f"Results directory for {dataset_name} not found: {results_dir}")
            continue

        # Find all experiment directories
        exp_dirs = [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ]

        if not exp_dirs:
            print(f"No experiment directories found in {results_dir}")
            continue

        print(
            f"Found {len(exp_dirs)} experiment directories for {dataset_name}: {exp_dirs}"
        )

        for exp_dir in exp_dirs:
            # Look for the specific prompt file
            exp_path = os.path.join(results_dir, exp_dir)
            result_file = None

            for file in os.listdir(exp_path):
                file_prompt_num = str(extract_prompt_number(file))
                if file_prompt_num == prompt_num and (
                    file.endswith(".json") or file.endswith(".txt")
                ):
                    result_file = file
                    break

            if result_file:
                file_path = os.path.join(exp_path, result_file)
                print(f"Processing {file_path}")
                results = parse_results_file(file_path)
                metrics = extract_key_metrics(
                    results, dataset_types.get(dataset_name, "unknown")
                )
                all_metrics[f"{dataset_name} ({exp_dir})"] = metrics
            else:
                print(f"No matching prompt {prompt_num} file found in {exp_path}")

    if not all_metrics:
        print(f"No matching prompt files found for prompt {prompt_num}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics).T

    # Handle output format
    if output_format in ["plot", "both"]:
        # Create directory for plots
        plots_dir = "cross_dataset_plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Create a plot for each metric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if not df[col].isnull().all()]

        for metric in numeric_cols:
            if metric in df.columns:
                try:
                    plt.figure(figsize=(12, 7))
                    sns.barplot(x=df.index, y=metric, data=df)
                    plt.title(f"{metric} for Prompt {prompt_num} across Datasets")
                    plt.ylabel(metric)
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plot_path = os.path.join(
                        plots_dir, f"prompt_{prompt_num}_{metric}_cross_dataset.png"
                    )
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Saved plot to {plot_path}")
                except Exception as e:
                    print(f"Error creating plot for {metric}: {e}")

        # Create a heatmap for common metrics only
        if numeric_cols:  # Only create heatmap if we have numeric columns
            common_df = df[numeric_cols].copy()
            if not common_df.empty and len(numeric_cols) > 1:
                try:
                    plt.figure(figsize=(14, 10))
                    sns.heatmap(common_df, annot=True, cmap="viridis", fmt=".3f")
                    plt.title(
                        f"Metrics Heatmap for Prompt {prompt_num} across Datasets"
                    )
                    plt.tight_layout()
                    heatmap_path = os.path.join(
                        plots_dir, f"prompt_{prompt_num}_cross_dataset_heatmap.png"
                    )
                    plt.savefig(heatmap_path)
                    plt.close()
                    print(f"Saved heatmap to {heatmap_path}")
                except Exception as e:
                    print(f"Error creating heatmap: {e}")
        else:
            print("No numeric data columns available for heatmap.")

        print(f"Cross-dataset plots saved to {plots_dir}")

    if output_format in ["dataframe", "both"]:
        return df
    return None


def compare_all_experiments(results_dir, dataset_type="diabetes", output_dir=None):
    """
    Compare results across all experiments for a given dataset

    Parameters:
    -----------
    results_dir : str
        Directory containing the evaluation results
    dataset_type : str
        Type of dataset (diabetes, stroke, or cirrhosis)
    output_dir : str, optional
        Directory to save the output, defaults to 'experiment_comparisons'

    Returns:
    --------
    dict
        Dictionary mapping experiment names to their comparison DataFrames
    """
    if output_dir is None:
        output_dir = f"{dataset_type}_experiment_comparisons"

    os.makedirs(output_dir, exist_ok=True)

    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return {}

    # Find all experiment directories
    exp_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    if not exp_dirs:
        print(f"No experiment directories found in {results_dir}")
        # List files in the directory for debugging
        list_files_in_dir(results_dir)
        return {}

    print(f"Found {len(exp_dirs)} experiment directories: {exp_dirs}")

    # Compare within each experiment
    comparisons = {}
    for exp_dir in exp_dirs:
        print(f"Comparing within experiment: {exp_dir}")
        df = compare_within_experiment(results_dir, exp_dir, dataset_type, "dataframe")
        if df is not None:
            comparisons[exp_dir] = df
            # Save the DataFrame to CSV
            csv_path = os.path.join(output_dir, f"{exp_dir}_comparison.csv")
            df.to_csv(csv_path)
            print(f"Saved comparison to {csv_path}")

    # Create a summary table with average metrics for each experiment
    if comparisons:
        summary_data = {}

        # Find common metrics that are numeric across all experiments
        common_metrics = set()
        for exp_name, df in comparisons.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not common_metrics:
                common_metrics = set(numeric_cols)
            else:
                common_metrics = common_metrics.intersection(set(numeric_cols))

        common_metrics = list(common_metrics)

        if not common_metrics:
            print("No common numeric metrics found across experiments.")
            return comparisons

        print(f"Common numeric metrics across experiments: {common_metrics}")

        # Calculate average metrics for each experiment
        for exp_name, df in comparisons.items():
            avg_metrics = {}
            for metric in common_metrics:
                if metric in df.columns:
                    avg_metrics[metric] = df[metric].mean()
            summary_data[exp_name] = avg_metrics

        summary_df = pd.DataFrame(summary_data).T
        summary_path = os.path.join(output_dir, "experiment_summary.csv")
        summary_df.to_csv(summary_path)
        print(f"Saved experiment summary to {summary_path}")

        # Create summary plots
        plots_dir = os.path.join(output_dir, "summary_plots")
        os.makedirs(plots_dir, exist_ok=True)

        for metric in common_metrics:
            try:
                if metric in summary_df.columns:
                    plt.figure(figsize=(12, 7))
                    sns.barplot(x=summary_df.index, y=metric, data=summary_df)
                    plt.title(f"Average {metric} across Experiments")
                    plt.ylabel(metric)
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plot_path = os.path.join(
                        plots_dir, f"avg_{metric}_experiment_comparison.png"
                    )
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Saved plot to {plot_path}")
            except Exception as e:
                print(f"Error creating plot for {metric}: {e}")

        # Create a heatmap
        try:
            if len(common_metrics) > 1:
                plt.figure(figsize=(14, 10))
                sns.heatmap(summary_df, annot=True, cmap="viridis", fmt=".3f")
                plt.title(f"Average Metrics Heatmap across Experiments")
                plt.tight_layout()
                heatmap_path = os.path.join(
                    plots_dir, f"experiment_summary_heatmap.png"
                )
                plt.savefig(heatmap_path)
                plt.close()
                print(f"Saved heatmap to {heatmap_path}")
            else:
                print("Not enough metrics for a heatmap.")
        except Exception as e:
            print(f"Error creating heatmap: {e}")

        print(f"Summary plots saved to {plots_dir}")

    return comparisons


# Debug function to explore a directory structure
def explore_directory(dir_path, max_depth=3, current_depth=0):
    """Recursively explore a directory structure for debugging"""
    if current_depth > max_depth:
        return

    indent = "  " * current_depth
    print(f"{indent}Contents of {dir_path}:")

    if not os.path.exists(dir_path):
        print(f"{indent}Directory does not exist!")
        return

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            print(f"{indent}- DIR: {item}/")
            explore_directory(item_path, max_depth, current_depth + 1)
        else:
            print(f"{indent}- FILE: {item}")


def create_interactive_menu():
    """Interactive menu for the user to choose comparison options"""
    print("\n=== EVALUATION RESULTS COMPARISON TOOL ===")
    print("1. Compare within a single experiment")
    print("2. Compare across datasets (diabetes vs stroke)")
    print("3. Compare all experiments for a dataset")
    print("4. Explore directory structure")
    print("5. Exit")

    choice = input("Enter your choice (1-5): ")

    if choice == "1":
        # Compare within a single experiment
        dataset = input("Enter dataset type (diabetes/stroke/cirrhosis): ").lower()
        results_dir = input(f"Enter the path to {dataset} results directory: ")

        # Verify directory exists
        if not os.path.exists(results_dir):
            print(f"Directory {results_dir} does not exist!")
            return

        # List available experiments
        print(f"Available experiments in {results_dir}:")
        exp_dirs = [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ]

        if not exp_dirs:
            print(f"No experiment directories found in {results_dir}")
            return

        for i, exp in enumerate(exp_dirs):
            print(f"{i+1}. {exp}")

        exp_choice = input(f"Enter experiment number (1-{len(exp_dirs)}): ")
        try:
            exp_idx = int(exp_choice) - 1
            if 0 <= exp_idx < len(exp_dirs):
                compare_within_experiment(
                    results_dir, exp_dirs[exp_idx], dataset, "both"
                )
            else:
                print("Invalid experiment number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    elif choice == "2":
        # Compare across datasets
        prompt_num = input("Enter prompt number to compare: ")

        # Get directories
        diabetes_dir = input("Enter the path to diabetes results directory: ")
        stroke_dir = input("Enter the path to stroke results directory: ")

        if not os.path.exists(diabetes_dir):
            print(f"Directory {diabetes_dir} does not exist!")
            return

        if not os.path.exists(stroke_dir):
            print(f"Directory {stroke_dir} does not exist!")
            return

        results_dirs = {"Diabetes": diabetes_dir, "Stroke": stroke_dir}
        dataset_types = {"Diabetes": "diabetes", "Stroke": "stroke"}
        compare_across_datasets(results_dirs, prompt_num, dataset_types, "both")

    elif choice == "3":
        # Compare all experiments for a dataset
        dataset = input("Enter dataset type (diabetes/stroke/cirrhosis): ").lower()
        results_dir = input(f"Enter the path to {dataset} results directory: ")

        # Verify directory exists
        if not os.path.exists(results_dir):
            print(f"Directory {results_dir} does not exist!")
            return

        compare_all_experiments(results_dir, dataset)

    elif choice == "4":
        # Explore directory structure
        dir_path = input("Enter directory path to explore: ")
        max_depth = int(input("Enter max depth to explore (1-5): ") or "3")
        max_depth = min(max(1, max_depth), 5)  # Limit between 1 and 5

        if os.path.exists(dir_path):
            explore_directory(dir_path, max_depth)
        else:
            print(f"Directory {dir_path} does not exist!")

    elif choice == "5":
        print("Exiting...")
        return

    else:
        print("Invalid choice. Please try again.")


# Main block to run the program
if __name__ == "__main__":
    # Debug: Explore directory structure
    print("\n=== WELCOME TO THE EVALUATION RESULTS COMPARISON TOOL ===")
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Try to detect results directories automatically
    potential_dirs = {
        "stroke": [
            "stroke_evaluation_results",
            "evaluation_results/stroke",
            "./results/stroke",
            "./stroke_results",
        ],
        "diabetes": [
            "diabetes_evaluation_results",
            "evaluation_results/diabetes",
            "./results/diabetes",
            "./diabetes_results",
        ],
        "cirrhosis": [
            "cirrhosis_evaluation_results",
            "evaluation_results/cirrhosis",
            "./results/cirrhosis",
            "./cirrhosis_results",
        ],
    }

    found_dirs = {}
    for dataset, dirs in potential_dirs.items():
        for dir_path in dirs:
            if os.path.exists(dir_path):
                found_dirs[dataset] = dir_path
                print(f"Found {dataset} results directory: {dir_path}")
                break

    # Display menu or run specific functions
    import sys

    if len(sys.argv) > 1:
        # Command-line mode
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python compare_results.py [option]")
            print("Options:")
            print("  --help, -h          Show this help message")
            print("  --menu              Show interactive menu")
            print(
                "  --within DIR EXP    Compare within experiment EXP in directory DIR"
            )
            print("  --across P D1 D2    Compare prompt P across directories D1 and D2")
            print(
                "  --all DIR TYPE      Compare all experiments for dataset TYPE in directory DIR"
            )
            print("  --explore DIR       Explore directory structure")
        elif sys.argv[1] == "--menu":
            create_interactive_menu()
        elif sys.argv[1] == "--within" and len(sys.argv) >= 4:
            results_dir = sys.argv[2]
            experiment = sys.argv[3]
            dataset_type = sys.argv[4] if len(sys.argv) > 4 else "diabetes"
            compare_within_experiment(results_dir, experiment, dataset_type, "both")
        elif sys.argv[1] == "--across" and len(sys.argv) >= 5:
            prompt_num = sys.argv[2]
            dir1 = sys.argv[3]
            dir2 = sys.argv[4]
            results_dirs = {"Dataset1": dir1, "Dataset2": dir2}
            dataset_types = {"Dataset1": "diabetes", "Dataset2": "stroke"}
            if len(sys.argv) > 5:
                dataset_types["Dataset1"] = sys.argv[5]
            if len(sys.argv) > 6:
                dataset_types["Dataset2"] = sys.argv[6]
            compare_across_datasets(results_dirs, prompt_num, dataset_types, "both")
        elif sys.argv[1] == "--all" and len(sys.argv) >= 4:
            results_dir = sys.argv[2]
            dataset_type = sys.argv[3]
            compare_all_experiments(results_dir, dataset_type)
        elif sys.argv[1] == "--explore" and len(sys.argv) >= 3:
            dir_path = sys.argv[2]
            max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
            explore_directory(dir_path, max_depth)
        else:
            print("Invalid command-line arguments. Use --help for usage information.")
    else:
        # Interactive mode - show the menu
        while True:
            try:
                create_interactive_menu()

                # Ask if the user wants to continue
                cont = input(
                    "\nDo you want to perform another operation? (y/n): "
                ).lower()
                if cont != "y":
                    print("Exiting...")
                    break
            except KeyboardInterrupt:
                print("\nOperation interrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                cont = input(
                    "\nDo you want to continue despite the error? (y/n): "
                ).lower()
                if cont != "y":
                    print("Exiting...")
                    break

# import os
# import json
# import re
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# # Regular expression pattern to extract numpy values from text format
# np_pattern = re.compile(r'np\.float64\(([^)]+)\)')

# def extract_value(value_str):
#     """Extract numerical values from string that might contain np.float64 notation"""
#     if isinstance(value_str, str):
#         # Check if it's an np.float64 representation
#         match = np_pattern.search(value_str)
#         if match:
#             try:
#                 return float(match.group(1))
#             except ValueError:
#                 return value_str
#         # Try to convert to float directly
#         try:
#             return float(value_str)
#         except ValueError:
#             return value_str
#     return value_str

# def parse_results_file(file_path):
#     """
#     Parse evaluation results from either JSON or text format
#     """
#     if not os.path.exists(file_path):
#         print(f"File not found: {file_path}")
#         return None

#     # Determine file type
#     if file_path.endswith('.json'):
#         # Load JSON file
#         with open(file_path, 'r') as f:
#             return json.load(f)
#     elif file_path.endswith('.txt'):
#         # Parse text file
#         results = {}
#         current_section = None

#         with open(file_path, 'r') as f:
#             content = f.read()

#             # Split the content by the headers (in uppercase followed by colon)
#             sections = re.split(r'\n([A-Z_]+):\n', content)

#             # First element is the intro text, skip it
#             if not sections[0].strip().upper().endswith('RESULTS'):
#                 sections = sections[1:]

#             # Process sections in pairs (name, content)
#             for i in range(0, len(sections), 2):
#                 if i + 1 < len(sections):
#                     section_name = sections[i].strip()
#                     section_content = sections[i + 1].strip()

#                     # Try to evaluate the section content to convert it to a Python object
#                     try:
#                         # Replace np.float64 with actual values
#                         section_content = np_pattern.sub(r'\1', section_content)
#                         # Convert 'nan' to actual NaN
#                         section_content = section_content.replace('nan', 'float("nan")')
#                         # Try to evaluate
#                         section_data = eval(section_content)
#                         results[section_name] = section_data
#                     except:
#                         # If evaluation fails, just store the raw text
#                         results[section_name] = section_content

#         return results
#     else:
#         print(f"Unsupported file type: {file_path}")
#         return None

# def extract_key_metrics(results, dataset_type='diabetes'):
#     """
#     Extract key metrics from evaluation results for comparison

#     Parameters:
#     -----------
#     results : dict
#         Parsed evaluation results
#     dataset_type : str
#         Type of dataset (diabetes, stroke, or cirrhosis)

#     Returns:
#     --------
#     dict
#         Dictionary of key metrics
#     """
#     metrics = {}

#     # Check if results is None
#     if results is None:
#         return {"error": "No results data available"}

#     # Common metrics regardless of dataset type
#     try:
#         # Get correlation matrix distance (synthetic vs real)
#         if 'feature_correlations' in results:
#             if isinstance(results['feature_correlations'], dict) and 'correlation_matrix_distance' in results['feature_correlations']:
#                 metrics['correlation_distance'] = extract_value(results['feature_correlations']['correlation_matrix_distance'])

#         # Get privacy risk
#         if 'privacy_assessment' in results:
#             privacy_metrics = results['privacy_assessment']
#             violation_rates = []
#             for key, value in privacy_metrics.items():
#                 if isinstance(value, dict) and 'violation_rate' in value:
#                     violation_rates.append(extract_value(value['violation_rate']))
#             if violation_rates:
#                 metrics['avg_privacy_violation'] = np.mean(violation_rates)

#         # Extract distribution metrics (average across numerical features)
#         if 'distribution_metrics' in results:
#             dist_metrics = results['distribution_metrics']
#             wasserstein_values = []
#             jensen_shannon_values = []

#             for feature, values in dist_metrics.items():
#                 if isinstance(values, dict):
#                     if 'wasserstein' in values:
#                         wasserstein_values.append(extract_value(values['wasserstein']))
#                     if 'jensen_shannon' in values:
#                         jensen_shannon_values.append(extract_value(values['jensen_shannon']))

#             if wasserstein_values:
#                 metrics['avg_wasserstein'] = np.mean([x for x in wasserstein_values if not np.isnan(x)])
#             if jensen_shannon_values:
#                 metrics['avg_jensen_shannon'] = np.mean([x for x in jensen_shannon_values if not np.isnan(x)])

#         # Extract numerical statistics
#         if 'numerical_statistics' in results:
#             num_stats = results['numerical_statistics']
#             mean_errors = []
#             std_diffs = []

#             for feature, values in num_stats.items():
#                 if isinstance(values, dict):
#                     if 'relative_mean_error' in values:
#                         mean_errors.append(extract_value(values['relative_mean_error']))
#                     if 'std_difference' in values:
#                         std_diffs.append(extract_value(values['std_difference']))

#             if mean_errors:
#                 metrics['avg_relative_mean_error'] = np.mean([x for x in mean_errors if not np.isnan(x)])
#             if std_diffs:
#                 metrics['avg_std_difference'] = np.mean([x for x in std_diffs if not np.isnan(x)])
#     except Exception as e:
#         metrics['error'] = f"Error extracting common metrics: {str(e)}"

#     # Dataset-specific metrics
#     try:
#         if dataset_type == 'diabetes':
#             # Extract diabetes-specific metrics
#             if 'medical_consistency' in results:
#                 med_consistency = results['medical_consistency']
#                 if isinstance(med_consistency, dict):
#                     if 'hba1c_diabetes_relationship' in med_consistency:
#                         hba1c_rel = med_consistency['hba1c_diabetes_relationship']
#                         if isinstance(hba1c_rel, dict) and 'real_difference' in hba1c_rel and 'synthetic_difference' in hba1c_rel:
#                             real_diff = extract_value(hba1c_rel['real_difference'])
#                             synth_diff = extract_value(hba1c_rel['synthetic_difference'])
#                             metrics['hba1c_diff_preservation'] = 1 - abs(real_diff - synth_diff) / real_diff

#                     if 'glucose_diabetes_relationship' in med_consistency:
#                         glucose_rel = med_consistency['glucose_diabetes_relationship']
#                         if isinstance(glucose_rel, dict) and 'real_difference' in glucose_rel and 'synthetic_difference' in glucose_rel:
#                             real_diff = extract_value(glucose_rel['real_difference'])
#                             synth_diff = extract_value(glucose_rel['synthetic_difference'])
#                             metrics['glucose_diff_preservation'] = 1 - abs(real_diff - synth_diff) / real_diff

#         elif dataset_type == 'stroke':
#             # Extract stroke-specific metrics
#             if 'medical_consistency' in results:
#                 med_consistency = results['medical_consistency']
#                 if isinstance(med_consistency, dict):
#                     if 'glucose_stroke_relationship' in med_consistency:
#                         glucose_rel = med_consistency['glucose_stroke_relationship']
#                         if isinstance(glucose_rel, dict) and 'difference_in_difference' in glucose_rel:
#                             metrics['glucose_stroke_diff'] = extract_value(glucose_rel['difference_in_difference'])

#                     if 'age_stroke_relationship' in med_consistency:
#                         age_rel = med_consistency['age_stroke_relationship']
#                         if isinstance(age_rel, dict) and 'difference_in_difference' in age_rel:
#                             metrics['age_stroke_diff'] = extract_value(age_rel['difference_in_difference'])

#             # Extract stroke prediction metrics
#             if 'advanced_metrics' in results and isinstance(results['advanced_metrics'], dict):
#                 adv_metrics = results['advanced_metrics']
#                 if 'stroke_prediction_metrics' in adv_metrics and isinstance(adv_metrics['stroke_prediction_metrics'], dict):
#                     stroke_pred = adv_metrics['stroke_prediction_metrics']
#                     if 'risk_factor_correlations' in stroke_pred and isinstance(stroke_pred['risk_factor_correlations'], dict):
#                         risk_corr = stroke_pred['risk_factor_correlations']
#                         if 'absolute_differences' in risk_corr and isinstance(risk_corr['absolute_differences'], dict):
#                             abs_diffs = risk_corr['absolute_differences']
#                             metrics['avg_risk_factor_diff'] = np.mean([extract_value(v) for v in abs_diffs.values() if not np.isnan(extract_value(v))])

#         elif dataset_type == 'cirrhosis':
#             # Extract cirrhosis-specific metrics
#             if 'medical_consistency' in results:
#                 med_consistency = results['medical_consistency']
#                 if isinstance(med_consistency, dict):
#                     if 'bilirubin_stage_relationship' in med_consistency:
#                         bili_rel = med_consistency['bilirubin_stage_relationship']
#                         if isinstance(bili_rel, dict):
#                             metrics['bilirubin_stage_preservation'] = 1  # Placeholder

#                     if 'albumin_stage_relationship' in med_consistency:
#                         alb_rel = med_consistency['albumin_stage_relationship']
#                         if isinstance(alb_rel, dict):
#                             metrics['albumin_stage_preservation'] = 1  # Placeholder
#     except Exception as e:
#         metrics['error_specific'] = f"Error extracting {dataset_type} metrics: {str(e)}"

#     return metrics

# def compare_within_experiment(results_dir, experiment_name, dataset_type='diabetes', output_format='dataframe'):
#     """
#     Compare results across different PROMPTs within the same experiment

#     Parameters:
#     -----------
#     results_dir : str
#         Directory containing the evaluation results
#     experiment_name : str
#         Name of the experiment to analyze
#     dataset_type : str
#         Type of dataset (diabetes, stroke, or cirrhosis)
#     output_format : str
#         Format of the output ('dataframe', 'plot', or 'both')

#     Returns:
#     --------
#     pd.DataFrame or None
#         DataFrame with comparison results, or None if output_format is 'plot'
#     """
#     # Create full path to experiment directory
#     exp_dir = os.path.join(results_dir, experiment_name)

#     if not os.path.exists(exp_dir):
#         print(f"Experiment directory not found: {exp_dir}")
#         return None

#     # Find all result files
#     result_files = []
#     for file in os.listdir(exp_dir):
#         if file.startswith('results_PROMPT_') and (file.endswith('.json') or file.endswith('.txt')):
#             result_files.append(file)

#     if not result_files:
#         print(f"No result files found in {exp_dir}")
#         return None

#     # Sort result files by PROMPT number
#     result_files.sort(key=lambda x: int(re.search(r'PROMPT_(\d+)', x).group(1)))

#     # Parse each result file and extract key metrics
#     all_metrics = {}
#     for file in result_files:
#         PROMPT_num = re.search(r'PROMPT_(\d+)', file).group(1)
#         file_path = os.path.join(exp_dir, file)
#         results = parse_results_file(file_path)
#         metrics = extract_key_metrics(results, dataset_type)
#         all_metrics[f"PROMPT {PROMPT_num}"] = metrics

#     # Convert to DataFrame
#     df = pd.DataFrame(all_metrics).T

#     # Handle output format
#     if output_format in ['plot', 'both']:
#         # Create directory for plots
#         plots_dir = os.path.join(results_dir, 'plots')
#         os.makedirs(plots_dir, exist_ok=True)

#         # Create a plot for each metric
#         for metric in df.columns:
#             plt.figure(figsize=(10, 6))
#             sns.barplot(x=df.index, y=metric, data=df)
#             plt.title(f"{metric} across PROMPTs in {experiment_name}")
#             plt.ylabel(metric)
#             plt.tight_layout()
#             plot_path = os.path.join(plots_dir, f"{experiment_name}_{metric}_comparison.png")
#             plt.savefig(plot_path)
#             plt.close()

#         # Create a heatmap of all metrics
#         plt.figure(figsize=(12, 8))
#         sns.heatmap(df, annot=True, cmap='viridis', fmt='.3f')
#         plt.title(f"Metrics Heatmap for {experiment_name}")
#         plt.tight_layout()
#         heatmap_path = os.path.join(plots_dir, f"{experiment_name}_metrics_heatmap.png")
#         plt.savefig(heatmap_path)
#         plt.close()

#         print(f"Plots saved to {plots_dir}")

#     if output_format in ['dataframe', 'both']:
#         return df
#     return None

# def compare_across_datasets(results_dirs, PROMPT_num, dataset_types, output_format='dataframe'):
#     """
#     Compare results for the same PROMPT number across different datasets

#     Parameters:
#     -----------
#     results_dirs : dict
#         Dictionary mapping dataset names to their result directories
#     PROMPT_num : int or str
#         PROMPT number to compare
#     dataset_types : dict
#         Dictionary mapping dataset names to their types (diabetes, stroke, or cirrhosis)
#     output_format : str
#         Format of the output ('dataframe', 'plot', or 'both')

#     Returns:
#     --------
#     pd.DataFrame or None
#         DataFrame with comparison results, or None if output_format is 'plot'
#     """
#     # Normalize PROMPT_num to string
#     PROMPT_num = str(PROMPT_num)

#     # Find and parse result files for each dataset
#     all_metrics = {}
#     for dataset_name, results_dir in results_dirs.items():
#         # Find all experiment directories
#         exp_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

#         for exp_dir in exp_dirs:
#             # Look for the specific PROMPT file
#             exp_path = os.path.join(results_dir, exp_dir)
#             result_file = None

#             for file in os.listdir(exp_path):
#                 if f'PROMPT_{PROMPT_num}' in file.lower() and (file.endswith('.json') or file.endswith('.txt')):
#                     result_file = file
#                     break

#             if result_file:
#                 file_path = os.path.join(exp_path, result_file)
#                 results = parse_results_file(file_path)
#                 metrics = extract_key_metrics(results, dataset_types.get(dataset_name, 'unknown'))
#                 all_metrics[f"{dataset_name} ({exp_dir})"] = metrics

#     if not all_metrics:
#         print(f"No matching PROMPT files found for PROMPT {PROMPT_num}")
#         return None

#     # Convert to DataFrame
#     df = pd.DataFrame(all_metrics).T

#     # Handle output format
#     if output_format in ['plot', 'both']:
#         # Create directory for plots
#         plots_dir = 'cross_dataset_plots'
#         os.makedirs(plots_dir, exist_ok=True)

#         # Create a plot for each metric
#         common_metrics = [col for col in df.columns if not col.startswith('error')]

#         for metric in common_metrics:
#             if metric in df.columns:
#                 plt.figure(figsize=(12, 7))
#                 sns.barplot(x=df.index, y=metric, data=df)
#                 plt.title(f"{metric} for PROMPT {PROMPT_num} across Datasets")
#                 plt.ylabel(metric)
#                 plt.xticks(rotation=45, ha='right')
#                 plt.tight_layout()
#                 plot_path = os.path.join(plots_dir, f"PROMPT_{PROMPT_num}_{metric}_cross_dataset.png")
#                 plt.savefig(plot_path)
#                 plt.close()

#         # Create a heatmap for common metrics only
#         common_df = df[common_metrics].copy()
#         if not common_df.empty and len(common_metrics) > 1:
#             plt.figure(figsize=(14, 10))
#             sns.heatmap(common_df, annot=True, cmap='viridis', fmt='.3f')
#             plt.title(f"Metrics Heatmap for PROMPT {PROMPT_num} across Datasets")
#             plt.tight_layout()
#             heatmap_path = os.path.join(plots_dir, f"PROMPT_{PROMPT_num}_cross_dataset_heatmap.png")
#             plt.savefig(heatmap_path)
#             plt.close()

#         print(f"Cross-dataset plots saved to {plots_dir}")

#     if output_format in ['dataframe', 'both']:
#         return df
#     return None

# def compare_all_experiments(results_dir, dataset_type='diabetes', output_dir=None):
#     """
#     Compare results across all experiments for a given dataset

#     Parameters:
#     -----------
#     results_dir : str
#         Directory containing the evaluation results
#     dataset_type : str
#         Type of dataset (diabetes, stroke, or cirrhosis)
#     output_dir : str, optional
#         Directory to save the output, defaults to 'experiment_comparisons'

#     Returns:
#     --------
#     dict
#         Dictionary mapping experiment names to their comparison DataFrames
#     """
#     if output_dir is None:
#         output_dir = 'experiment_comparisons'

#     os.makedirs(output_dir, exist_ok=True)

#     # Find all experiment directories
#     exp_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

#     if not exp_dirs:
#         print(f"No experiment directories found in {results_dir}")
#         return {}

#     # Compare within each experiment
#     comparisons = {}
#     for exp_dir in exp_dirs:
#         print(f"Comparing within experiment: {exp_dir}")
#         df = compare_within_experiment(results_dir, exp_dir, dataset_type, 'dataframe')
#         if df is not None:
#             comparisons[exp_dir] = df
#             # Save the DataFrame to CSV
#             csv_path = os.path.join(output_dir, f"{exp_dir}_comparison.csv")
#             df.to_csv(csv_path)
#             print(f"Saved comparison to {csv_path}")

#     # Create a summary table with average metrics for each experiment
#     if comparisons:
#         summary_data = {}
#         common_metrics = set()

#         # Find common metrics across all experiments
#         for exp_name, df in comparisons.items():
#             if not common_metrics:
#                 common_metrics = set(df.columns)
#             else:
#                 common_metrics = common_metrics.intersection(set(df.columns))

#         # Calculate average metrics for each experiment
#         for exp_name, df in comparisons.items():
#             avg_metrics = {}
#             for metric in common_metrics:
#                 avg_metrics[metric] = df[metric].mean()
#             summary_data[exp_name] = avg_metrics

#         summary_df = pd.DataFrame(summary_data).T
#         summary_path = os.path.join(output_dir, 'experiment_summary.csv')
#         summary_df.to_csv(summary_path)
#         print(f"Saved experiment summary to {summary_path}")

#         # Create summary plots
#         plots_dir = os.path.join(output_dir, 'summary_plots')
#         os.makedirs(plots_dir, exist_ok=True)

#         for metric in common_metrics:
#             plt.figure(figsize=(12, 7))
#             sns.barplot(x=summary_df.index, y=metric, data=summary_df)
#             plt.title(f"Average {metric} across Experiments")
#             plt.ylabel(metric)
#             plt.xticks(rotation=45, ha='right')
#             plt.tight_layout()
#             plot_path = os.path.join(plots_dir, f"avg_{metric}_experiment_comparison.png")
#             plt.savefig(plot_path)
#             plt.close()

#         # Create a heatmap
#         plt.figure(figsize=(14, 10))
#         sns.heatmap(summary_df, annot=True, cmap='viridis', fmt='.3f')
#         plt.title(f"Average Metrics Heatmap across Experiments")
#         plt.tight_layout()
#         heatmap_path = os.path.join(plots_dir, f"experiment_summary_heatmap.png")
#         plt.savefig(heatmap_path)
#         plt.close()

#         print(f"Summary plots saved to {plots_dir}")

#     return comparisons

# # Example usage
# if __name__ == "__main__":
#     # Example 1: Compare within a single experiment
#     # compare_within_experiment('evaluation_results', 'experiment_1', 'diabetes', 'both')
#     compare_within_experiment('stroke_evaluation_results', 'experiment_1', 'stroke', 'both')

#     # Example 2: Compare across datasets
#     # results_dirs = {
#     #     'Diabetes': 'diabetes_results',
#     #     'Stroke': 'stroke_results'
#     # }
#     # dataset_types = {
#     #     'Diabetes': 'diabetes',
#     #     'Stroke': 'stroke'
#     # }
#     # compare_across_datasets(results_dirs, 1, dataset_types, 'both')

#     # Example 3: Compare all experiments for a dataset
#     # compare_all_experiments('diabetes_results', 'diabetes')

#     # Set your actual directories and configurations here
#     # ...

#     # DIABETES EVALUATION
#     # Assuming your diabetes results are in 'diabetes_evaluation_results' directory
#     print("\n=== Comparing Diabetes Experiments ===")
#     diabetes_comparisons = compare_all_experiments('diabetes_evaluation_results', 'diabetes')

#     # STROKE EVALUATION
#     # Assuming your stroke results are in 'stroke_evaluation_results' directory
#     print("\n=== Comparing Stroke Experiments ===")
#     stroke_comparisons = compare_all_experiments('stroke_evaluation_results', 'stroke')

#     # CROSS-DATASET COMPARISON
#     # Compare PROMPT 1 across datasets
#     print("\n=== Comparing PROMPT 1 Across Datasets ===")
#     results_dirs = {
#         'Diabetes': 'diabetes_evaluation_results',
#         'Stroke': 'stroke_evaluation_results'
#     }
#     dataset_types = {
#         'Diabetes': 'diabetes',
#         'Stroke': 'stroke'
#     }
#     cross_comparison = compare_across_datasets(results_dirs, 1, dataset_types, 'both')

#     if cross_comparison is not None:
#         cross_comparison.to_csv('cross_dataset_comparison.csv')
#         print("Cross-dataset comparison saved to cross_dataset_comparison.csv")
