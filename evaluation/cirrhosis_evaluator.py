import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import (
    anderson_ksamp,
    chi2_contingency,
    entropy,
    ks_2samp,
    wasserstein_distance,
)
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


class CirrhosisDataEvaluator:
    def __init__(self, real_data, synthetic_data):
        """
        Initialize with real and synthetic datasets

        Parameters:
        real_data (pd.DataFrame): Original cirrhosis dataset
        synthetic_data (pd.DataFrame): Generated synthetic cirrhosis dataset
        """
        # Make copies to avoid modifying original data
        self.real_data = real_data.copy()
        # Sample real data to match synthetic data size
        if len(self.real_data) > len(synthetic_data):
            self.real_data = self.real_data.sample(
                n=len(synthetic_data), random_state=42
            )
        self.synthetic_data = synthetic_data.copy()

        # Define column types for cirrhosis data
        self.numerical_cols = [
            "Age",
            "Bilirubin",
            "Cholesterol",
            "Albumin",
            "Copper",
            "Alk_Phos",
            "SGOT",
            "Tryglicerides",
            "Platelets",
            "Prothrombin",
            "N_Days",
        ]
        self.categorical_cols = ["Sex", "Drug"]
        self.binary_cols = ["Ascites", "Hepatomegaly", "Spiders", "Edema"]
        self.ordinal_cols = ["Stage"]  # This is an ordinal variable (1-4)
        self.status_col = "Status"  # Special column for patient status (C, CL, D)

        # Clean data
        self._clean_data()

        # Initialize encoders
        self.encoders = {}
        self._prepare_encoders()

    def _clean_data(self):
        """Clean and prepare the cirrhosis data"""
        # Remove any rows that contain prompts or instructions
        for df in [self.real_data, self.synthetic_data]:
            for col in df.columns:
                mask = (
                    df[col]
                    .astype(str)
                    .str.contains(
                        "generate|Generate|GENERATE|format|Format|FORMAT",
                        case=False,
                        na=False,
                    )
                )
                df.drop(df[mask].index, inplace=True)

        # Convert binary columns to proper format
        for col in self.binary_cols:
            # Y/N to 1/0
            for df in [self.real_data, self.synthetic_data]:
                df[col] = df[col].map(
                    {"Y": 1, "N": 0, "S": 0.5}
                )  # S is "slight" -> 0.5

        # Convert numerical columns to float
        for col in self.numerical_cols:
            self.real_data[col] = pd.to_numeric(self.real_data[col], errors="coerce")
            self.synthetic_data[col] = pd.to_numeric(
                self.synthetic_data[col], errors="coerce"
            )

        # Convert Stage to numeric
        for df in [self.real_data, self.synthetic_data]:
            df["Stage"] = pd.to_numeric(df["Stage"], errors="coerce")

        # Handle missing values
        self.real_data.fillna(self.real_data.mean(numeric_only=True), inplace=True)
        self.synthetic_data.fillna(
            self.synthetic_data.mean(numeric_only=True), inplace=True
        )

    def _prepare_encoders(self):
        """Prepare label encoders for categorical variables"""
        for col in self.categorical_cols + [self.status_col]:
            # Combine unique values from both datasets
            all_categories = set(self.real_data[col].astype(str).unique()) | set(
                self.synthetic_data[col].astype(str).unique()
            )

            # Initialize encoder
            le = LabelEncoder()
            # Fit with all possible categories
            le.fit(list(all_categories))
            self.encoders[col] = le

            # Transform the data
            self.real_data[f"{col}_encoded"] = le.transform(
                self.real_data[col].astype(str)
            )
            self.synthetic_data[f"{col}_encoded"] = le.transform(
                self.synthetic_data[col].astype(str)
            )

    def calculate_numerical_statistics(self):
        """Calculate statistical similarities for numerical variables"""
        stats = {}
        for col in self.numerical_cols + ["Stage"]:
            try:
                real_mean = self.real_data[col].mean()
                synth_mean = self.synthetic_data[col].mean()
                real_std = self.real_data[col].std()
                synth_std = self.synthetic_data[col].std()

                stats[col] = {
                    "mean_difference": abs(real_mean - synth_mean),
                    "std_difference": abs(real_std - synth_std),
                    "relative_mean_error": (
                        abs(real_mean - synth_mean) / real_mean if real_mean != 0 else 0
                    ),
                    "real_quartiles": self.real_data[col]
                    .quantile([0.25, 0.5, 0.75])
                    .tolist(),
                    "synthetic_quartiles": self.synthetic_data[col]
                    .quantile([0.25, 0.5, 0.75])
                    .tolist(),
                }
            except Exception as e:
                stats[col] = f"Error calculating statistics: {str(e)}"
        return stats

    def calculate_categorical_similarity(self):
        """Compare distributions of categorical variables"""
        similarity = {}
        for col in self.categorical_cols + self.binary_cols + [self.status_col]:
            try:
                real_dist = self.real_data[col].value_counts(normalize=True)
                synth_dist = self.synthetic_data[col].value_counts(normalize=True)

                # Align distributions
                all_categories = sorted(set(real_dist.index) | set(synth_dist.index))
                real_dist = real_dist.reindex(all_categories, fill_value=0)
                synth_dist = synth_dist.reindex(all_categories, fill_value=0)

                # Calculate chi-square test if possible
                try:
                    chi2, p_value = chi2_contingency(
                        pd.DataFrame({"real": real_dist, "synthetic": synth_dist}).T
                    )[:2]
                except:
                    chi2, p_value = np.nan, np.nan

                similarity[col] = {
                    "chi2_statistic": chi2,
                    "p_value": p_value,
                    "distribution_difference": dict(abs(real_dist - synth_dist)),
                }
            except Exception as e:
                similarity[col] = f"Error calculating similarity: {str(e)}"
        return similarity

    def calculate_feature_correlations(self):
        """Compare correlation matrices between real and synthetic data"""
        try:
            # Include all numeric columns (numerical_cols + binary_cols converted to numeric)
            numeric_cols = self.numerical_cols + self.binary_cols + ["Stage"]

            real_corr = self.real_data[numeric_cols].corr()
            synth_corr = self.synthetic_data[numeric_cols].corr()

            correlation_distance = np.linalg.norm(real_corr - synth_corr)

            return {
                "correlation_matrix_distance": correlation_distance,
                "real_correlations": real_corr.to_dict(),
                "synthetic_correlations": synth_corr.to_dict(),
            }
        except Exception as e:
            return f"Error calculating correlations: {str(e)}"

    def calculate_medical_consistency(self):
        """Check medical consistency of the synthetic data for cirrhosis"""
        consistency_metrics = {}

        try:
            # Check Bilirubin and Stage relationship
            real_bilirubin_by_stage = self.real_data.groupby("Stage")[
                "Bilirubin"
            ].mean()
            synth_bilirubin_by_stage = self.synthetic_data.groupby("Stage")[
                "Bilirubin"
            ].mean()

            consistency_metrics["bilirubin_stage_relationship"] = {
                "real_means": real_bilirubin_by_stage.to_dict(),
                "synthetic_means": synth_bilirubin_by_stage.to_dict(),
            }

            # Check Albumin and Stage relationship (Albumin generally decreases with disease progression)
            real_albumin_by_stage = self.real_data.groupby("Stage")["Albumin"].mean()
            synth_albumin_by_stage = self.synthetic_data.groupby("Stage")[
                "Albumin"
            ].mean()

            consistency_metrics["albumin_stage_relationship"] = {
                "real_means": real_albumin_by_stage.to_dict(),
                "synthetic_means": synth_albumin_by_stage.to_dict(),
            }

            # Check Survival status and Bilirubin relationship
            real_bilirubin_by_status = self.real_data.groupby(self.status_col)[
                "Bilirubin"
            ].mean()
            synth_bilirubin_by_status = self.synthetic_data.groupby(self.status_col)[
                "Bilirubin"
            ].mean()

            consistency_metrics["bilirubin_status_relationship"] = {
                "real_means": real_bilirubin_by_status.to_dict(),
                "synthetic_means": synth_bilirubin_by_status.to_dict(),
            }

        except Exception as e:
            consistency_metrics["relationship_error"] = str(e)

        return consistency_metrics

    def privacy_risk_assessment(self, k=5):
        """Assess privacy risks using k-anonymity for sensitive columns"""
        try:
            identifying_cols = ["Age", "Sex", "Bilirubin"]

            def check_k_anonymity(df, cols):
                return df.groupby(cols).size().reset_index(name="count")

            privacy_metrics = {}
            for i in range(1, len(identifying_cols) + 1):
                for cols in itertools.combinations(identifying_cols, i):
                    cols = list(cols)  # Convert to list for indexing
                    real_counts = check_k_anonymity(self.real_data, cols)
                    synth_counts = check_k_anonymity(self.synthetic_data, cols)

                    privacy_metrics[f"{'_'.join(cols)}_violations"] = {
                        "records_below_k": len(synth_counts[synth_counts["count"] < k]),
                        "violation_rate": len(synth_counts[synth_counts["count"] < k])
                        / len(self.synthetic_data),
                    }

            return privacy_metrics
        except Exception as e:
            return f"Error in privacy assessment: {str(e)}"

    def calculate_distribution_metrics(self):
        """Calculate distribution similarity metrics for numerical columns"""
        metrics = {}

        for col in self.numerical_cols:
            try:
                # Skip if either dataset has all NaN values for this column
                if (
                    self.real_data[col].isna().all()
                    or self.synthetic_data[col].isna().all()
                ):
                    metrics[col] = {
                        "wasserstein": np.nan,
                        "jensen_shannon": np.nan,
                        "kl_divergence": np.nan,
                        "error": "All values are NaN",
                    }
                    continue

                # Normalize the data for fair comparison
                real_min = self.real_data[col].min()
                real_max = self.real_data[col].max()

                # Check for zero range
                if real_max == real_min:
                    metrics[col] = {
                        "wasserstein": 0.0,
                        "jensen_shannon": 0.0,
                        "kl_divergence": 0.0,
                        "note": "All real values are identical",
                    }
                    continue

                real_norm = (self.real_data[col] - real_min) / (real_max - real_min)

                synth_min = self.synthetic_data[col].min()
                synth_max = self.synthetic_data[col].max()

                # Check for zero range
                if synth_max == synth_min:
                    metrics[col] = {
                        "wasserstein": 1.0,
                        "jensen_shannon": 1.0,
                        "kl_divergence": np.inf,
                        "note": "All synthetic values are identical",
                    }
                    continue

                synth_norm = (self.synthetic_data[col] - synth_min) / (
                    synth_max - synth_min
                )

                # Calculate histograms for KL and JS divergence
                hist_real, bins = np.histogram(
                    real_norm.dropna(), bins=30, density=True
                )
                hist_synth, _ = np.histogram(
                    synth_norm.dropna(), bins=bins, density=True
                )

                # Add small constant to avoid division by zero
                hist_real = hist_real + 1e-10
                hist_synth = hist_synth + 1e-10

                # Normalize histograms
                hist_real = hist_real / hist_real.sum()
                hist_synth = hist_synth / hist_synth.sum()

                metrics[col] = {
                    "wasserstein": wasserstein_distance(
                        real_norm.dropna(), synth_norm.dropna()
                    ),
                    "jensen_shannon": jensenshannon(hist_real, hist_synth),
                    "kl_divergence": entropy(hist_real, hist_synth),
                }
            except Exception as e:
                metrics[col] = {
                    "wasserstein": np.nan,
                    "jensen_shannon": np.nan,
                    "kl_divergence": np.nan,
                    "error": str(e),
                }

        return metrics

    def calculate_advanced_metrics(self):
        """Calculate additional advanced metrics"""
        metrics = {
            "numerical_metrics": self._analyze_numerical_features(),
            "categorical_metrics": self._analyze_categorical_features(),
            "privacy_metrics": self._calculate_privacy_metrics(),
            "statistical_tests": self._perform_statistical_tests(),
        }
        return metrics

    def _analyze_numerical_features(self):
        results = {}

        for col in self.numerical_cols:
            try:
                real_data = self.real_data[col].dropna()
                synth_data = self.synthetic_data[col].dropna()

                if len(real_data) == 0 or len(synth_data) == 0:
                    results[col] = {"error": "Not enough non-NA values for analysis"}
                    continue

                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = ks_2samp(real_data, synth_data)

                # Quartile comparison
                real_quartiles = np.percentile(real_data, [25, 50, 75])
                synth_quartiles = np.percentile(synth_data, [25, 50, 75])

                results[col] = {
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pval,
                    "quartile_differences": np.abs(
                        real_quartiles - synth_quartiles
                    ).mean(),
                    "std_difference": abs(real_data.std() - synth_data.std()),
                    "range_coverage": (
                        min(synth_data.max(), real_data.max())
                        - max(synth_data.min(), real_data.min())
                    )
                    / (real_data.max() - real_data.min())
                    if real_data.max() != real_data.min()
                    else 0,
                }
            except Exception as e:
                results[col] = {"error": str(e)}

        return results

    def _analyze_categorical_features(self):
        results = {}

        for col in self.categorical_cols + [self.status_col]:
            try:
                real_dist = self.real_data[col].value_counts(normalize=True)
                synth_dist = self.synthetic_data[col].value_counts(normalize=True)

                # Category preservation
                shared_categories = set(real_dist.index) & set(synth_dist.index)

                results[col] = {
                    "category_preservation": len(shared_categories)
                    / len(set(real_dist.index))
                    if len(set(real_dist.index)) > 0
                    else 0,
                    "distribution_difference": np.mean(
                        abs(real_dist - synth_dist.reindex(real_dist.index).fillna(0))
                    ),
                    "mutual_information": self._calculate_mutual_info(col),
                }
            except Exception as e:
                results[col] = {"error": str(e)}

        return results

    def _calculate_privacy_metrics(self):
        """Calculate privacy-related metrics"""
        try:
            # Nearest neighbor distance ratio
            def calculate_nn_distance(data):
                # Handle potential NaNs
                data = data.dropna(axis=1, how="all")
                data = data.fillna(data.mean())

                if data.empty or len(data) < 2:
                    return np.array([np.nan])

                nbrs = NearestNeighbors(n_neighbors=2).fit(data)
                distances, _ = nbrs.kneighbors(data)
                return distances[:, 1]  # Distance to nearest neighbor

            # Sample subset of data for computational efficiency
            sample_size = min(1000, len(self.real_data))
            real_sample = self.real_data.select_dtypes(include=[np.number]).sample(
                sample_size, replace=True
            )
            synth_sample = self.synthetic_data.select_dtypes(
                include=[np.number]
            ).sample(sample_size, replace=True)

            real_nn_dist = calculate_nn_distance(real_sample)
            synth_nn_dist = calculate_nn_distance(synth_sample)

            return {
                "nn_distance_ratio": (
                    np.mean(synth_nn_dist) / np.mean(real_nn_dist)
                    if not np.isnan(np.mean(real_nn_dist))
                    and np.mean(real_nn_dist) != 0
                    else np.nan
                ),
                "identifiability_score": len(
                    set(self.synthetic_data.astype(str).apply(tuple, axis=1))
                    & set(self.real_data.astype(str).apply(tuple, axis=1))
                )
                / len(self.synthetic_data)
                if len(self.synthetic_data) > 0
                else 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def _perform_statistical_tests(self):
        """Perform statistical tests for similarity"""
        results = {}

        for col in self.numerical_cols:
            try:
                # Filter out NaN values
                real_vals = self.real_data[col].dropna().values
                synth_vals = self.synthetic_data[col].dropna().values

                if len(real_vals) < 2 or len(synth_vals) < 2:
                    results[col] = {
                        "anderson_darling_statistic": None,
                        "anderson_darling_pvalue": None,
                        "error": "Not enough non-NA values for test",
                    }
                    continue

                # Anderson-Darling test for similarity of distributions
                ad_stat = anderson_ksamp([real_vals, synth_vals])
                results[col] = {
                    "anderson_darling_statistic": ad_stat.statistic,
                    "anderson_darling_pvalue": (
                        ad_stat.pvalue if hasattr(ad_stat, "pvalue") else None
                    ),
                }
            except Exception as e:
                results[col] = {
                    "anderson_darling_statistic": None,
                    "anderson_darling_pvalue": None,
                    "error": str(e),
                }
        return results

    def _calculate_mutual_info(self, column):
        """Calculate mutual information between real and synthetic data for a column"""
        try:
            le = LabelEncoder()
            # Convert to string to handle potential numeric values in categorical columns
            real_col = self.real_data[column].astype(str)
            synth_col = self.synthetic_data[column].astype(str)

            # Get all unique values
            all_values = np.union1d(real_col.unique(), synth_col.unique())
            le.fit(all_values)

            real_encoded = le.transform(real_col)
            synth_encoded = le.transform(synth_col)

            return mutual_info_score(real_encoded, synth_encoded)
        except Exception as e:
            return np.nan

    def evaluate_all(self):
        """Run all evaluations"""
        return {
            "numerical_statistics": self.calculate_numerical_statistics(),
            "categorical_similarity": self.calculate_categorical_similarity(),
            "feature_correlations": self.calculate_feature_correlations(),
            "medical_consistency": self.calculate_medical_consistency(),
            "distribution_metrics": self.calculate_distribution_metrics(),
            "advanced_metrics": self.calculate_advanced_metrics(),
            "privacy_assessment": self.privacy_risk_assessment(),
        }


def main():
    try:
        # Load the original cirrhosis dataset
        print("Loading original cirrhosis dataset...")
        real_data = pd.read_csv("cirrhosis.csv")
        print(f"Real data shape: {real_data.shape}")

        # Define synthetic data files
        synthetic_files = [
            "synthetic_cirrhosis_1.csv",
            "synthetic_cirrhosis_2.csv",
            # Add more synthetic data files as needed
        ]

        # Compare each synthetic dataset with the original
        for synthetic_file in synthetic_files:
            try:
                print(f"\n=== Evaluating {synthetic_file} ===")

                # Load synthetic data
                synthetic_data = pd.read_csv(synthetic_file)
                print(f"Synthetic data shape: {synthetic_data.shape}")

                # Initialize evaluator for this comparison
                print("Initializing evaluator...")
                evaluator = CirrhosisDataEvaluator(real_data, synthetic_data)

                # Run evaluation
                print("Running evaluation...")
                evaluation_results = evaluator.evaluate_all()

                # Print results for this file
                print(f"\nEvaluation Results for {synthetic_file}:")
                for metric, results in evaluation_results.items():
                    print(f"\n{metric.upper()}:")
                    print(results)

                # Optional: Save results to file
                output_file = (
                    f"evaluation_results_{synthetic_file.replace('.csv', '.txt')}"
                )
                with open(output_file, "w") as f:
                    f.write(f"Evaluation Results for {synthetic_file}:\n")
                    for metric, results in evaluation_results.items():
                        f.write(f"\n{metric.upper()}:\n")
                        f.write(str(results))
                        f.write("\n")

            except Exception as e:
                print(f"Error processing {synthetic_file}: {str(e)}")
                continue

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
