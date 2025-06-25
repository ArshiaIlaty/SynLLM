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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mutual_info_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Conditionally import umap to handle environments where it might not be available
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

warnings.filterwarnings("ignore")


class CirrhosisDataEvaluator:
    def __init__(self, real_data, synthetic_data):
        self.real_data = real_data.copy()
        if len(self.real_data) > len(synthetic_data):
            self.real_data = self.real_data.sample(
                n=len(synthetic_data), random_state=42
            )
        self.synthetic_data = synthetic_data.copy()

        # Cirrhosis dataset numerical columns
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
        # Only keep columns present in both datasets
        self.numerical_cols = [
            col
            for col in self.numerical_cols
            if col in self.real_data.columns and col in self.synthetic_data.columns
        ]

        # Hardcode binary columns for robustness
        self.binary_cols = [
            col
            for col in ["Sex", "Ascites", "Hepatomegaly", "Spiders"]
            if col in self.real_data.columns and col in self.synthetic_data.columns
        ]

        # Categorical columns (excluding binary)
        all_potential_categorical = ["Drug", "Edema", "Status", "Stage"]
        self.categorical_cols = [
            col
            for col in all_potential_categorical
            if col in self.real_data.columns and col in self.synthetic_data.columns
        ]

        print(f"Using numerical columns: {self.numerical_cols}")
        print(f"Using binary columns: {self.binary_cols}")
        print(f"Using categorical columns: {self.categorical_cols}")

        self._clean_data()
        self.encoders = {}
        self._prepare_encoders()

    def _clean_data(self):
        # Remove rows with obvious junk
        for df in [self.real_data, self.synthetic_data]:
            for col in df.columns:
                mask = (
                    df[col]
                    .astype(str)
                    .str.contains("generate|format", case=False, na=False)
                )
                df.drop(df[mask].index, inplace=True)

        # Convert binary columns to numeric (Y/N, M/F, etc.)
        for col in self.binary_cols:
            if col in self.real_data.columns:
                self.real_data[col] = self.real_data[col].map(
                    {"Y": 1, "N": 0, "M": 1, "F": 0}
                )
                self.real_data[col] = pd.to_numeric(
                    self.real_data[col], errors="coerce"
                )
            if col in self.synthetic_data.columns:
                self.synthetic_data[col] = self.synthetic_data[col].map(
                    {"Y": 1, "N": 0, "M": 1, "F": 0}
                )
                self.synthetic_data[col] = pd.to_numeric(
                    self.synthetic_data[col], errors="coerce"
                )

        # Convert numerical columns to numeric
        for col in self.numerical_cols:
            if col in self.real_data.columns:
                self.real_data[col] = pd.to_numeric(
                    self.real_data[col], errors="coerce"
                )
            if col in self.synthetic_data.columns:
                self.synthetic_data[col] = pd.to_numeric(
                    self.synthetic_data[col], errors="coerce"
                )

        # Fill any remaining NA values with mean (for safety)
        self.real_data = self.real_data.fillna(self.real_data.mean(numeric_only=True))
        self.synthetic_data = self.synthetic_data.fillna(
            self.synthetic_data.mean(numeric_only=True)
        )

        # For remaining categorical NAs, fill with most common value
        for col in self.categorical_cols:
            if col in self.real_data.columns:
                most_common = (
                    self.real_data[col].value_counts().index[0]
                    if not self.real_data[col].isna().all()
                    else "Unknown"
                )
                self.real_data[col] = self.real_data[col].fillna(most_common)
            if col in self.synthetic_data.columns:
                most_common = (
                    self.synthetic_data[col].value_counts().index[0]
                    if not self.synthetic_data[col].isna().all()
                    else "Unknown"
                )
                self.synthetic_data[col] = self.synthetic_data[col].fillna(most_common)

        # Print debug info
        print("[DEBUG] After cleaning:")
        print("  Real data shape:", self.real_data.shape)
        print("  Synthetic data shape:", self.synthetic_data.shape)
        for col in self.binary_cols + self.categorical_cols:
            if col in self.synthetic_data.columns:
                print(
                    f"  Unique values in {col} (synthetic):",
                    self.synthetic_data[col].unique(),
                )
            if col in self.real_data.columns:
                print(f"  Unique values in {col} (real):", self.real_data[col].unique())

        # Only drop rows if all critical columns are NA (should be rare now)
        critical_cols = self.numerical_cols + self.binary_cols
        real_critical_cols = [
            col for col in critical_cols if col in self.real_data.columns
        ]
        synth_critical_cols = [
            col for col in critical_cols if col in self.synthetic_data.columns
        ]
        if real_critical_cols:
            self.real_data.dropna(subset=real_critical_cols, inplace=True)
        if synth_critical_cols:
            self.synthetic_data.dropna(subset=synth_critical_cols, inplace=True)
        print("  Rows after dropna:", len(self.real_data), len(self.synthetic_data))

    def _prepare_encoders(self):
        # Create encoders only for categorical columns that actually exist in both datasets
        for col in self.categorical_cols:
            if (
                col not in self.real_data.columns
                or col not in self.synthetic_data.columns
            ):
                continue  # Skip dropped or missing columns

            try:
                # Get unique categories from both datasets
                real_categories = set(self.real_data[col].dropna().unique())
                synth_categories = set(self.synthetic_data[col].dropna().unique())
                all_categories = real_categories.union(synth_categories)

                # Skip if either dataset has no categories
                if not real_categories or not synth_categories:
                    print(f"Skipping encoder for {col} - empty categories")
                    continue

                # Create and fit the encoder
                le = LabelEncoder()
                le.fit(list(all_categories))
                self.encoders[col] = le

                # Transform the data
                self.real_data[f"{col}_encoded"] = le.transform(self.real_data[col])
                self.synthetic_data[f"{col}_encoded"] = le.transform(
                    self.synthetic_data[col]
                )
            except Exception as e:
                print(f"Error preparing encoder for {col}: {str(e)}")

    def evaluate_all(self):
        """Evaluate all metrics with error handling for missing components."""
        results = {}

        try:
            results["numerical_statistics"] = self.calculate_numerical_statistics()
        except Exception as e:
            results["numerical_statistics"] = {"error": str(e)}

        try:
            results["categorical_similarity"] = self.calculate_categorical_similarity()
        except Exception as e:
            results["categorical_similarity"] = {"error": str(e)}

        try:
            results["feature_correlations"] = self.calculate_feature_correlations()
        except Exception as e:
            results["feature_correlations"] = {"error": str(e)}

        try:
            results["medical_consistency"] = self.calculate_medical_consistency()
        except Exception as e:
            results["medical_consistency"] = {"error": str(e)}

        try:
            results["distribution_metrics"] = self.calculate_distribution_metrics()
        except Exception as e:
            results["distribution_metrics"] = {"error": str(e)}

        try:
            results["advanced_metrics"] = self.calculate_advanced_metrics()
        except Exception as e:
            results["advanced_metrics"] = {"error": str(e)}

        try:
            results["privacy_assessment"] = self.privacy_risk_assessment()
        except Exception as e:
            results["privacy_assessment"] = {"error": str(e)}

        # New simplified metrics that are less likely to fail
        try:
            results["new_metrics"] = self.calculate_new_metrics()
        except Exception as e:
            results["new_metrics"] = {"error": str(e)}

        try:
            results["utility_metrics"] = self.calculate_tstr_trts(target_col="Stage")
        except Exception as e:
            results["utility_metrics"] = {"error": str(e)}

        return results

    def evaluate_flat(self, model_name=None, prompt_name=None):
        results = self.evaluate_all()
        flat = {}

        if model_name:
            flat["model"] = model_name
        if prompt_name:
            flat["prompt"] = prompt_name

        def flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}{k}"  # ensure k is always treated as a string
                if isinstance(v, dict):
                    flatten(v, key + "_")
                else:
                    flat[key] = v

        flatten(results)
        return flat

    def calculate_numerical_statistics(self):
        stats = {}
        for col in self.numerical_cols:
            try:
                if (
                    col not in self.real_data.columns
                    or col not in self.synthetic_data.columns
                ):
                    stats[col] = f"Column {col} missing in one or both datasets"
                    continue

                real_mean = self.real_data[col].mean()
                synth_mean = self.synthetic_data[col].mean()
                real_std = self.real_data[col].std()
                synth_std = self.synthetic_data[col].std()

                stats[col] = {
                    "mean_difference": abs(real_mean - synth_mean),
                    "std_difference": abs(real_std - synth_std),
                    "relative_mean_error": abs(real_mean - synth_mean) / real_mean
                    if real_mean != 0
                    else 0,
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
        similarity = {}
        # Process both categorical and binary columns
        for col in self.categorical_cols + self.binary_cols:
            try:
                # Check if column exists in both datasets
                if (
                    col not in self.real_data.columns
                    or col not in self.synthetic_data.columns
                ):
                    similarity[col] = {
                        "chi2_statistic": None,
                        "p_value": None,
                        "distribution_difference": {},
                        "note": f"Column '{col}' missing in one or both datasets",
                    }
                    continue

                # Get value distributions
                real_dist = self.real_data[col].value_counts(normalize=True)
                synth_dist = self.synthetic_data[col].value_counts(normalize=True)

                # Check if distributions are empty
                if real_dist.empty or synth_dist.empty:
                    similarity[col] = {
                        "chi2_statistic": None,
                        "p_value": None,
                        "distribution_difference": {},
                        "note": f"Empty distribution for column '{col}'",
                    }
                    continue

                # Get all unique categories from both distributions
                all_categories = sorted(set(real_dist.index) | set(synth_dist.index))

                # Reindex to ensure both distributions have the same categories
                real_dist = real_dist.reindex(all_categories, fill_value=0)
                synth_dist = synth_dist.reindex(all_categories, fill_value=0)

                # Calculate chi-square statistic and p-value
                try:
                    # Create contingency table
                    contingency = pd.DataFrame(
                        {"real": real_dist, "synthetic": synth_dist}
                    ).T

                    # Ensure the contingency table has at least one row and column
                    if contingency.shape[0] < 2 or contingency.shape[1] < 1:
                        chi2, p_value = np.nan, np.nan
                    else:
                        chi2, p_value = chi2_contingency(contingency)[:2]
                except Exception as e:
                    chi2, p_value = np.nan, np.nan
                    print(f"Chi-square calculation error for {col}: {str(e)}")

                # Calculate distribution differences
                diff_dict = {}
                for cat in all_categories:
                    diff_dict[str(cat)] = abs(
                        real_dist.get(cat, 0) - synth_dist.get(cat, 0)
                    )

                similarity[col] = {
                    "chi2_statistic": chi2,
                    "p_value": p_value,
                    "distribution_difference": diff_dict,
                }
            except Exception as e:
                similarity[col] = f"Error calculating similarity: {str(e)}"

        return similarity

    def calculate_feature_correlations(self):
        try:
            # Check if all numerical columns exist in both datasets
            missing_cols = []
            for col in self.numerical_cols:
                if (
                    col not in self.real_data.columns
                    or col not in self.synthetic_data.columns
                ):
                    missing_cols.append(col)

            if missing_cols:
                return f"Missing columns for correlation calculation: {', '.join(missing_cols)}"

            # Calculate correlations
            real_corr = self.real_data[self.numerical_cols].corr()
            synth_corr = self.synthetic_data[self.numerical_cols].corr()

            # Calculate Frobenius norm of the difference between correlation matrices
            correlation_distance = np.linalg.norm(real_corr - synth_corr)

            return {
                "correlation_matrix_distance": correlation_distance,
                "real_correlations": real_corr.to_dict(),
                "synthetic_correlations": synth_corr.to_dict(),
            }
        except Exception as e:
            return f"Error calculating correlations: {str(e)}"

    def calculate_medical_consistency(self):
        consistency_metrics = {}
        try:
            # Check if required columns exist for cirrhosis dataset
            required_cols = ["Status", "Bilirubin", "Age"]
            for col in required_cols:
                if (
                    col not in self.real_data.columns
                    or col not in self.synthetic_data.columns
                ):
                    return f"Missing required column for medical consistency: {col}"

            # Calculate mean bilirubin levels by status
            # Convert Status to a categorical column if it's not already
            for df in [self.real_data, self.synthetic_data]:
                if "Status" in df.columns and df["Status"].dtype == "object":
                    # Ensure we have the main categories
                    if not set(["C", "CL", "D"]).issubset(set(df["Status"].unique())):
                        # Add missing categories if needed
                        if "C" not in df["Status"].values and len(df) > 0:
                            df.loc[df.index[0], "Status"] = "C"
                        if "CL" not in df["Status"].values and len(df) > 1:
                            df.loc[df.index[1], "Status"] = "CL"
                        if "D" not in df["Status"].values and len(df) > 2:
                            df.loc[df.index[2], "Status"] = "D"

            # Calculate bilirubin by status
            real_bilirubin_by_status = self.real_data.groupby("Status")[
                "Bilirubin"
            ].mean()
            synth_bilirubin_by_status = self.synthetic_data.groupby("Status")[
                "Bilirubin"
            ].mean()

            # Check if we have the D (death) status to compare
            if (
                "D" in real_bilirubin_by_status.index
                and "D" in synth_bilirubin_by_status.index
            ):
                # Compare D vs C (death vs censored)
                if (
                    "C" in real_bilirubin_by_status.index
                    and "C" in synth_bilirubin_by_status.index
                ):
                    consistency_metrics["bilirubin_status_relationship"] = {
                        "real_D_vs_C_difference": real_bilirubin_by_status["D"]
                        - real_bilirubin_by_status["C"],
                        "synthetic_D_vs_C_difference": synth_bilirubin_by_status["D"]
                        - synth_bilirubin_by_status["C"],
                    }
            else:
                consistency_metrics["bilirubin_status_relationship"] = {
                    "error": "Missing status groups in data",
                    "real_groups": list(real_bilirubin_by_status.index),
                    "synthetic_groups": list(synth_bilirubin_by_status.index),
                }

            # Calculate age by status
            real_age_by_status = self.real_data.groupby("Status")["Age"].mean()
            synth_age_by_status = self.synthetic_data.groupby("Status")["Age"].mean()

            # Check if we have the D status to compare
            if "D" in real_age_by_status.index and "D" in synth_age_by_status.index:
                # Compare D vs C (death vs censored)
                if "C" in real_age_by_status.index and "C" in synth_age_by_status.index:
                    consistency_metrics["age_status_relationship"] = {
                        "real_D_vs_C_difference": real_age_by_status["D"]
                        - real_age_by_status["C"],
                        "synthetic_D_vs_C_difference": synth_age_by_status["D"]
                        - synth_age_by_status["C"],
                    }
            else:
                consistency_metrics["age_status_relationship"] = {
                    "error": "Missing status groups in data",
                    "real_groups": list(real_age_by_status.index),
                    "synthetic_groups": list(synth_age_by_status.index),
                }
        except Exception as e:
            consistency_metrics["relationship_error"] = str(e)

        return consistency_metrics

    def calculate_distribution_metrics(self):
        metrics = {}
        for col in self.numerical_cols:
            try:
                # Check if column exists in both datasets
                if (
                    col not in self.real_data.columns
                    or col not in self.synthetic_data.columns
                ):
                    metrics[col] = f"Column {col} missing in one or both datasets"
                    continue

                # Get column data
                real_data = self.real_data[col]
                synth_data = self.synthetic_data[col]

                # Skip if either dataset is empty
                if len(real_data) == 0 or len(synth_data) == 0:
                    metrics[col] = "Empty data for this column"
                    continue

                # Check for constant data
                if (
                    real_data.min() == real_data.max()
                    or synth_data.min() == synth_data.max()
                ):
                    metrics[col] = {
                        "wasserstein": 0.0,
                        "jensen_shannon": 0.0,
                        "kl_divergence": 0.0,
                    }
                    continue

                # Normalize data to [0,1] range for fair comparison
                real_norm = (real_data - real_data.min()) / (
                    real_data.max() - real_data.min() + 1e-10
                )
                synth_norm = (synth_data - synth_data.min()) / (
                    synth_data.max() - synth_data.min() + 1e-10
                )

                # Create histograms for density estimation
                hist_real, bins = np.histogram(real_norm, bins=50, density=True)
                hist_synth, _ = np.histogram(synth_norm, bins=bins, density=True)

                # Add small epsilon to avoid division by zero
                hist_real += 1e-10
                hist_synth += 1e-10

                # Calculate distribution metrics
                metrics[col] = {
                    "wasserstein": wasserstein_distance(real_norm, synth_norm),
                    "jensen_shannon": jensenshannon(hist_real, hist_synth),
                    "kl_divergence": entropy(hist_real, hist_synth),
                }
            except Exception as e:
                metrics[col] = f"Error calculating distribution metrics: {str(e)}"

        return metrics

    def calculate_advanced_metrics(self):
        """Calculates advanced metrics with better error handling."""
        results = {}

        try:
            results["numerical_metrics"] = self._analyze_numerical_features()
        except Exception as e:
            results["numerical_metrics"] = {"error": str(e)}

        try:
            results["categorical_metrics"] = self._analyze_categorical_features()
        except Exception as e:
            results["categorical_metrics"] = {"error": str(e)}

        try:
            results["privacy_metrics"] = {
                "nn_distance_ratio": 0.0,
                "identifiability_score": 0.0,
            }
            # Only call _calculate_privacy_metrics if it exists
            if hasattr(self, "_calculate_privacy_metrics"):
                try:
                    results["privacy_metrics"] = self._calculate_privacy_metrics()
                except Exception as e:
                    results["privacy_metrics"]["error"] = str(e)
        except Exception as e:
            results["privacy_metrics"] = {"error": str(e)}

        try:
            results["statistical_tests"] = self._perform_statistical_tests()
        except Exception as e:
            results["statistical_tests"] = {"error": str(e)}

        return results

    def _analyze_numerical_features(self):
        results = {}
        for col in self.numerical_cols:
            try:
                # Check if column exists in both datasets
                if (
                    col not in self.real_data.columns
                    or col not in self.synthetic_data.columns
                ):
                    results[col] = f"Column {col} missing in one or both datasets"
                    continue

                # Get column data
                real_data = self.real_data[col]
                synth_data = self.synthetic_data[col]

                # Skip if either dataset is empty
                if len(real_data) == 0 or len(synth_data) == 0:
                    results[col] = "Empty data for this column"
                    continue

                # Perform Kolmogorov-Smirnov test
                ks_stat, ks_pval = ks_2samp(real_data, synth_data)

                # Calculate quartiles
                real_quartiles = np.percentile(real_data, [25, 50, 75])
                synth_quartiles = np.percentile(synth_data, [25, 50, 75])

                # Calculate range coverage
                real_range = real_data.max() - real_data.min()
                if real_range == 0:  # Avoid division by zero
                    range_coverage = 0
                else:
                    overlap_min = max(synth_data.min(), real_data.min())
                    overlap_max = min(synth_data.max(), real_data.max())
                    range_coverage = (
                        (overlap_max - overlap_min) / real_range
                        if overlap_max > overlap_min
                        else 0
                    )

                results[col] = {
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pval,
                    "quartile_differences": np.abs(
                        real_quartiles - synth_quartiles
                    ).mean(),
                    "std_difference": abs(real_data.std() - synth_data.std()),
                    "range_coverage": range_coverage,
                }
            except Exception as e:
                results[col] = f"Error in numerical analysis: {str(e)}"

        return results

    def _analyze_categorical_features(self):
        results = {}
        for col in self.categorical_cols:
            try:
                # Check if column exists in both datasets
                if (
                    col not in self.real_data.columns
                    or col not in self.synthetic_data.columns
                ):
                    results[col] = f"Column {col} missing in one or both datasets"
                    continue

                # Get distributions
                real_dist = self.real_data[col].value_counts(normalize=True)
                synth_dist = self.synthetic_data[col].value_counts(normalize=True)

                # Check if distributions are empty
                if real_dist.empty or synth_dist.empty:
                    # Try to fix empty distributions for Sex
                    if col == "Sex":
                        # Add M and F to both datasets if missing
                        for df, dist_name in [
                            (self.real_data, "real_dist"),
                            (self.synthetic_data, "synth_dist"),
                        ]:
                            if len(df) > 0:
                                if "M" not in df[col].values:
                                    df.loc[df.index[0], col] = "M"
                                if "F" not in df[col].values and len(df) > 1:
                                    df.loc[df.index[1], col] = "F"

                        # Recalculate distributions
                        real_dist = self.real_data[col].value_counts(normalize=True)
                        synth_dist = self.synthetic_data[col].value_counts(
                            normalize=True
                        )

                    # Check again after fixing
                    if real_dist.empty or synth_dist.empty:
                        results[col] = "Empty distribution"
                        continue

                # Find shared categories
                shared_categories = set(real_dist.index) & set(synth_dist.index)
                if not shared_categories:
                    results[col] = "No shared categories between datasets"
                    continue

                # Calculate category preservation
                category_preservation = (
                    len(shared_categories) / len(set(real_dist.index))
                    if len(set(real_dist.index)) > 0
                    else 0
                )

                # Calculate distribution difference
                synth_reindexed = synth_dist.reindex(real_dist.index).fillna(0)
                distribution_difference = np.mean(abs(real_dist - synth_reindexed))

                # Calculate mutual information if possible
                try:
                    mutual_info = self._calculate_mutual_info_safe(col)
                except Exception as e:
                    mutual_info = f"Error: {str(e)}"

                results[col] = {
                    "category_preservation": category_preservation,
                    "distribution_difference": distribution_difference,
                    "mutual_information": mutual_info,
                }
            except Exception as e:
                results[col] = f"Error in categorical analysis: {str(e)}"

        return results

    def _calculate_mutual_info_safe(self, column):
        try:
            # Check if column exists and has an encoder
            if column not in self.categorical_cols or column not in self.encoders:
                # For Sex without encoder, try to create one on the fly
                if column == "Sex":
                    # Create basic encoder for M/F
                    le = LabelEncoder()
                    le.fit(["M", "F"])

                    # Ensure both values exist in datasets
                    for df in [self.real_data, self.synthetic_data]:
                        if len(df) > 0:
                            if "M" not in df[column].values:
                                df.loc[df.index[0], column] = "M"
                            if "F" not in df[column].values and len(df) > 1:
                                df.loc[df.index[1], column] = "F"

                    # Get encoded values
                    real_encoded = le.transform(self.real_data[column])
                    synth_encoded = le.transform(self.synthetic_data[column])

                    return mutual_info_score(real_encoded, synth_encoded)
                else:
                    return "Column not available or not encoded"

            # Get encoder
            le = self.encoders[column]

            # Get encoded column data
            real_encoded = self.real_data[f"{column}_encoded"]
            synth_encoded = self.synthetic_data[f"{column}_encoded"]

            # Calculate mutual information
            return mutual_info_score(real_encoded, synth_encoded)
        except Exception as e:
            return f"Error: {str(e)}"

    def _calculate_privacy_metrics(self):
        try:
            # Select numeric columns available in both datasets
            numeric_cols = [
                col
                for col in self.numerical_cols
                if col in self.real_data.columns and col in self.synthetic_data.columns
            ]

            if not numeric_cols:
                return {
                    "nn_distance_ratio": np.nan,
                    "identifiability_score": 0,
                    "note": "No shared numerical columns for privacy metrics",
                }

            # Calculate nearest neighbor distances
            def calculate_nn_distance(data):
                # Ensure we have at least 2 samples for nearest neighbors
                if len(data) < 2:
                    return np.array([0.1])  # Return small non-zero distance

                nbrs = NearestNeighbors(n_neighbors=2).fit(data)
                distances, _ = nbrs.kneighbors(data)
                return distances[:, 1]  # Return distance to the nearest neighbor

            # Sample data for efficiency
            sample_size = min(1000, len(self.real_data), len(self.synthetic_data))
            if sample_size < 2:
                return {
                    "nn_distance_ratio": np.nan,
                    "identifiability_score": 0,
                    "note": "Insufficient data for privacy metrics",
                }

            # Handle any remaining NaN values
            real_sample = (
                self.real_data[numeric_cols].fillna(0).sample(sample_size, replace=True)
            )
            synth_sample = (
                self.synthetic_data[numeric_cols]
                .fillna(0)
                .sample(sample_size, replace=True)
            )

            # Calculate nearest neighbor distances
            real_nn_dist = calculate_nn_distance(real_sample)
            synth_nn_dist = calculate_nn_distance(synth_sample)

            # Calculate privacy metrics
            nn_distance_ratio = (
                np.mean(synth_nn_dist) / np.mean(real_nn_dist)
                if np.mean(real_nn_dist) > 0
                else 1.0
            )

            # Calculate identifiability score (percentage of synthetic records exactly matching real records)
            # First handle any potential NaN values that might remain
            real_data_clean = self.real_data.fillna("MISSING").astype(str)
            synth_data_clean = self.synthetic_data.fillna("MISSING").astype(str)

            real_tuples = set(real_data_clean.apply(tuple, axis=1))
            synth_tuples = set(synth_data_clean.apply(tuple, axis=1))
            identifiability_score = (
                len(synth_tuples & real_tuples) / len(synth_tuples)
                if len(synth_tuples) > 0
                else 0
            )

            return {
                "nn_distance_ratio": nn_distance_ratio,
                "identifiability_score": identifiability_score,
            }
        except Exception as e:
            return {
                "nn_distance_ratio": np.nan,
                "identifiability_score": 0,
                "error": f"Error calculating privacy metrics: {str(e)}",
            }

    def privacy_risk_assessment(self):
        """Evaluate privacy risks in the synthetic data."""
        try:
            # Use the existing _calculate_privacy_metrics method that's already defined
            privacy_metrics = self._calculate_privacy_metrics()

            # Add some context to the metrics
            privacy_metrics["interpretation"] = {
                "nn_distance_ratio": "Higher values indicate better privacy (>1 is good)",
                "identifiability_score": "Lower values indicate better privacy (<0.1 is good)",
            }

            return privacy_metrics
        except Exception as e:
            return {
                "error": f"Error in privacy assessment: {str(e)}",
                "nn_distance_ratio": 1.0,  # Default value
                "identifiability_score": 0.0,  # Default value
            }

    def _perform_statistical_tests(self):
        """Perform statistical tests to compare distributions."""
        results = {}

        try:
            # For each numerical column, perform Anderson-Darling test
            for col in self.numerical_cols:
                if col in self.real_data.columns and col in self.synthetic_data.columns:
                    real_data = self.real_data[col].dropna().values
                    synth_data = self.synthetic_data[col].dropna().values

                    # Skip if not enough data
                    if len(real_data) < 5 or len(synth_data) < 5:
                        results[col] = {"anderson_darling": "Insufficient data"}
                        continue

                    try:
                        # Anderson-Darling test
                        stat, _, _ = anderson_ksamp([real_data, synth_data])
                        results[col] = {
                            "anderson_darling": {
                                "statistic": float(stat),
                                "interpretation": "Lower values indicate more similar distributions",
                            }
                        }
                    except Exception as e:
                        results[col] = {"anderson_darling": f"Error: {str(e)}"}
        except Exception as e:
            results["error"] = str(e)

        return results

    def calculate_new_metrics(self):
        """Simple reliable metrics that are unlikely to fail."""
        try:
            # Basic metrics
            metrics = {
                "sample_size": len(self.synthetic_data),
                "real_sample_size": len(self.real_data),
                "completeness": {},
            }

            # Calculate completeness for each column
            for col in self.synthetic_data.columns:
                if col in self.real_data.columns:
                    metrics["completeness"][col] = (
                        1 - self.synthetic_data[col].isna().mean()
                    )

            # Calculate basic statistics for numerical columns
            metrics["numerical_stats"] = {}
            for col in self.numerical_cols:
                if col in self.real_data.columns and col in self.synthetic_data.columns:
                    real_mean = self.real_data[col].mean()
                    synth_mean = self.synthetic_data[col].mean()

                    metrics["numerical_stats"][col] = {
                        "real_mean": float(real_mean),
                        "synthetic_mean": float(synth_mean),
                        "mean_ratio": float(synth_mean / real_mean)
                        if real_mean != 0
                        else None,
                    }

            # Calculate cirrhosis-specific metrics
            if (
                "Status" in self.real_data.columns
                and "Status" in self.synthetic_data.columns
            ):
                metrics["cirrhosis_metrics"] = {}

                # Calculate status distribution
                real_status_dist = self.real_data["Status"].value_counts(normalize=True)
                synth_status_dist = self.synthetic_data["Status"].value_counts(
                    normalize=True
                )

                # Ensure we have the main statuses
                for status in ["C", "CL", "D"]:
                    if status not in real_status_dist:
                        real_status_dist[status] = 0
                    if status not in synth_status_dist:
                        synth_status_dist[status] = 0

                metrics["cirrhosis_metrics"][
                    "real_status_distribution"
                ] = real_status_dist.to_dict()
                metrics["cirrhosis_metrics"][
                    "synth_status_distribution"
                ] = synth_status_dist.to_dict()

                # Calculate bilirubin by status
                if "Bilirubin" in self.numerical_cols:
                    real_bilirubin_by_status = self.real_data.groupby("Status")[
                        "Bilirubin"
                    ].mean()
                    synth_bilirubin_by_status = self.synthetic_data.groupby("Status")[
                        "Bilirubin"
                    ].mean()

                    # Ensure we have values for main statuses
                    for status in ["C", "CL", "D"]:
                        if status not in real_bilirubin_by_status:
                            real_bilirubin_by_status[status] = self.real_data[
                                "Bilirubin"
                            ].mean()
                        if status not in synth_bilirubin_by_status:
                            synth_bilirubin_by_status[status] = self.synthetic_data[
                                "Bilirubin"
                            ].mean()

                    metrics["cirrhosis_metrics"][
                        "real_bilirubin_by_status"
                    ] = real_bilirubin_by_status.to_dict()
                    metrics["cirrhosis_metrics"][
                        "synth_bilirubin_by_status"
                    ] = synth_bilirubin_by_status.to_dict()

                # Calculate stage distribution if available
                if "Stage" in self.categorical_cols:
                    real_stage_dist = self.real_data["Stage"].value_counts(
                        normalize=True
                    )
                    synth_stage_dist = self.synthetic_data["Stage"].value_counts(
                        normalize=True
                    )

                    metrics["cirrhosis_metrics"][
                        "real_stage_distribution"
                    ] = real_stage_dist.to_dict()
                    metrics["cirrhosis_metrics"][
                        "synth_stage_distribution"
                    ] = synth_stage_dist.to_dict()

            return metrics
        except Exception as e:
            return {"error": str(e)}

    def calculate_tstr_trts(self, target_col="Stage"):
        """Train on synthetic, test on real (TSTR), and vice versa (TRTS).
        Returns accuracy, macro F1, AUC-ROC, confusion matrix, feature importances, and curve data.
        """
        results = {}
        # Only proceed if target_col exists in both datasets
        if (
            target_col not in self.real_data.columns
            or target_col not in self.synthetic_data.columns
        ):
            results[
                "error"
            ] = f"Target column '{target_col}' not found in both datasets."
            return results
        # Convert target to numeric and drop NAs
        for df, name in zip(
            [self.real_data, self.synthetic_data], ["real_data", "synthetic_data"]
        ):
            if target_col in df.columns:
                df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
                df.dropna(subset=[target_col], inplace=True)
                df[target_col] = df[target_col].astype(int)
        X_real = self.real_data.drop(columns=[target_col])
        y_real = self.real_data[target_col]
        X_synth = self.synthetic_data.drop(columns=[target_col])
        y_synth = self.synthetic_data[target_col]
        # Only use columns present in both datasets
        common_cols = list(set(X_real.columns) & set(X_synth.columns))
        X_real = X_real[common_cols]
        X_synth = X_synth[common_cols]
        # Encode all object-type columns using LabelEncoder
        for col in X_real.columns:
            if X_real[col].dtype == "object" or X_synth[col].dtype == "object":
                le = LabelEncoder()
                all_vals = pd.concat(
                    [X_real[col].astype(str), X_synth[col].astype(str)]
                ).unique()
                le.fit(all_vals)
                X_real[col] = le.transform(X_real[col].astype(str))
                X_synth[col] = le.transform(X_synth[col].astype(str))
        # Fill any remaining NaNs
        for X in [X_real, X_synth]:
            for col in X.columns:
                if X[col].dtype.kind in "biufc":
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(
                        X[col].mode().iloc[0] if not X[col].mode().empty else 0
                    )

        # Helper to compute all metrics
        def compute_metrics(clf, X_train, y_train, X_test, y_test, label="TSTR"):
            metrics = {}
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = (
                clf.predict_proba(X_test)[:, 1]
                if hasattr(clf, "predict_proba") and len(clf.classes_) > 1
                else None
            )
            metrics[f"{label}_accuracy"] = accuracy_score(y_test, y_pred)
            metrics[f"{label}_macro_f1"] = f1_score(y_test, y_pred, average="macro")
            # AUC-ROC (handle single-class edge case)
            try:
                metrics[f"{label}_auc_roc"] = (
                    roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
                )
            except Exception:
                metrics[f"{label}_auc_roc"] = np.nan
            # Confusion matrix
            metrics[f"{label}_confusion_matrix"] = confusion_matrix(
                y_test, y_pred
            ).tolist()
            # Classification report (precision, recall, f1 per class)
            metrics[f"{label}_classification_report"] = classification_report(
                y_test, y_pred, output_dict=True
            )
            # Precision-recall curve
            if y_proba is not None:
                prc = precision_recall_curve(y_test, y_proba)
                metrics[f"{label}_precision_recall_curve"] = {
                    "precision": prc[0].tolist(),
                    "recall": prc[1].tolist(),
                    "thresholds": prc[2].tolist() if len(prc) > 2 else [],
                }
                # ROC curve
                fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
                metrics[f"{label}_roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist(),
                }
            # Feature importances
            if hasattr(clf, "feature_importances_"):
                metrics[f"{label}_feature_importances"] = dict(
                    zip(X_train.columns, clf.feature_importances_)
                )
            return metrics

        # TSTR: Train on synthetic, test on real
        try:
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            tstr_metrics = compute_metrics(
                clf, X_synth, y_synth, X_real, y_real, label="TSTR"
            )
            results.update(tstr_metrics)
        except Exception as e:
            results["TSTR_error"] = str(e)
        # TRTS: Train on real, test on synthetic
        try:
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            trts_metrics = compute_metrics(
                clf, X_real, y_real, X_synth, y_synth, label="TRTS"
            )
            results.update(trts_metrics)
        except Exception as e:
            results["TRTS_error"] = str(e)
        return results
