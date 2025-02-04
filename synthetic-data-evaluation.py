import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, wasserstein_distance
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


class SyntheticDataEvaluator:
    def __init__(self, real_data, synthetic_data):
        """
        Initialize with real and synthetic datasets

        Parameters:
        real_data (pd.DataFrame): Original dataset
        synthetic_data (pd.DataFrame): Generated synthetic dataset
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.columns = real_data.columns

        # Identify numeric and categorical columns
        self.numeric_columns = real_data.select_dtypes(include=[np.number]).columns
        self.categorical_columns = real_data.select_dtypes(exclude=[np.number]).columns

        # Encode categorical variables for certain analyses
        self.label_encoders = {}
        self._encode_categorical()

    def _encode_categorical(self):
        """Encode categorical variables for numerical analysis"""
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.real_data[f"{col}_encoded"] = le.fit_transform(self.real_data[col])
            self.synthetic_data[f"{col}_encoded"] = le.transform(
                self.synthetic_data[col]
            )
            self.label_encoders[col] = le

    def calculate_statistical_similarity(self):
        """Calculate statistical similarities between real and synthetic data"""
        stats = {}

        # Numeric columns
        for col in self.numeric_columns:
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
                "range_difference": abs(
                    self.real_data[col].max() - self.synthetic_data[col].max()
                ),
            }

        # Categorical columns
        for col in self.categorical_columns:
            real_dist = self.real_data[col].value_counts(normalize=True)
            synth_dist = self.synthetic_data[col].value_counts(normalize=True)

            stats[col] = {
                "category_distribution_difference": sum(
                    abs(real_dist - synth_dist.reindex(real_dist.index).fillna(0))
                )
                / 2
            }

        return stats

    def calculate_distributions_distance(self):
        """Calculate distribution distances for each feature"""
        distances = {}

        # Numeric columns
        for col in self.numeric_columns:
            real_hist, bins = np.histogram(self.real_data[col], bins=50, density=True)
            synth_hist, _ = np.histogram(
                self.synthetic_data[col], bins=bins, density=True
            )

            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            real_hist = real_hist + epsilon
            synth_hist = synth_hist + epsilon

            # Normalize
            real_hist = real_hist / real_hist.sum()
            synth_hist = synth_hist / synth_hist.sum()

            distances[col] = {
                "jensen_shannon": jensenshannon(real_hist, synth_hist),
                "wasserstein": wasserstein_distance(
                    self.real_data[col], self.synthetic_data[col]
                ),
            }

        # Categorical columns
        for col in self.categorical_columns:
            real_dist = self.real_data[col].value_counts(normalize=True)
            synth_dist = self.synthetic_data[col].value_counts(normalize=True)

            # Align distributions
            all_categories = sorted(set(real_dist.index) | set(synth_dist.index))
            real_aligned = np.array([real_dist.get(cat, 0) for cat in all_categories])
            synth_aligned = np.array([synth_dist.get(cat, 0) for cat in all_categories])

            # Add small epsilon and normalize
            real_aligned = (real_aligned + epsilon) / (real_aligned + epsilon).sum()
            synth_aligned = (synth_aligned + epsilon) / (synth_aligned + epsilon).sum()

            distances[col] = {
                "jensen_shannon": jensenshannon(real_aligned, synth_aligned)
            }

        return distances

    def calculate_correlation_similarity(self):
        """Compare correlation matrices between real and synthetic data"""
        # Use encoded versions for categorical variables
        real_data_encoded = self.real_data.copy()
        synth_data_encoded = self.synthetic_data.copy()

        for col in self.categorical_columns:
            real_data_encoded[col] = real_data_encoded[f"{col}_encoded"]
            synth_data_encoded[col] = synth_data_encoded[f"{col}_encoded"]

        real_corr = real_data_encoded[self.columns].corr()
        synth_corr = synth_data_encoded[self.columns].corr()

        # Calculate Frobenius norm of difference
        correlation_distance = np.linalg.norm(real_corr - synth_corr)

        return {
            "correlation_matrix_distance": correlation_distance,
            "real_correlations": real_corr,
            "synthetic_correlations": synth_corr,
        }

    def calculate_mutual_information(self):
        """Calculate mutual information between features in both datasets"""
        mi_scores = {}

        # Include both numeric and encoded categorical columns
        all_columns = list(self.numeric_columns)
        for col in self.categorical_columns:
            all_columns.append(f"{col}_encoded")

        for i, col1 in enumerate(all_columns):
            for col2 in all_columns[i + 1 :]:
                real_mi = mutual_info_score(self.real_data[col1], self.real_data[col2])
                synth_mi = mutual_info_score(
                    self.synthetic_data[col1], self.synthetic_data[col2]
                )
                mi_scores[f"{col1}-{col2}"] = {
                    "real_mi": real_mi,
                    "synthetic_mi": synth_mi,
                    "difference": abs(real_mi - synth_mi),
                }

        return mi_scores

    def privacy_risk_assessment(self, k=5):
        """
        Privacy risk assessment using k-anonymity principle and uniqueness analysis
        """

        def get_uniqueness_score(df):
            return len(df.drop_duplicates()) / len(df)

        # Calculate k-anonymity violations
        def count_k_anonymity_violations(df, k):
            counts = df.groupby(list(df.columns)).size()
            return (counts < k).sum()

        # Calculate for both numeric and categorical features
        real_uniqueness = get_uniqueness_score(self.real_data)
        synth_uniqueness = get_uniqueness_score(self.synthetic_data)

        # Calculate k-anonymity violations
        real_violations = count_k_anonymity_violations(self.real_data, k)
        synth_violations = count_k_anonymity_violations(self.synthetic_data, k)

        return {
            "real_uniqueness": real_uniqueness,
            "synthetic_uniqueness": synth_uniqueness,
            "uniqueness_difference": abs(real_uniqueness - synth_uniqueness),
            "k_anonymity_violations": {
                "real": real_violations,
                "synthetic": synth_violations,
                "violation_rate": synth_violations / len(self.synthetic_data),
            },
        }

    def evaluate_all(self):
        """Run all evaluation metrics and return comprehensive report"""
        return {
            "statistical_similarity": self.calculate_statistical_similarity(),
            "distribution_distances": self.calculate_distributions_distance(),
            "correlation_similarity": self.calculate_correlation_similarity(),
            "mutual_information": self.calculate_mutual_information(),
            "privacy_assessment": self.privacy_risk_assessment(),
        }


# Usage example:
"""
# Load your data
real_data = pd.read_csv('diabetes_prediction_dataset.csv')
synthetic_data = pd.read_csv('synthetic_diabetes.csv')

# Initialize evaluator
evaluator = SyntheticDataEvaluator(real_data, synthetic_data)

# Get comprehensive evaluation
evaluation_results = evaluator.evaluate_all()

# Print results
for metric, results in evaluation_results.items():
    print(f"\n{metric.upper()}:")
    print(results)
"""
