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


class DiabetesDataEvaluator:
    def __init__(self, real_data, synthetic_data):
        """
        Initialize with real and synthetic datasets

        Parameters:
        real_data (pd.DataFrame): Original dataset
        synthetic_data (pd.DataFrame): Generated synthetic dataset
        """
        # Make copies to avoid modifying original data
        self.real_data = real_data.copy()
        # Sample real data to match synthetic data size
        if len(self.real_data) > len(synthetic_data):
            self.real_data = self.real_data.sample(
                n=len(synthetic_data), random_state=42
            )
        self.synthetic_data = synthetic_data.copy()

        # Define column types
        self.numerical_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
        self.categorical_cols = ["gender", "smoking_history"]
        self.binary_cols = ["hypertension", "heart_disease", "diabetes"]

        # Clean data
        self._clean_data()

        # Initialize encoders
        self.encoders = {}
        self._prepare_encoders()

    def _clean_data(self):
        """Clean and prepare the data"""
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

        # Convert binary columns to int
        for col in self.binary_cols:
            self.real_data[col] = pd.to_numeric(self.real_data[col], errors="coerce")
            self.synthetic_data[col] = pd.to_numeric(
                self.synthetic_data[col], errors="coerce"
            )

        # Convert numerical columns to float
        for col in self.numerical_cols:
            self.real_data[col] = pd.to_numeric(self.real_data[col], errors="coerce")
            self.synthetic_data[col] = pd.to_numeric(
                self.synthetic_data[col], errors="coerce"
            )

        # Handle missing values
        self.real_data.fillna(self.real_data.mean(numeric_only=True), inplace=True)
        self.synthetic_data.fillna(
            self.synthetic_data.mean(numeric_only=True), inplace=True
        )

    def _prepare_encoders(self):
        """Prepare label encoders for categorical variables"""
        for col in self.categorical_cols:
            # Combine unique values from both datasets
            all_categories = set(self.real_data[col].unique()) | set(
                self.synthetic_data[col].unique()
            )

            # Initialize encoder
            le = LabelEncoder()
            # Fit with all possible categories
            le.fit(list(all_categories))
            self.encoders[col] = le

            # Transform the data
            self.real_data[f"{col}_encoded"] = le.transform(self.real_data[col])
            self.synthetic_data[f"{col}_encoded"] = le.transform(
                self.synthetic_data[col]
            )

    def calculate_numerical_statistics(self):
        """Calculate statistical similarities for numerical variables"""
        stats = {}
        for col in self.numerical_cols:
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
        for col in self.categorical_cols + self.binary_cols:
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
            real_corr = self.real_data[self.numerical_cols].corr()
            synth_corr = self.synthetic_data[self.numerical_cols].corr()

            correlation_distance = np.linalg.norm(real_corr - synth_corr)

            return {
                "correlation_matrix_distance": correlation_distance,
                "real_correlations": real_corr.to_dict(),
                "synthetic_correlations": synth_corr.to_dict(),
            }
        except Exception as e:
            return f"Error calculating correlations: {str(e)}"

    def calculate_medical_consistency(self):
        """Check medical consistency of the synthetic data"""
        consistency_metrics = {}

        try:
            # Check HbA1c and diabetes relationship
            real_hba1c_by_diabetes = self.real_data.groupby("diabetes")[
                "HbA1c_level"
            ].mean()
            synth_hba1c_by_diabetes = self.synthetic_data.groupby("diabetes")[
                "HbA1c_level"
            ].mean()

            consistency_metrics["hba1c_diabetes_relationship"] = {
                "real_difference": real_hba1c_by_diabetes[1]
                - real_hba1c_by_diabetes[0],
                "synthetic_difference": synth_hba1c_by_diabetes[1]
                - synth_hba1c_by_diabetes[0],
            }

            # Check glucose level and diabetes relationship
            real_glucose_by_diabetes = self.real_data.groupby("diabetes")[
                "blood_glucose_level"
            ].mean()
            synth_glucose_by_diabetes = self.synthetic_data.groupby("diabetes")[
                "blood_glucose_level"
            ].mean()

            consistency_metrics["glucose_diabetes_relationship"] = {
                "real_difference": real_glucose_by_diabetes[1]
                - real_glucose_by_diabetes[0],
                "synthetic_difference": synth_glucose_by_diabetes[1]
                - synth_glucose_by_diabetes[0],
            }
        except Exception as e:
            consistency_metrics["relationship_error"] = str(e)

        return consistency_metrics

    def privacy_risk_assessment(self, k=5):
        """Assess privacy risks using k-anonymity for sensitive columns"""
        try:
            identifying_cols = ["age", "gender", "bmi"]

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
        numerical_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
        metrics = {}

        for col in numerical_cols:
            # Normalize the data for fair comparison
            real_norm = (self.real_data[col] - self.real_data[col].min()) / (
                self.real_data[col].max() - self.real_data[col].min()
            )
            synth_norm = (self.synthetic_data[col] - self.synthetic_data[col].min()) / (
                self.synthetic_data[col].max() - self.synthetic_data[col].min()
            )

            # Calculate histograms for KL and JS divergence
            hist_real, bins = np.histogram(real_norm, bins=50, density=True)
            hist_synth, _ = np.histogram(synth_norm, bins=bins, density=True)

            # Add small constant to avoid division by zero
            hist_real = hist_real + 1e-10
            hist_synth = hist_synth + 1e-10

            metrics[col] = {
                "wasserstein": wasserstein_distance(real_norm, synth_norm),
                "jensen_shannon": jensenshannon(hist_real, hist_synth),
                "kl_divergence": entropy(hist_real, hist_synth),
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
        numerical_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
        results = {}

        for col in numerical_cols:
            real_data = self.real_data[col]
            synth_data = self.synthetic_data[col]

            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = ks_2samp(real_data, synth_data)

            # Quartile comparison
            real_quartiles = np.percentile(real_data, [25, 50, 75])
            synth_quartiles = np.percentile(synth_data, [25, 50, 75])

            results[col] = {
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "quartile_differences": np.abs(real_quartiles - synth_quartiles).mean(),
                "std_difference": abs(real_data.std() - synth_data.std()),
                "range_coverage": (
                    min(synth_data.max(), real_data.max())
                    - max(synth_data.min(), real_data.min())
                )
                / (real_data.max() - real_data.min()),
            }
        return results

    def _analyze_categorical_features(self):
        categorical_cols = ["gender", "smoking_history"]
        results = {}

        for col in categorical_cols:
            real_dist = self.real_data[col].value_counts(normalize=True)
            synth_dist = self.synthetic_data[col].value_counts(normalize=True)

            # Category preservation
            shared_categories = set(real_dist.index) & set(synth_dist.index)

            results[col] = {
                "category_preservation": len(shared_categories)
                / len(set(real_dist.index)),
                "distribution_difference": np.mean(
                    abs(real_dist - synth_dist.reindex(real_dist.index).fillna(0))
                ),
                "mutual_information": self._calculate_mutual_info(col),
            }
        return results

    def _calculate_privacy_metrics(self):
        """Calculate privacy-related metrics"""

        # Nearest neighbor distance ratio
        def calculate_nn_distance(data):
            nbrs = NearestNeighbors(n_neighbors=2).fit(data)
            distances, _ = nbrs.kneighbors(data)
            return distances[:, 1]  # Distance to nearest neighbor

        # Sample subset of data for computational efficiency
        sample_size = min(1000, len(self.real_data))
        real_sample = self.real_data.select_dtypes(include=[np.number]).sample(
            sample_size
        )
        synth_sample = self.synthetic_data.select_dtypes(include=[np.number]).sample(
            sample_size
        )

        real_nn_dist = calculate_nn_distance(real_sample)
        synth_nn_dist = calculate_nn_distance(synth_sample)

        return {
            "nn_distance_ratio": np.mean(synth_nn_dist) / np.mean(real_nn_dist),
            "identifiability_score": len(
                set(self.synthetic_data.astype(str).apply(tuple, axis=1))
                & set(self.real_data.astype(str).apply(tuple, axis=1))
            )
            / len(self.synthetic_data),
        }

    def _perform_statistical_tests(self):
        """Perform statistical tests for similarity"""
        numerical_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
        results = {}

        for col in numerical_cols:
            try:
                # Anderson-Darling test for similarity of distributions
                ad_stat = anderson_ksamp(
                    [self.real_data[col], self.synthetic_data[col]]
                )
                results[col] = {
                    "anderson_darling_statistic": ad_stat.statistic,
                    "anderson_darling_pvalue": (
                        ad_stat.pvalue if hasattr(ad_stat, "pvalue") else None
                    ),
                }
            except:
                results[col] = {
                    "anderson_darling_statistic": None,
                    "anderson_darling_pvalue": None,
                }
        return results

    def _calculate_mutual_info(self, column):
        """Calculate mutual information between real and synthetic data for a column"""
        le = LabelEncoder()
        real_encoded = le.fit_transform(self.real_data[column])
        synth_encoded = le.transform(self.synthetic_data[column])
        return mutual_info_score(real_encoded, synth_encoded)

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
        # Load the original dataset
        print("Loading original dataset...")
        real_data = pd.read_csv("diabetes_prediction_dataset_900K.csv")
        print(f"Real data shape: {real_data.shape}")

        # Define prompt files Mistral
        # prompt_files = [
        #     'mistral_PROMPT_1_data.csv',
        #     'mistral_PROMPT_2_data.csv',
        #     # 'mistral_PROMPT_3_data.csv',
        #     # 'mistral_PROMPT_4_data.csv'
        # ]

        # GPT2
        # prompt_files = [
        #     'PROMPT_1_data.csv',
        #     'PROMPT_2_data.csv',
        #     'PROMPT_3_data.csv',
        #     PROMPT_4_data.csv'
        # ]

        # llama
        prompt_files = ["llama_PROMPT_1_data.csv"]

        # Compare each synthetic dataset with the original
        for prompt_file in prompt_files:
            try:
                print(f"\n=== Evaluating {prompt_file} ===")

                # Load synthetic data
                # synthetic_data = pd.read_csv(f'/home/ailaty3088@id.sdsu.edu/SynLLM/opensource/gpt2-prompts/{prompt_file}')
                # synthetic_data = pd.read_csv(f'/home/ailaty3088@id.sdsu.edu/SynLLM/opensource/model_comparison/{prompt_file}')
                synthetic_data = pd.read_csv(
                    f"/home/ailaty3088@id.sdsu.edu/SynLLM/opensource/model_comparison/experiment_20250204_131313/{prompt_file}"
                )
                print(f"Synthetic data shape: {synthetic_data.shape}")

                # Initialize evaluator for this comparison
                print("Initializing evaluator...")
                evaluator = DiabetesDataEvaluator(real_data, synthetic_data)

                # Run evaluation
                print("Running evaluation...")
                evaluation_results = evaluator.evaluate_all()

                # Print results for this prompt
                print(f"\nEvaluation Results for {prompt_file}:")
                for metric, results in evaluation_results.items():
                    print(f"\n{metric.upper()}:")
                    print(results)

                # Optional: Save results to file
                output_file = (
                    f"evaluation_results_{prompt_file.replace('.csv', '.txt')}"
                )
                with open(output_file, "w") as f:
                    f.write(f"Evaluation Results for {prompt_file}:\n")
                    for metric, results in evaluation_results.items():
                        f.write(f"\n{metric.upper()}:\n")
                        f.write(str(results))
                        f.write("\n")

            except Exception as e:
                print(f"Error processing {prompt_file}: {str(e)}")
                continue

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import jensenshannon
# from scipy.stats import chi2_contingency, wasserstein_distance
# from sklearn.metrics import mutual_info_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler


# class SyntheticDataEvaluator:
#     def __init__(self, real_data, synthetic_data):
#         """
#         Initialize with real and synthetic datasets

#         Parameters:
#         real_data (pd.DataFrame): Original dataset
#         synthetic_data (pd.DataFrame): Generated synthetic dataset
#         """
#         self.real_data = real_data.copy()
#         self.synthetic_data = synthetic_data.copy()
#         self.columns = real_data.columns

#         # Identify numeric and categorical columns
#         self.numeric_columns = real_data.select_dtypes(include=[np.number]).columns
#         self.categorical_columns = real_data.select_dtypes(exclude=[np.number]).columns

#         # Encode categorical variables for certain analyses
#         self.label_encoders = {}
#         self._encode_categorical()

#     def _encode_categorical(self):
#         """Encode categorical variables for numerical analysis"""
#         for col in self.categorical_columns:
#             le = LabelEncoder()
#             self.real_data[f"{col}_encoded"] = le.fit_transform(self.real_data[col])
#             self.synthetic_data[f"{col}_encoded"] = le.transform(
#                 self.synthetic_data[col]
#             )
#             self.label_encoders[col] = le

#     def calculate_statistical_similarity(self):
#         """Calculate statistical similarities between real and synthetic data"""
#         stats = {}

#         # Numeric columns
#         for col in self.numeric_columns:
#             real_mean = self.real_data[col].mean()
#             synth_mean = self.synthetic_data[col].mean()
#             real_std = self.real_data[col].std()
#             synth_std = self.synthetic_data[col].std()

#             stats[col] = {
#                 "mean_difference": abs(real_mean - synth_mean),
#                 "std_difference": abs(real_std - synth_std),
#                 "relative_mean_error": abs(real_mean - synth_mean) / real_mean
#                 if real_mean != 0
#                 else 0,
#                 "range_difference": abs(
#                     self.real_data[col].max() - self.synthetic_data[col].max()
#                 ),
#             }

#         # Categorical columns
#         for col in self.categorical_columns:
#             real_dist = self.real_data[col].value_counts(normalize=True)
#             synth_dist = self.synthetic_data[col].value_counts(normalize=True)

#             stats[col] = {
#                 "category_distribution_difference": sum(
#                     abs(real_dist - synth_dist.reindex(real_dist.index).fillna(0))
#                 )
#                 / 2
#             }

#         return stats

#     def calculate_distributions_distance(self):
#         """Calculate distribution distances for each feature"""
#         distances = {}

#         # Numeric columns
#         for col in self.numeric_columns:
#             real_hist, bins = np.histogram(self.real_data[col], bins=50, density=True)
#             synth_hist, _ = np.histogram(
#                 self.synthetic_data[col], bins=bins, density=True
#             )

#             # Add small epsilon to avoid division by zero
#             epsilon = 1e-10
#             real_hist = real_hist + epsilon
#             synth_hist = synth_hist + epsilon

#             # Normalize
#             real_hist = real_hist / real_hist.sum()
#             synth_hist = synth_hist / synth_hist.sum()

#             distances[col] = {
#                 "jensen_shannon": jensenshannon(real_hist, synth_hist),
#                 "wasserstein": wasserstein_distance(
#                     self.real_data[col], self.synthetic_data[col]
#                 ),
#             }

#         # Categorical columns
#         for col in self.categorical_columns:
#             real_dist = self.real_data[col].value_counts(normalize=True)
#             synth_dist = self.synthetic_data[col].value_counts(normalize=True)

#             # Align distributions
#             all_categories = sorted(set(real_dist.index) | set(synth_dist.index))
#             real_aligned = np.array([real_dist.get(cat, 0) for cat in all_categories])
#             synth_aligned = np.array([synth_dist.get(cat, 0) for cat in all_categories])

#             # Add small epsilon and normalize
#             real_aligned = (real_aligned + epsilon) / (real_aligned + epsilon).sum()
#             synth_aligned = (synth_aligned + epsilon) / (synth_aligned + epsilon).sum()

#             distances[col] = {
#                 "jensen_shannon": jensenshannon(real_aligned, synth_aligned)
#             }

#         return distances

#     def calculate_correlation_similarity(self):
#         """Compare correlation matrices between real and synthetic data"""
#         # Use encoded versions for categorical variables
#         real_data_encoded = self.real_data.copy()
#         synth_data_encoded = self.synthetic_data.copy()

#         for col in self.categorical_columns:
#             real_data_encoded[col] = real_data_encoded[f"{col}_encoded"]
#             synth_data_encoded[col] = synth_data_encoded[f"{col}_encoded"]

#         real_corr = real_data_encoded[self.columns].corr()
#         synth_corr = synth_data_encoded[self.columns].corr()

#         # Calculate Frobenius norm of difference
#         correlation_distance = np.linalg.norm(real_corr - synth_corr)

#         return {
#             "correlation_matrix_distance": correlation_distance,
#             "real_correlations": real_corr,
#             "synthetic_correlations": synth_corr,
#         }

#     def calculate_mutual_information(self):
#         """Calculate mutual information between features in both datasets"""
#         mi_scores = {}

#         # Include both numeric and encoded categorical columns
#         all_columns = list(self.numeric_columns)
#         for col in self.categorical_columns:
#             all_columns.append(f"{col}_encoded")

#         for i, col1 in enumerate(all_columns):
#             for col2 in all_columns[i + 1 :]:
#                 real_mi = mutual_info_score(self.real_data[col1], self.real_data[col2])
#                 synth_mi = mutual_info_score(
#                     self.synthetic_data[col1], self.synthetic_data[col2]
#                 )
#                 mi_scores[f"{col1}-{col2}"] = {
#                     "real_mi": real_mi,
#                     "synthetic_mi": synth_mi,
#                     "difference": abs(real_mi - synth_mi),
#                 }

#         return mi_scores

#     def privacy_risk_assessment(self, k=5):
#         """
#         Privacy risk assessment using k-anonymity principle and uniqueness analysis
#         """

#         def get_uniqueness_score(df):
#             return len(df.drop_duplicates()) / len(df)

#         # Calculate k-anonymity violations
#         def count_k_anonymity_violations(df, k):
#             counts = df.groupby(list(df.columns)).size()
#             return (counts < k).sum()

#         # Calculate for both numeric and categorical features
#         real_uniqueness = get_uniqueness_score(self.real_data)
#         synth_uniqueness = get_uniqueness_score(self.synthetic_data)

#         # Calculate k-anonymity violations
#         real_violations = count_k_anonymity_violations(self.real_data, k)
#         synth_violations = count_k_anonymity_violations(self.synthetic_data, k)

#         return {
#             "real_uniqueness": real_uniqueness,
#             "synthetic_uniqueness": synth_uniqueness,
#             "uniqueness_difference": abs(real_uniqueness - synth_uniqueness),
#             "k_anonymity_violations": {
#                 "real": real_violations,
#                 "synthetic": synth_violations,
#                 "violation_rate": synth_violations / len(self.synthetic_data),
#             },
#         }

#     def evaluate_all(self):
#         """Run all evaluation metrics and return comprehensive report"""
#         return {
#             "statistical_similarity": self.calculate_statistical_similarity(),
#             "distribution_distances": self.calculate_distributions_distance(),
#             "correlation_similarity": self.calculate_correlation_similarity(),
#             "mutual_information": self.calculate_mutual_information(),
#             "privacy_assessment": self.privacy_risk_assessment(),
#         }


# # Usage example:
# """
# # Load your data
# real_data = pd.read_csv('diabetes_prediction_dataset.csv')
# synthetic_data = pd.read_csv('synthetic_diabetes.csv')

# # Initialize evaluator
# evaluator = SyntheticDataEvaluator(real_data, synthetic_data)

# # Get comprehensive evaluation
# evaluation_results = evaluator.evaluate_all()

# # Print results
# for metric, results in evaluation_results.items():
#     print(f"\n{metric.upper()}:")
#     print(results)
# """
# # Load your data
# real_data = pd.read_csv('/home/ailaty3088@id.sdsu.edu/SynLLM/diabetes_prediction_dataset_900K.csv')
# synthetic_data = pd.read_csv('/home/ailaty3088@id.sdsu.edu/SynLLM/opensource/synthetic_data.csv')

# # Initialize evaluator
# evaluator = SyntheticDataEvaluator(real_data, synthetic_data)

# # Get comprehensive evaluation
# evaluation_results = evaluator.evaluate_all()

# # Print results
# for metric, results in evaluation_results.items():
#     print(f"\n{metric.upper()}:")
#     print(results)
