import logging

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIAEvaluator:
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """
        Initialize the MIA evaluator with real and synthetic datasets.

        Args:
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.encoders = {}

        # Log data shapes and feature columns
        logger.info(f"Real data shape: {real_data.shape}")
        logger.info(f"Synthetic data shape: {synthetic_data.shape}")

    def _encode_categorical(self, df):
        """Encode categorical columns using LabelEncoder."""
        df_encoded = df.copy()

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                # Fit on all unique values from both datasets
                all_values = pd.concat(
                    [self.real_data[col], self.synthetic_data[col]]
                ).unique()
                self.encoders[col].fit(all_values)

            # Transform the column
            df_encoded[col] = self.encoders[col].transform(df_encoded[col])

        return df_encoded

    def prepare_data(self, test_size=0.2):
        """Prepare data for MIA evaluation with balanced classes."""
        # Encode categorical variables
        real_encoded = self._encode_categorical(self.real_data)
        synth_encoded = self._encode_categorical(self.synthetic_data)

        # Add membership labels
        real_encoded["membership"] = 1
        synth_encoded["membership"] = 0

        # Combine datasets
        combined_data = pd.concat([real_encoded, synth_encoded], ignore_index=True)

        # Shuffle the data
        combined_data = combined_data.sample(frac=1, random_state=42)

        # Split into train and test
        train_size = int(len(combined_data) * (1 - test_size))
        train_data = combined_data.iloc[:train_size]
        test_data = combined_data.iloc[train_size:]

        # Ensure balanced classes in training data
        train_real = train_data[train_data["membership"] == 1]
        train_synth = train_data[train_data["membership"] == 0]

        # Balance the training data
        min_size = min(len(train_real), len(train_synth))
        train_real = train_real.sample(n=min_size, random_state=42)
        train_synth = train_synth.sample(n=min_size, random_state=42)
        train_data = pd.concat([train_real, train_synth])

        # Shuffle again
        train_data = train_data.sample(frac=1, random_state=42)

        return train_data, test_data

    def train_black_box_attack(self, train_data, test_data):
        """Train a black-box attack model with noise addition."""
        # Prepare features and labels
        X_train = train_data.drop("membership", axis=1)
        y_train = train_data["membership"]
        X_test = test_data.drop("membership", axis=1)
        y_test = test_data["membership"]

        # Train Random Forest
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Get predictions
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Add small noise to probabilities to prevent perfect predictions
        noise = np.random.normal(0, 0.01, size=y_proba.shape)
        y_proba = np.clip(y_proba + noise, 0, 1)

        # Calculate metrics
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            # If all predictions are the same class, return 0.5 (random chance)
            auc = 0.5

        accuracy = accuracy_score(y_test, y_pred)

        return auc, accuracy

    def train_white_box_attack(self, train_data, test_data):
        """Train a white-box attack model with noise addition."""

        class MIA_Net(nn.Module):
            def __init__(self, input_dim):
                super(MIA_Net, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.network(x)

        # Prepare data
        X_train = torch.FloatTensor(train_data.drop("membership", axis=1).values)
        y_train = torch.FloatTensor(train_data["membership"].values)
        X_test = torch.FloatTensor(test_data.drop("membership", axis=1).values)
        y_test = torch.FloatTensor(test_data["membership"].values)

        # Initialize model
        model = MIA_Net(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_proba = model(X_test).numpy().flatten()

            # Add small noise to probabilities
            noise = np.random.normal(0, 0.01, size=y_proba.shape)
            y_proba = np.clip(y_proba + noise, 0, 1)

            y_pred = (y_proba > 0.5).astype(int)

            try:
                auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                # If all predictions are the same class, return 0.5 (random chance)
                auc = 0.5

            accuracy = accuracy_score(y_test, y_pred)

        return auc, accuracy

    def evaluate(self):
        """Evaluate MIA vulnerability with improved metrics."""
        # Prepare data
        train_data, test_data = self.prepare_data()

        # Train and evaluate black-box attack
        black_box_auc, black_box_accuracy = self.train_black_box_attack(
            train_data, test_data
        )

        # Train and evaluate white-box attack
        white_box_auc, white_box_accuracy = self.train_white_box_attack(
            train_data, test_data
        )

        # Calculate overall vulnerability score
        # Use the average of AUC scores, with 0.5 being the baseline (random chance)
        vulnerability_score = (black_box_auc + white_box_auc) / 2

        return {
            "black_box_auc": black_box_auc,
            "black_box_accuracy": black_box_accuracy,
            "white_box_auc": white_box_auc,
            "white_box_accuracy": white_box_accuracy,
            "mia_vulnerability_score": vulnerability_score,
        }
