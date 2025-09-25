"""
Main module for car price prediction using multinomial logistic regression.
"""

from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class CarPricePredictor:
    """
    A class for predicting car prices using multinomial logistic regression.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the car price predictor.

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            random_state=random_state,
            max_iter=1000,
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.is_fitted = False

    def preprocess_data(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for training or prediction.

        Args:
            df: Input dataframe
            target_column: Name of the target column

        Returns:
            Tuple of features and target arrays
        """
        # Separate features and target
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()

        # Encode categorical features
        for column in X.columns:
            if X[column].dtype == "object":
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    X[column] = self.label_encoders[column].fit_transform(X[column])
                else:
                    X[column] = self.label_encoders[column].transform(X[column])

        # Encode target
        if not self.is_fitted:
            y = self.target_encoder.fit_transform(y)
        else:
            y = self.target_encoder.transform(y)

        # Scale features
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def fit(self, df: pd.DataFrame, target_column: str) -> "CarPricePredictor":
        """
        Train the model on the provided data.

        Args:
            df: Training dataframe
            target_column: Name of the target column

        Returns:
            Self for method chaining
        """
        X, y = self.preprocess_data(df, target_column)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: Input dataframe for prediction

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Create dummy target column for preprocessing
        df_copy = df.copy()
        df_copy["_dummy_target"] = self.target_encoder.classes_[0]

        X, _ = self.preprocess_data(df_copy, "_dummy_target")
        predictions = self.model.predict(X)

        # Decode predictions back to original labels
        return self.target_encoder.inverse_transform(predictions)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            df: Input dataframe for prediction

        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Create dummy target column for preprocessing
        df_copy = df.copy()
        df_copy["_dummy_target"] = self.target_encoder.classes_[0]

        X, _ = self.preprocess_data(df_copy, "_dummy_target")
        return self.model.predict_proba(X)

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "target_encoder": self.target_encoder,
            "random_state": self.random_state,
        }
        joblib.dump(model_data, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "CarPricePredictor":
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded CarPricePredictor instance
        """
        model_data = joblib.load(filepath)

        predictor = cls(random_state=model_data["random_state"])
        predictor.model = model_data["model"]
        predictor.scaler = model_data["scaler"]
        predictor.label_encoders = model_data["label_encoders"]
        predictor.target_encoder = model_data["target_encoder"]
        predictor.is_fitted = True

        return predictor
