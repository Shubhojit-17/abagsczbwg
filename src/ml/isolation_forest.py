"""
Isolation Forest Anomaly Detection for SentinelIQ.
Detects anomalous user behavior patterns using unsupervised ML.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Isolation Forest based anomaly detection for identity risk."""

    # Features used for anomaly detection
    FEATURE_COLUMNS = [
        "days_inactive",
        "systems_count",
        "privilege_score",
        "high_sensitivity_access_count",
        "department_risk_score",
        "total_events",
        "after_hours_ratio",
        "failed_login_count",
        "night_event_count",
        "high_sensitivity_event_count",
        "unique_resources",
        "admin_operations_count",
        "export_count",
    ]

    def __init__(self, contamination: float = 0.15, random_state: int = 42):
        """
        Initialize Anomaly Detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5).
            random_state: Random seed for reproducibility.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
            max_samples="auto",
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit model and predict anomalies.
        
        Args:
            features_df: DataFrame with user features.
            
        Returns:
            DataFrame with anomaly scores and labels.
        """
        # Select available features
        available_features = [
            col for col in self.FEATURE_COLUMNS if col in features_df.columns
        ]

        if len(available_features) < 3:
            logger.warning("Insufficient features for anomaly detection")
            features_df["anomaly_score"] = 0.0
            features_df["is_anomaly"] = False
            return features_df

        # Prepare feature matrix
        X = features_df[available_features].fillna(0).values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit and predict
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Get predictions (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X_scaled)

        # Get anomaly scores (lower = more anomalous)
        raw_scores = self.model.decision_function(X_scaled)

        # Normalize scores to 0-100 (higher = more anomalous)
        anomaly_scores = self._normalize_scores(raw_scores)

        # Add results to DataFrame
        result_df = features_df.copy()
        result_df["anomaly_score"] = anomaly_scores
        result_df["is_anomaly"] = predictions == -1
        result_df["anomaly_raw_score"] = raw_scores

        # Feature importance (contribution to anomaly score)
        result_df["top_anomaly_features"] = self._get_feature_contributions(
            X_scaled, available_features
        )

        anomaly_count = (predictions == -1).sum()
        logger.info(
            f"Anomaly detection complete: {anomaly_count}/{len(features_df)} anomalies detected "
            f"({anomaly_count/len(features_df)*100:.1f}%)"
        )

        return result_df

    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Normalize Isolation Forest scores to 0-100 scale."""
        # Raw scores: negative = more anomalous
        # We invert and scale to 0-100
        min_score = raw_scores.min()
        max_score = raw_scores.max()

        if max_score == min_score:
            return np.full_like(raw_scores, 50.0)

        # Invert: more anomalous = higher score
        normalized = (max_score - raw_scores) / (max_score - min_score) * 100
        return np.clip(normalized, 0, 100)

    def _get_feature_contributions(self, X_scaled: np.ndarray,
                                    feature_names: List[str]) -> List[str]:
        """Identify top contributing features for each anomaly."""
        contributions = []

        # Simple approach: features furthest from mean
        feature_means = X_scaled.mean(axis=0)

        for i in range(len(X_scaled)):
            deviations = np.abs(X_scaled[i] - feature_means)
            top_indices = deviations.argsort()[-3:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            contributions.append(", ".join(top_features))

        return contributions

    def get_model_params(self) -> dict:
        """Get model parameters for reporting."""
        return {
            "model": "IsolationForest",
            "contamination": self.contamination,
            "n_estimators": 200,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "features_used": self.FEATURE_COLUMNS,
        }
