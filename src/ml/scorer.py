"""
ML Scorer Module for SentinelIQ.
Converts ML anomaly outputs into normalized risk scores.
"""

import numpy as np
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class MLScorer:
    """Converts ML model outputs into risk scores."""

    def score(self, anomaly_results: pd.DataFrame) -> pd.DataFrame:
        """
        Convert anomaly detection results to ML risk scores.
        
        Args:
            anomaly_results: DataFrame with anomaly_score column.
            
        Returns:
            DataFrame with ml_risk_score column.
        """
        df = anomaly_results.copy()

        # ML risk score is the normalized anomaly score
        if "anomaly_score" in df.columns:
            df["ml_risk_score"] = df["anomaly_score"]
        else:
            df["ml_risk_score"] = 0.0

        # Boost score for detected anomalies
        if "is_anomaly" in df.columns:
            df.loc[df["is_anomaly"], "ml_risk_score"] = df.loc[
                df["is_anomaly"], "ml_risk_score"
            ].clip(lower=50)

        logger.info(f"ML scoring complete. Mean score: {df['ml_risk_score'].mean():.1f}")
        return df
