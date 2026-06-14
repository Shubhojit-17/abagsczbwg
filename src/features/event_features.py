"""
Event Feature Engineering for SentinelIQ.
Generates event-level features for anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EventFeatureEngine:
    """Generates event-level features for risk detection."""

    def generate_features(self, events_df: pd.DataFrame,
                          users_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate event-level features.
        
        Args:
            events_df: Events DataFrame.
            users_df: Optional users DataFrame for enrichment.
            
        Returns:
            DataFrame with engineered event features.
        """
        df = events_df.copy()

        # Time-based features
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["is_weekend"] = df["day_of_week"].isin([5, 6])
        df["is_night_access"] = df["hour_of_day"].apply(lambda h: h >= 22 or h <= 5)
        df["is_early_morning"] = df["hour_of_day"].between(0, 6)

        # Sensitivity flags
        df["is_high_sensitivity"] = df["resource_sensitivity"] == "high"
        df["is_admin_action"] = df["action"] == "admin_operation"
        df["is_export"] = df["action"] == "export_data"
        df["is_failed"] = df["status"] == "failure"

        # Cross-department flag (requires user enrichment)
        if users_df is not None:
            df = self._add_cross_department_flag(df, users_df)

        # Action risk scoring
        action_risk = {
            "login": 1,
            "file_access": 2,
            "api_call": 2,
            "sql_query": 3,
            "export_data": 4,
            "admin_operation": 5,
            "iam_policy_change": 8,
            "privilege_escalation": 9,
        }
        df["action_risk_score"] = df["action"].map(action_risk).fillna(2)

        # Sensitivity scoring
        sensitivity_score = {"low": 1, "medium": 3, "high": 5}
        df["sensitivity_score"] = df["resource_sensitivity"].map(sensitivity_score).fillna(1)

        # Combined event risk
        df["event_risk_score"] = (
            df["action_risk_score"] * 0.4 +
            df["sensitivity_score"] * 0.3 +
            df["is_night_access"].astype(int) * 2 +
            df["is_weekend"].astype(int) * 1.5
        )

        logger.info(f"Generated event features for {len(df)} events")
        return df

    def _add_cross_department_flag(self, events_df: pd.DataFrame,
                                    users_df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-department access flags."""
        from ..constants import RESOURCE_DEPARTMENT_MAP

        # Merge user department
        df = events_df.merge(
            users_df[["user_id", "department"]].rename(columns={"department": "user_department"}),
            on="user_id",
            how="left"
        )

        # Map resource to expected department
        df["resource_department"] = df["resource"].map(RESOURCE_DEPARTMENT_MAP)

        # Cross-department flag
        df["is_cross_department"] = (
            df["resource_department"].notna() &
            (df["resource_department"] != df["user_department"])
        )

        return df
