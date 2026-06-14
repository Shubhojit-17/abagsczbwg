"""
User Feature Engineering for SentinelIQ.
Generates risk-relevant features from user identity data.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class UserFeatureEngine:
    """Generates user-level features for risk analysis."""

    # Privilege level weights
    PRIVILEGE_WEIGHTS = {
        "user": 1,
        "power-user": 3,
        "admin": 5,
        "service-account": 4,
    }

    # Department risk weights
    DEPARTMENT_RISK = {
        "Finance": 8,
        "HR": 7,
        "Security": 6,
        "IT": 6,
        "Executive": 9,
        "Engineering": 5,
        "Compliance": 7,
        "Operations": 4,
        "Sales": 3,
        "Marketing": 2,
        "Legal": 6,
        "Support": 3,
    }

    # High-sensitivity systems
    HIGH_SENSITIVITY_SYSTEMS = {
        "PROD_DB", "ADMIN_SYS", "SIEM", "AWS_IAM", "GCP"
    }

    def generate_features(self, users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive user features.
        
        Args:
            users_df: Users DataFrame.
            events_df: Events DataFrame.
            
        Returns:
            DataFrame with engineered features.
        """
        df = users_df.copy()

        # Base features
        df["privilege_score"] = df["privilege_level"].map(self.PRIVILEGE_WEIGHTS).fillna(1)
        df["department_risk_score"] = df["department"].map(self.DEPARTMENT_RISK).fillna(3)

        # Systems complexity
        df["high_sensitivity_access_count"] = df["systems_list"].apply(
            lambda systems: sum(1 for s in systems if s in self.HIGH_SENSITIVITY_SYSTEMS)
        )

        # Tenure features
        if "hire_date" in df.columns:
            today = pd.Timestamp.now()
            df["tenure_days"] = (today - df["hire_date"]).dt.days
            df["is_new_hire"] = df["tenure_days"] < 30

        # Account type features
        df["is_service_account"] = (
            df["privilege_level"] == "service-account"
        ) | df["username"].str.startswith("svc_", na=False)

        df["is_admin"] = df["privilege_level"] == "admin"

        # Event-based features per user
        event_features = self._compute_event_features(events_df)
        df = df.merge(event_features, on="user_id", how="left")

        # Fill NaN for users with no events
        event_cols = [
            "total_events", "failed_login_count", "after_hours_ratio",
            "night_event_count", "high_sensitivity_event_count",
            "unique_resources", "admin_operations_count", "export_count"
        ]
        for col in event_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        logger.info(f"Generated {len(df.columns)} features for {len(df)} users")
        return df

    def _compute_event_features(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Compute aggregated event features per user."""
        if events_df.empty:
            return pd.DataFrame(columns=["user_id"])

        agg = events_df.groupby("user_id").agg(
            total_events=("action", "count"),
            failed_login_count=("status", lambda x: (x == "failure").sum()),
            night_event_count=("is_night", "sum"),
            high_sensitivity_event_count=(
                "resource_sensitivity", lambda x: (x == "high").sum()
            ),
            unique_resources=("resource", "nunique"),
            admin_operations_count=(
                "action", lambda x: (x == "admin_operation").sum()
            ),
            export_count=("action", lambda x: (x == "export_data").sum()),
        ).reset_index()

        # After-hours ratio
        time_class_counts = events_df.groupby("user_id")["time_classification"].value_counts().unstack(fill_value=0)
        if any(col in time_class_counts.columns for col in ["unusual_hours", "night", "weekend"]):
            unusual = time_class_counts.get("unusual_hours", 0)
            night = time_class_counts.get("night", 0)
            weekend = time_class_counts.get("weekend", 0)
            total = time_class_counts.sum(axis=1)
            after_hours_ratio = ((unusual + night + weekend) / total).fillna(0)
            after_hours_df = after_hours_ratio.reset_index()
            after_hours_df.columns = ["user_id", "after_hours_ratio"]
            agg = agg.merge(after_hours_df, on="user_id", how="left")
        else:
            agg["after_hours_ratio"] = 0.0

        return agg
