"""
Feature Aggregation Module for SentinelIQ.
Combines user and event features into final feature matrix.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """Aggregates all features into unified risk feature matrix."""

    def aggregate(self, user_features: pd.DataFrame,
                  event_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create unified feature matrix for risk scoring.
        
        Args:
            user_features: User-level features DataFrame.
            event_features: Event-level features DataFrame.
            
        Returns:
            Unified feature matrix.
        """
        # Aggregate event features to user level
        agg_dict = {
            "avg_event_risk": ("event_risk_score", "mean"),
            "max_event_risk": ("event_risk_score", "max"),
            "total_high_sensitivity_events": ("is_high_sensitivity", "sum"),
            "total_night_events": ("is_night_access", "sum"),
            "total_admin_actions": ("is_admin_action", "sum"),
            "total_exports": ("is_export", "sum"),
            "unique_ips": ("source_ip", "nunique"),
            "event_span_days": ("timestamp", lambda x: (x.max() - x.min()).days if len(x) > 1 else 0),
        }

        # Only include cross_dept_events if the column exists
        if "is_cross_department" in event_features.columns:
            agg_dict["cross_dept_events"] = ("is_cross_department", "sum")

        event_agg = event_features.groupby("user_id").agg(**agg_dict).reset_index()

        # Add cross_dept_events as 0 if it wasn't in the data
        if "cross_dept_events" not in event_agg.columns:
            event_agg["cross_dept_events"] = 0

        # Merge with user features
        combined = user_features.merge(event_agg, on="user_id", how="left")

        # Fill NaN values
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        combined[numeric_cols] = combined[numeric_cols].fillna(0)

        logger.info(f"Aggregated features: {combined.shape[1]} columns for {len(combined)} users")
        return combined
