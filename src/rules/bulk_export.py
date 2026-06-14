"""
Bulk Export Detection Rule.
Identifies suspicious data exfiltration attempts.
"""

import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class BulkExportRule:
    """Detect potential data exfiltration via bulk exports."""

    RULE_NAME = "bulk_data_export"
    WEIGHT = 30
    SEVERITY = "CRITICAL"

    def __init__(self, export_threshold: int = 3):
        """
        Args:
            export_threshold: Number of export events to trigger alert.
        """
        self.export_threshold = export_threshold

    def evaluate(self, events_df: pd.DataFrame, users_df: pd.DataFrame) -> List[Dict]:
        """
        Evaluate bulk export rule.
        
        Returns:
            List of findings.
        """
        findings = []

        # Filter export events
        export_events = events_df[events_df["action"] == "export_data"]

        # Group by user
        user_exports = export_events.groupby("user_id")

        for user_id, user_export_events in user_exports:
            if len(user_export_events) < self.export_threshold:
                continue

            user_info = users_df[users_df["user_id"] == user_id]
            if user_info.empty:
                continue
            user_info = user_info.iloc[0]

            # Check for high-sensitivity exports
            high_sens_exports = user_export_events[
                user_export_events["resource_sensitivity"] == "high"
            ]

            # Check if exports happened in short timeframe
            timestamps = user_export_events["timestamp"].sort_values()
            if len(timestamps) >= 2:
                time_span = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600
            else:
                time_span = 24

            # Night/weekend exports are more suspicious
            night_exports = user_export_events[
                user_export_events["time_classification"].isin(["night", "unusual_hours", "weekend"])
            ]

            score = min(100, self.WEIGHT + len(user_export_events) * 10)
            if len(high_sens_exports) > 0:
                score += 15
            if len(night_exports) > 0:
                score += 10

            score = min(100, score)

            findings.append({
                "user_id": user_id,
                "username": user_info.get("username", "unknown"),
                "rule": self.RULE_NAME,
                "severity": self.SEVERITY if score >= 70 else "HIGH",
                "weight": self.WEIGHT,
                "score": score,
                "description": (
                    f"User '{user_info.get('username')}' performed {len(user_export_events)} "
                    f"data exports ({len(high_sens_exports)} high-sensitivity) "
                    f"within {time_span:.1f} hours."
                ),
                "evidence": {
                    "export_count": len(user_export_events),
                    "high_sensitivity_exports": len(high_sens_exports),
                    "night_exports": len(night_exports),
                    "resources": user_export_events["resource"].unique().tolist(),
                    "time_span_hours": time_span,
                },
                "recommendation": "URGENT: Potential data exfiltration. Block account and investigate immediately.",
            })

        logger.info(f"Bulk export rule: {len(findings)} findings")
        return findings
