"""
Off-Hours Activity Detection Rule.
Identifies suspicious after-hours administrative activity.
"""

import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class OffHoursActivityRule:
    """Detect suspicious after-hours activity."""

    RULE_NAME = "off_hours_admin_activity"
    WEIGHT = 15
    SEVERITY = "MEDIUM"

    def __init__(self, night_start: int = 22, night_end: int = 5):
        self.night_start = night_start
        self.night_end = night_end

    def evaluate(self, events_df: pd.DataFrame, users_df: pd.DataFrame) -> List[Dict]:
        """
        Evaluate off-hours activity rule.
        
        Args:
            events_df: Events DataFrame.
            users_df: Users DataFrame.
            
        Returns:
            List of findings.
        """
        findings = []

        # Get admin users
        admin_users = set(users_df[users_df["privilege_level"] == "admin"]["user_id"])

        # Filter night/unusual/weekend events
        night_events = events_df[
            (events_df["time_classification"].isin(["night", "unusual_hours", "weekend"])) |
            (events_df["is_night"])
        ]

        # Check admin users with night activity
        for user_id in night_events["user_id"].unique():
            user_night_events = night_events[night_events["user_id"] == user_id]
            is_admin = user_id in admin_users

            # Admin + night + sensitive resource = high risk
            high_sens_night = user_night_events[
                user_night_events["resource_sensitivity"] == "high"
            ]

            if is_admin and len(high_sens_night) > 0:
                user_info = users_df[users_df["user_id"] == user_id].iloc[0] if len(
                    users_df[users_df["user_id"] == user_id]) > 0 else pd.Series()

                findings.append({
                    "user_id": user_id,
                    "username": user_info.get("username", "unknown"),
                    "rule": self.RULE_NAME,
                    "severity": "HIGH",
                    "weight": self.WEIGHT + 10,
                    "score": min(100, 60 + len(high_sens_night) * 10),
                    "description": (
                        f"Admin '{user_info.get('username', user_id)}' performed "
                        f"{len(high_sens_night)} high-sensitivity operations during off-hours."
                    ),
                    "evidence": {
                        "night_events_count": len(user_night_events),
                        "high_sensitivity_night_events": len(high_sens_night),
                        "actions": user_night_events["action"].value_counts().to_dict(),
                        "resources": user_night_events["resource"].unique().tolist(),
                        "timestamps": user_night_events["timestamp"].dt.strftime(
                            "%Y-%m-%d %H:%M"
                        ).tolist()[:5],
                    },
                    "recommendation": "Verify if after-hours access was authorized. Check for on-call schedule.",
                })

            # Non-admin but high off-hours ratio
            elif not is_admin and len(user_night_events) >= 3:
                user_info = users_df[users_df["user_id"] == user_id].iloc[0] if len(
                    users_df[users_df["user_id"] == user_id]) > 0 else pd.Series()

                findings.append({
                    "user_id": user_id,
                    "username": user_info.get("username", "unknown"),
                    "rule": "unusual_hours_activity",
                    "severity": "MEDIUM",
                    "weight": self.WEIGHT,
                    "score": min(70, 30 + len(user_night_events) * 5),
                    "description": (
                        f"User '{user_info.get('username', user_id)}' has {len(user_night_events)} "
                        f"events during unusual hours."
                    ),
                    "evidence": {
                        "night_events_count": len(user_night_events),
                        "actions": user_night_events["action"].value_counts().to_dict(),
                    },
                    "recommendation": "Review access patterns. Confirm if shift work or timezone difference.",
                })

        logger.info(f"Off-hours rule: {len(findings)} findings")
        return findings
