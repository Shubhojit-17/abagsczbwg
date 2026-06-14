"""
Privilege Escalation Detection Rule.
Identifies potential privilege escalation attempts.
"""

import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PrivilegeEscalationRule:
    """Detect privilege escalation patterns."""

    RULE_NAME = "privilege_escalation"
    WEIGHT = 25
    SEVERITY = "CRITICAL"

    # High-risk actions indicating escalation
    ESCALATION_ACTIONS = {
        "iam_policy_change", "privilege_escalation", "admin_operation"
    }

    def evaluate(self, events_df: pd.DataFrame, users_df: pd.DataFrame) -> List[Dict]:
        """
        Evaluate privilege escalation rule.
        
        Returns:
            List of findings.
        """
        findings = []

        # Find escalation-type actions by non-admin users
        escalation_events = events_df[
            events_df["action"].isin(self.ESCALATION_ACTIONS)
        ]

        # Get non-admin users
        non_admin_users = set(
            users_df[users_df["privilege_level"].isin(["user", "power-user"])]["user_id"]
        )

        for user_id in escalation_events["user_id"].unique():
            user_events = escalation_events[escalation_events["user_id"] == user_id]
            is_non_admin = user_id in non_admin_users

            user_info = users_df[users_df["user_id"] == user_id]
            if user_info.empty:
                continue
            user_info = user_info.iloc[0]

            # Non-admin performing admin operations
            if is_non_admin and len(user_events) > 0:
                findings.append({
                    "user_id": user_id,
                    "username": user_info.get("username", "unknown"),
                    "rule": self.RULE_NAME,
                    "severity": self.SEVERITY,
                    "weight": self.WEIGHT,
                    "score": min(100, 70 + len(user_events) * 10),
                    "description": (
                        f"Non-admin user '{user_info.get('username')}' "
                        f"({user_info.get('privilege_level')}) performed "
                        f"{len(user_events)} administrative/escalation actions."
                    ),
                    "evidence": {
                        "privilege_level": user_info.get("privilege_level", ""),
                        "escalation_actions": user_events["action"].value_counts().to_dict(),
                        "resources_targeted": user_events["resource"].unique().tolist(),
                        "timestamps": user_events["timestamp"].dt.strftime(
                            "%Y-%m-%d %H:%M"
                        ).tolist()[:5],
                    },
                    "recommendation": "URGENT: Investigate immediately. Possible compromised account or insider threat.",
                })

            # Admin with excessive admin operations
            elif not is_non_admin and len(user_events) >= 5:
                findings.append({
                    "user_id": user_id,
                    "username": user_info.get("username", "unknown"),
                    "rule": "excessive_admin_operations",
                    "severity": "HIGH",
                    "weight": self.WEIGHT - 5,
                    "score": min(80, 40 + len(user_events) * 5),
                    "description": (
                        f"Admin '{user_info.get('username')}' performed unusually high "
                        f"number of admin operations ({len(user_events)})."
                    ),
                    "evidence": {
                        "admin_operations_count": len(user_events),
                        "resources": user_events["resource"].unique().tolist(),
                    },
                    "recommendation": "Review if operations were part of authorized change management.",
                })

        logger.info(f"Privilege escalation rule: {len(findings)} findings")
        return findings
