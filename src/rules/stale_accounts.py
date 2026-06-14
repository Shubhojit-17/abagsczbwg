"""
Stale Accounts Detection Rule.
Identifies accounts that are inactive but still retain privileged access.
"""

import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StaleAccountRule:
    """Detect stale privileged accounts."""

    RULE_NAME = "stale_privileged_account"
    WEIGHT = 25
    SEVERITY = "HIGH"

    def __init__(self, inactive_threshold_admin: int = 30,
                 inactive_threshold_user: int = 60,
                 inactive_threshold_contractor: int = 45):
        self.inactive_threshold_admin = inactive_threshold_admin
        self.inactive_threshold_user = inactive_threshold_user
        self.inactive_threshold_contractor = inactive_threshold_contractor

    def evaluate(self, users_df: pd.DataFrame) -> List[Dict]:
        """
        Evaluate stale account rule.
        
        Args:
            users_df: Users DataFrame with features.
            
        Returns:
            List of findings dictionaries.
        """
        findings = []

        for _, user in users_df.iterrows():
            finding = self._check_user(user)
            if finding:
                findings.append(finding)

        logger.info(f"Stale account rule: {len(findings)} findings")
        return findings

    def _check_user(self, user: pd.Series) -> Optional[Dict]:
        """Check a single user for stale account risk."""
        days_inactive = user.get("days_inactive", 0)
        privilege_level = user.get("privilege_level", "user")
        is_active = user.get("is_active", True)

        # Admin accounts inactive > 30 days
        if privilege_level == "admin" and days_inactive > self.inactive_threshold_admin:
            return {
                "user_id": user["user_id"],
                "username": user.get("username", ""),
                "rule": self.RULE_NAME,
                "severity": "CRITICAL" if days_inactive > 60 else self.SEVERITY,
                "weight": self.WEIGHT,
                "score": min(100, self.WEIGHT + (days_inactive - self.inactive_threshold_admin)),
                "description": (
                    f"Admin account '{user.get('username')}' has been inactive for "
                    f"{days_inactive} days but retains admin privileges across "
                    f"{user.get('systems_count', 0)} systems."
                ),
                "evidence": {
                    "days_inactive": days_inactive,
                    "privilege_level": privilege_level,
                    "systems_count": user.get("systems_count", 0),
                    "last_login": str(user.get("last_login", "")),
                },
                "recommendation": "Immediately revoke admin privileges or confirm continued need with manager.",
            }

        # Power-user accounts inactive > 45 days
        if privilege_level == "power-user" and days_inactive > 45:
            return {
                "user_id": user["user_id"],
                "username": user.get("username", ""),
                "rule": self.RULE_NAME,
                "severity": "HIGH",
                "weight": self.WEIGHT - 5,
                "score": min(80, self.WEIGHT - 5 + (days_inactive - 45)),
                "description": (
                    f"Power-user '{user.get('username')}' inactive for {days_inactive} days "
                    f"with elevated privileges."
                ),
                "evidence": {
                    "days_inactive": days_inactive,
                    "privilege_level": privilege_level,
                },
                "recommendation": "Review access and demote to standard user if not required.",
            }

        # Service accounts inactive > 30 days (potential orphaned)
        if privilege_level == "service-account" and days_inactive > 30:
            return {
                "user_id": user["user_id"],
                "username": user.get("username", ""),
                "rule": "orphaned_service_account",
                "severity": "HIGH",
                "weight": self.WEIGHT,
                "score": min(90, self.WEIGHT + (days_inactive - 30) * 0.5),
                "description": (
                    f"Service account '{user.get('username')}' appears orphaned - "
                    f"inactive for {days_inactive} days."
                ),
                "evidence": {
                    "days_inactive": days_inactive,
                    "privilege_level": privilege_level,
                    "systems": user.get("systems_access", ""),
                },
                "recommendation": "Identify owner. If no owner, disable immediately and schedule deletion.",
            }

        return None
