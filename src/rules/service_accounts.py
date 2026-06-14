"""
Service Account Misuse Detection Rule.
Identifies service accounts with anomalous behavior patterns.
"""

import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ServiceAccountRule:
    """Detect service account misuse."""

    RULE_NAME = "service_account_misuse"
    WEIGHT = 20
    SEVERITY = "HIGH"

    def evaluate(self, events_df: pd.DataFrame, users_df: pd.DataFrame) -> List[Dict]:
        """
        Evaluate service account misuse rule.
        
        Returns:
            List of findings.
        """
        findings = []

        # Identify service accounts
        service_accounts = users_df[
            (users_df["privilege_level"] == "service-account") |
            (users_df["username"].str.startswith("svc_", na=False))
        ]

        for _, svc_account in service_accounts.iterrows():
            user_id = svc_account["user_id"]
            user_events = events_df[events_df["user_id"] == user_id]

            issues = []

            # Service account with interactive login
            logins = user_events[user_events["action"] == "login"]
            if len(logins) > 0:
                issues.append(f"Interactive logins detected ({len(logins)})")

            # Service account accessing non-typical resources
            if len(user_events) > 0:
                sensitive_events = user_events[user_events["resource_sensitivity"] == "high"]
                if len(sensitive_events) > 3:
                    issues.append(f"Excessive high-sensitivity access ({len(sensitive_events)})")

            # Service account active during unusual hours
            unusual_events = user_events[
                user_events["time_classification"].isin(["unusual_hours", "night"])
            ]
            # Note: Service accounts often run 24/7, so unusual hours may be normal
            # Only flag if combined with other issues

            # Service account with data exports
            exports = user_events[user_events["action"] == "export_data"]
            if len(exports) > 0:
                issues.append(f"Data export operations ({len(exports)})")

            if issues:
                score = min(100, self.WEIGHT + len(issues) * 15)
                findings.append({
                    "user_id": user_id,
                    "username": svc_account.get("username", ""),
                    "rule": self.RULE_NAME,
                    "severity": "CRITICAL" if len(issues) >= 2 else self.SEVERITY,
                    "weight": self.WEIGHT,
                    "score": score,
                    "description": (
                        f"Service account '{svc_account.get('username')}' shows "
                        f"anomalous behavior: {'; '.join(issues)}"
                    ),
                    "evidence": {
                        "issues": issues,
                        "total_events": len(user_events),
                        "systems_access": svc_account.get("systems_access", ""),
                    },
                    "recommendation": "Verify service account owner. Check if behavior matches expected automation patterns.",
                })

        logger.info(f"Service account rule: {len(findings)} findings")
        return findings
