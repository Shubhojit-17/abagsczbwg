"""
Cross-Department Access Detection Rule.
Identifies users accessing resources outside their department scope.
"""

import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


from src.constants import RESOURCE_DEPARTMENT_MAP, SHARED_RESOURCES


class CrossDepartmentRule:
    """Detect cross-department access violations."""

    RULE_NAME = "cross_department_access"
    WEIGHT = 15
    SEVERITY = "MEDIUM"

    # Use centralized resource-department mapping
    RESOURCE_OWNERSHIP = {
        k: v for k, v in RESOURCE_DEPARTMENT_MAP.items() if v is not None
    }

    def evaluate(self, events_df: pd.DataFrame, users_df: pd.DataFrame) -> List[Dict]:
        """
        Evaluate cross-department access rule.
        
        Returns:
            List of findings.
        """
        findings = []

        # Merge user department into events (exclude username to avoid column collision)
        events_with_dept = events_df.merge(
            users_df[["user_id", "department", "privilege_level"]],
            on="user_id", how="left"
        )

        for user_id in events_with_dept["user_id"].unique():
            user_events = events_with_dept[events_with_dept["user_id"] == user_id]
            if user_events.empty:
                continue

            user_dept = user_events["department"].iloc[0]
            privilege = user_events["privilege_level"].iloc[0]
            username = user_events["username"].iloc[0] if "username" in user_events.columns else user_id

            # Skip admins and IT (they have broader access)
            if privilege == "admin" or user_dept in ["IT", "Security"]:
                continue

            violations = []
            for _, event in user_events.iterrows():
                resource = event.get("resource", "")
                if resource in SHARED_RESOURCES:
                    continue

                expected_dept = self.RESOURCE_OWNERSHIP.get(resource)
                if expected_dept and expected_dept != user_dept:
                    violations.append({
                        "resource": resource,
                        "expected_dept": expected_dept,
                        "timestamp": str(event.get("timestamp", "")),
                        "action": event.get("action", ""),
                    })

            if violations:
                score = min(100, self.WEIGHT + len(violations) * 8)
                severity = "HIGH" if len(violations) >= 3 else self.SEVERITY

                findings.append({
                    "user_id": user_id,
                    "username": username,
                    "rule": self.RULE_NAME,
                    "severity": severity,
                    "weight": self.WEIGHT,
                    "score": score,
                    "description": (
                        f"User '{username}' ({user_dept}) accessed {len(violations)} resources "
                        f"outside department scope."
                    ),
                    "evidence": {
                        "user_department": user_dept,
                        "violations": violations[:5],
                        "total_violations": len(violations),
                    },
                    "recommendation": "Verify business justification for cross-department access.",
                })

        logger.info(f"Cross-department rule: {len(findings)} findings")
        return findings
