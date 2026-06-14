"""
Excessive Privileges Detection Rule.
Identifies accounts with more access than their role requires.
"""

import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ExcessivePrivilegesRule:
    """Detect over-privileged accounts."""

    RULE_NAME = "excessive_privileges"
    WEIGHT = 20
    SEVERITY = "HIGH"

    # Expected max systems by privilege level
    MAX_SYSTEMS_BY_LEVEL = {
        "user": 3,
        "power-user": 5,
        "admin": 7,
        "service-account": 3,
    }

    # Roles that should NOT have admin access
    NON_ADMIN_ROLES = {
        "Coordinator", "Specialist", "Officer", "Developer"
    }

    def evaluate(self, users_df: pd.DataFrame) -> List[Dict]:
        """
        Evaluate excessive privileges rule.
        
        Returns:
            List of findings.
        """
        findings = []

        for _, user in users_df.iterrows():
            finding = self._check_user(user)
            if finding:
                findings.append(finding)

        logger.info(f"Excessive privileges rule: {len(findings)} findings")
        return findings

    def _check_user(self, user: pd.Series) -> Dict:
        """Check a single user for excessive privilege risk."""
        privilege_level = user.get("privilege_level", "user")
        systems_count = user.get("systems_count", 0)
        job_title = user.get("job_title", "")
        high_sens_count = user.get("high_sensitivity_access_count", 0)

        max_systems = self.MAX_SYSTEMS_BY_LEVEL.get(privilege_level, 3)
        issues = []

        # Check if user has more systems than expected
        if systems_count > max_systems:
            issues.append(f"Has access to {systems_count} systems (expected max {max_systems})")

        # Check if non-admin role has admin privilege
        if privilege_level == "admin" and job_title in self.NON_ADMIN_ROLES:
            issues.append(f"Job title '{job_title}' has admin privileges")

        # Check high sensitivity access for low-privilege users
        if privilege_level == "user" and high_sens_count >= 2:
            issues.append(f"Standard user has access to {high_sens_count} high-sensitivity systems")

        if not issues:
            return {}

        score = min(100, self.WEIGHT + len(issues) * 10 + systems_count * 3)

        return {
            "user_id": user["user_id"],
            "username": user.get("username", ""),
            "rule": self.RULE_NAME,
            "severity": "CRITICAL" if len(issues) >= 2 else self.SEVERITY,
            "weight": self.WEIGHT,
            "score": score,
            "description": (
                f"Account '{user.get('username')}' ({privilege_level}) has excessive privileges: "
                f"{'; '.join(issues)}"
            ),
            "evidence": {
                "privilege_level": privilege_level,
                "systems_count": systems_count,
                "high_sensitivity_count": high_sens_count,
                "job_title": job_title,
                "issues": issues,
            },
            "recommendation": "Apply least-privilege principle. Remove unnecessary system access.",
        }
