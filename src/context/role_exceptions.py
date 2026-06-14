"""
Role Exceptions for Context Intelligence.
Handles legitimate elevated access for specific roles.
"""

import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RoleExceptionEngine:
    """Applies context-aware adjustments based on role exceptions."""

    # Roles with inherently broad access (reduce false positives)
    BROAD_ACCESS_ROLES = {
        "CTO": -20,
        "CISO": -18,
        "VP": -15,
        "Director": -10,
        "Executive": -12,
    }

    # Job titles that justify admin operations
    ADMIN_JUSTIFIED_TITLES = {
        "Administrator", "System Administrator", "DBA",
        "DevOps Engineer", "Security Engineer", "IT Manager"
    }

    def apply_exceptions(self, user: pd.Series, current_score: float) -> Tuple[float, str]:
        """
        Apply role-based score adjustments.
        
        Args:
            user: User Series with profile data.
            current_score: Current risk score.
            
        Returns:
            Tuple of (adjusted_score, reason).
        """
        adjustment = 0
        reasons = []
        job_title = user.get("job_title", "")
        department = user.get("department", "")

        # CTO/Executive exception
        for role_keyword, score_adj in self.BROAD_ACCESS_ROLES.items():
            if role_keyword.lower() in job_title.lower():
                adjustment += score_adj
                reasons.append(f"Executive role ({job_title}): {score_adj}")
                break

        # IT/Security department has broader legitimate access
        if department in ["IT", "Security"]:
            adjustment -= 8
            reasons.append(f"{department} department: -8")

        # Admin-justified title
        if job_title in self.ADMIN_JUSTIFIED_TITLES:
            adjustment -= 10
            reasons.append(f"Admin-justified title ({job_title}): -10")

        adjusted_score = max(0, min(100, current_score + adjustment))
        reason = "; ".join(reasons) if reasons else "No role exceptions applied"

        return adjusted_score, reason
