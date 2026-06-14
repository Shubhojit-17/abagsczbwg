"""
New Hire Rules for Context Intelligence.
Adjusts risk scoring for newly onboarded employees.
"""

import pandas as pd
from typing import Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class NewHireRules:
    """Context rules for new hire identity risk adjustment."""

    # Days considered "new hire" period
    NEW_HIRE_PERIOD_DAYS = 30

    # Learning period - higher activity expected
    LEARNING_PERIOD_DAYS = 90

    def apply_new_hire_context(self, user: pd.Series,
                                current_score: float) -> Tuple[float, str]:
        """
        Apply new hire context adjustments.
        
        Args:
            user: User Series with profile data.
            current_score: Current risk score.
            
        Returns:
            Tuple of (adjusted_score, reason).
        """
        hire_date = user.get("hire_date")
        if pd.isna(hire_date):
            return current_score, "No hire date available"

        today = pd.Timestamp.now()
        if isinstance(hire_date, str):
            hire_date = pd.Timestamp(hire_date)

        days_since_hire = (today - hire_date).days
        adjustment = 0
        reasons = []

        # Very new hire (< 30 days): Expect lots of access requests/exploration
        if days_since_hire < self.NEW_HIRE_PERIOD_DAYS:
            adjustment -= 15
            reasons.append(f"New hire ({days_since_hire} days): -15")

            # But new hire with admin access IS suspicious
            if user.get("privilege_level") == "admin":
                adjustment += 20
                reasons.append("New hire with admin: +20 (suspicious)")

        # Learning period (30-90 days): Slight reduction
        elif days_since_hire < self.LEARNING_PERIOD_DAYS:
            adjustment -= 5
            reasons.append(f"Learning period ({days_since_hire} days): -5")

        adjusted_score = max(0, min(100, current_score + adjustment))
        reason = "; ".join(reasons) if reasons else "Not a new hire"

        return adjusted_score, reason
