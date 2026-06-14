"""
Contractor Rules for Context Intelligence.
Applies stricter monitoring rules for contractor accounts.
"""

import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ContractorRules:
    """Context rules for contractor identity management."""

    # Contractor indicators
    CONTRACTOR_INDICATORS = {
        "contractor", "temp", "vendor", "external", "consultant"
    }

    # Max inactive days before flagging contractors
    CONTRACTOR_INACTIVE_THRESHOLD = 14

    def is_contractor(self, user: pd.Series) -> bool:
        """Determine if user is likely a contractor."""
        job_title = str(user.get("job_title", "")).lower()
        username = str(user.get("username", "")).lower()
        email = str(user.get("email", "")).lower()

        for indicator in self.CONTRACTOR_INDICATORS:
            if indicator in job_title or indicator in username or indicator in email:
                return True
        return False

    def apply_contractor_context(self, user: pd.Series,
                                  current_score: float) -> Tuple[float, str]:
        """
        Apply contractor-specific risk adjustments.
        
        Args:
            user: User Series with profile data.
            current_score: Current risk score.
            
        Returns:
            Tuple of (adjusted_score, reason).
        """
        if not self.is_contractor(user):
            return current_score, "Not a contractor"

        adjustment = 0
        reasons = []

        days_inactive = user.get("days_inactive", 0)
        privilege_level = user.get("privilege_level", "user")

        # Contractors with admin access are inherently riskier
        if privilege_level in ["admin", "power-user"]:
            adjustment += 15
            reasons.append(f"Contractor with {privilege_level} privileges: +15")

        # Inactive contractors should be flagged sooner
        if days_inactive > self.CONTRACTOR_INACTIVE_THRESHOLD:
            adjustment += 10
            reasons.append(f"Inactive contractor ({days_inactive} days): +10")

        # Contractors accessing many systems
        systems_count = user.get("systems_count", 0)
        if systems_count > 3:
            adjustment += 5
            reasons.append(f"Contractor with {systems_count} system access: +5")

        adjusted_score = max(0, min(100, current_score + adjustment))
        reason = "; ".join(reasons) if reasons else "Standard contractor monitoring"

        return adjusted_score, reason
