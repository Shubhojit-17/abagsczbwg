"""
Banking Calendar Context for Finance Department.
Reduces false positives during known high-activity periods.
"""

import pandas as pd
from datetime import datetime, date
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class BankingCalendar:
    """Context-aware adjustments for financial calendar events."""

    # Month-end days (high activity expected in Finance)
    MONTH_END_DAYS = {28, 29, 30, 31}

    # Quarter-end months
    QUARTER_END_MONTHS = {3, 6, 9, 12}

    # Financial year-end (varies by org, using March/April)
    FISCAL_YEAR_END_MONTHS = {3, 4}

    def apply_calendar_context(self, user: pd.Series,
                                current_score: float,
                                event_date: datetime = None) -> Tuple[float, str]:
        """
        Apply finance calendar adjustments.
        
        Args:
            user: User Series with profile data.
            current_score: Current risk score.
            event_date: Date of the event being analyzed.
            
        Returns:
            Tuple of (adjusted_score, reason).
        """
        department = user.get("department", "")
        if department != "Finance":
            return current_score, "Not finance department"

        if event_date is None:
            event_date = datetime.now()

        adjustment = 0
        reasons = []

        day = event_date.day
        month = event_date.month

        # Month-end: Finance legitimately works late
        if day in self.MONTH_END_DAYS:
            adjustment -= 10
            reasons.append(f"Month-end activity (day {day}): -10")

        # Quarter-end: Even higher activity expected
        if month in self.QUARTER_END_MONTHS and day >= 25:
            adjustment -= 5
            reasons.append(f"Quarter-end period: -5")

        # Fiscal year-end
        if month in self.FISCAL_YEAR_END_MONTHS:
            adjustment -= 5
            reasons.append(f"Fiscal year-end month: -5")

        adjusted_score = max(0, min(100, current_score + adjustment))
        reason = "; ".join(reasons) if reasons else "No calendar adjustments"

        return adjusted_score, reason
