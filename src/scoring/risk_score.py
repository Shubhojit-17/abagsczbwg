"""
Risk Score Calculation Engine for SentinelIQ.
Combines Rule Engine, ML, and Context scores into final risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from ..context.role_exceptions import RoleExceptionEngine
from ..context.banking_calendar import BankingCalendar
from ..context.contractor_rules import ContractorRules
from ..context.new_hire_rules import NewHireRules

logger = logging.getLogger(__name__)


class RiskScorer:
    """
    Combines multiple risk signals into final risk score.
    
    Formula:
        Final Risk = 40% Rule Score + 35% ML Score + 25% Context Score
    """

    # Risk level thresholds
    RISK_LEVELS = {
        (0, 30): "LOW",
        (31, 60): "MEDIUM",
        (61, 80): "HIGH",
        (81, 100): "CRITICAL",
    }

    # Weights for score components
    RULE_WEIGHT = 0.40
    ML_WEIGHT = 0.35
    CONTEXT_WEIGHT = 0.25

    def __init__(self):
        self.role_exceptions = RoleExceptionEngine()
        self.banking_calendar = BankingCalendar()
        self.contractor_rules = ContractorRules()
        self.new_hire_rules = NewHireRules()

    def calculate_scores(self, features_df: pd.DataFrame,
                         rule_findings: List[Dict],
                         ml_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final risk scores for all users.
        
        Args:
            features_df: User features DataFrame.
            rule_findings: List of rule engine findings.
            ml_scores: DataFrame with ML anomaly scores.
            
        Returns:
            DataFrame with final risk scores and levels.
        """
        # Create score DataFrame
        scores_df = features_df[["user_id", "username", "department",
                                  "job_title", "privilege_level"]].copy()

        # Calculate rule scores per user
        rule_scores = self._aggregate_rule_scores(rule_findings)
        scores_df = scores_df.merge(rule_scores, on="user_id", how="left")
        scores_df["rule_score"] = scores_df["rule_score"].fillna(0)

        # Merge ML scores
        if "ml_risk_score" in ml_scores.columns:
            scores_df = scores_df.merge(
                ml_scores[["user_id", "ml_risk_score", "is_anomaly"]],
                on="user_id", how="left"
            )
        else:
            scores_df["ml_risk_score"] = 0.0
            scores_df["is_anomaly"] = False

        scores_df["ml_risk_score"] = scores_df["ml_risk_score"].fillna(0)

        # Calculate context scores
        context_scores = []
        context_reasons = []
        for _, user in features_df.iterrows():
            ctx_score, ctx_reason = self._calculate_context_score(user)
            context_scores.append(ctx_score)
            context_reasons.append(ctx_reason)

        scores_df["context_adjustment"] = context_scores
        scores_df["context_reason"] = context_reasons

        # Calculate final score
        scores_df["final_risk_score"] = (
            scores_df["rule_score"] * self.RULE_WEIGHT +
            scores_df["ml_risk_score"] * self.ML_WEIGHT +
            (50 + scores_df["context_adjustment"]) * self.CONTEXT_WEIGHT
        ).clip(0, 100).round(1)

        # Assign risk levels
        scores_df["risk_level"] = scores_df["final_risk_score"].apply(self._get_risk_level)

        # Add confidence score
        scores_df["confidence"] = self._calculate_confidence(scores_df, rule_findings)

        logger.info(
            f"Risk scoring complete. Distribution: "
            f"CRITICAL={len(scores_df[scores_df['risk_level']=='CRITICAL'])}, "
            f"HIGH={len(scores_df[scores_df['risk_level']=='HIGH'])}, "
            f"MEDIUM={len(scores_df[scores_df['risk_level']=='MEDIUM'])}, "
            f"LOW={len(scores_df[scores_df['risk_level']=='LOW'])}"
        )

        return scores_df

    def _aggregate_rule_scores(self, findings: List[Dict]) -> pd.DataFrame:
        """Aggregate multiple rule findings per user into single score."""
        if not findings:
            return pd.DataFrame(columns=["user_id", "rule_score", "rules_triggered"])

        findings_df = pd.DataFrame(findings)
        user_scores = findings_df.groupby("user_id").agg(
            rule_score=("score", "max"),  # Take highest rule score
            rules_triggered=("rule", lambda x: list(x.unique())),
            finding_count=("rule", "count"),
        ).reset_index()

        # Boost score based on number of rules triggered
        user_scores["rule_score"] = (
            user_scores["rule_score"] + user_scores["finding_count"] * 3
        ).clip(0, 100)

        return user_scores[["user_id", "rule_score", "rules_triggered"]]

    def _calculate_context_score(self, user: pd.Series) -> Tuple[float, str]:
        """Calculate context-aware score adjustment for a user."""
        total_adjustment = 0
        all_reasons = []

        # Apply role exceptions
        role_score, role_reason = self.role_exceptions.apply_exceptions(user, 50)
        adj = role_score - 50
        total_adjustment += adj
        if "No role" not in role_reason:
            all_reasons.append(role_reason)

        # Apply new hire rules
        hire_score, hire_reason = self.new_hire_rules.apply_new_hire_context(user, 50)
        adj = hire_score - 50
        total_adjustment += adj
        if "Not a new" not in hire_reason and "No hire" not in hire_reason:
            all_reasons.append(hire_reason)

        # Apply contractor rules
        contractor_score, contractor_reason = self.contractor_rules.apply_contractor_context(user, 50)
        adj = contractor_score - 50
        total_adjustment += adj
        if "Not a contractor" not in contractor_reason:
            all_reasons.append(contractor_reason)

        # Apply banking calendar
        cal_score, cal_reason = self.banking_calendar.apply_calendar_context(user, 50)
        adj = cal_score - 50
        total_adjustment += adj
        if "Not finance" not in cal_reason and "No calendar" not in cal_reason:
            all_reasons.append(cal_reason)

        reason = "; ".join(all_reasons) if all_reasons else "No context adjustments"
        return total_adjustment, reason

    def _get_risk_level(self, score: float) -> str:
        """Map score to risk level."""
        if score >= 81:
            return "CRITICAL"
        elif score >= 61:
            return "HIGH"
        elif score >= 31:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_confidence(self, scores_df: pd.DataFrame,
                               findings: List[Dict]) -> pd.Series:
        """Calculate confidence score for each risk assessment."""
        confidence = pd.Series(0.5, index=scores_df.index)

        # Higher confidence when multiple signals agree
        for idx, row in scores_df.iterrows():
            signals = 0
            if row.get("rule_score", 0) > 30:
                signals += 1
            if row.get("ml_risk_score", 0) > 40:
                signals += 1
            if row.get("is_anomaly", False):
                signals += 1

            # More signals = higher confidence
            confidence.iloc[idx] = min(0.95, 0.5 + signals * 0.15)

        return confidence.round(2)
