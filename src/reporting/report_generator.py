"""
Report Generator for SentinelIQ.
Produces structured risk reports for each user.
"""

import pandas as pd
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive risk reports."""

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_user_report(self, user_id: str, scores_df: pd.DataFrame,
                              findings: List[Dict], llm_analysis: Dict,
                              recommendations: List[Dict],
                              blast_radius: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive report for a single user.
        
        Returns:
            Report dictionary.
        """
        user_row = scores_df[scores_df["user_id"] == user_id]
        if user_row.empty:
            return {"error": f"User {user_id} not found"}

        user = user_row.iloc[0]

        report = {
            "report_id": f"RPT-{user_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "user_profile": {
                "user_id": user_id,
                "username": user.get("username", ""),
                "department": user.get("department", ""),
                "job_title": user.get("job_title", ""),
                "privilege_level": user.get("privilege_level", ""),
            },
            "risk_assessment": {
                "final_risk_score": float(user.get("final_risk_score", 0)),
                "risk_level": user.get("risk_level", "LOW"),
                "confidence": float(user.get("confidence", 0.5)),
                "rule_score": float(user.get("rule_score", 0)),
                "ml_score": float(user.get("ml_risk_score", 0)),
                "context_adjustment": float(user.get("context_adjustment", 0)),
            },
            "findings": [f for f in findings if f.get("user_id") == user_id],
            "llm_analysis": llm_analysis,
            "recommendations": recommendations,
            "blast_radius": blast_radius,
            "compliance_mapping": self._map_compliance(findings, user_id),
        }

        return report

    def generate_executive_summary(self, scores_df: pd.DataFrame,
                                    all_findings: List[Dict]) -> Dict:
        """Generate executive-level summary report."""
        summary = {
            "report_type": "executive_summary",
            "generated_at": datetime.now().isoformat(),
            "overview": {
                "total_users_analyzed": len(scores_df),
                "critical_risk_users": len(scores_df[scores_df["risk_level"] == "CRITICAL"]),
                "high_risk_users": len(scores_df[scores_df["risk_level"] == "HIGH"]),
                "medium_risk_users": len(scores_df[scores_df["risk_level"] == "MEDIUM"]),
                "low_risk_users": len(scores_df[scores_df["risk_level"] == "LOW"]),
                "average_risk_score": round(scores_df["final_risk_score"].mean(), 1),
                "total_findings": len(all_findings),
            },
            "risk_distribution": scores_df["risk_level"].value_counts().to_dict(),
            "department_risk": (
                scores_df.groupby("department")["final_risk_score"]
                .mean().round(1).sort_values(ascending=False).to_dict()
            ),
            "top_risks": (
                scores_df.nlargest(10, "final_risk_score")[
                    ["user_id", "username", "department", "final_risk_score", "risk_level"]
                ].to_dict("records")
            ),
            "findings_by_rule": self._count_findings_by_rule(all_findings),
            "key_metrics": {
                "stale_accounts_pct": self._calc_stale_pct(scores_df, all_findings),
                "over_privileged_pct": self._calc_overprivileged_pct(all_findings, len(scores_df)),
            },
        }

        return summary

    def export_json(self, report: Dict, filename: str) -> str:
        """Export report to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report exported to {filepath}")
        return str(filepath)

    def _map_compliance(self, findings: List[Dict], user_id: str) -> List[Dict]:
        """Map findings to compliance frameworks."""
        user_findings = [f for f in findings if f.get("user_id") == user_id]
        compliance_items = []

        for finding in user_findings:
            rule = finding.get("rule", "")
            if "stale" in rule or "orphaned" in rule:
                compliance_items.append({
                    "framework": "NIST SP 800-53",
                    "control": "AC-2",
                    "description": "Account Management - Inactive account not disabled",
                })
            if "privilege" in rule or "excessive" in rule:
                compliance_items.append({
                    "framework": "NIST SP 800-53",
                    "control": "AC-6",
                    "description": "Least Privilege - Excessive permissions detected",
                })
            if "export" in rule:
                compliance_items.append({
                    "framework": "GDPR",
                    "control": "Article 32",
                    "description": "Security of Processing - Potential data exfiltration",
                })

        return compliance_items

    def _count_findings_by_rule(self, findings: List[Dict]) -> Dict:
        """Count findings by rule type."""
        counts = {}
        for f in findings:
            rule = f.get("rule", "unknown")
            counts[rule] = counts.get(rule, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def _calc_stale_pct(self, scores_df: pd.DataFrame, findings: List[Dict]) -> float:
        """Calculate percentage of stale accounts."""
        stale_users = set(
            f["user_id"] for f in findings
            if "stale" in f.get("rule", "") or "orphaned" in f.get("rule", "")
        )
        return round(len(stale_users) / max(1, len(scores_df)) * 100, 1)

    def _calc_overprivileged_pct(self, findings: List[Dict], total: int) -> float:
        """Calculate percentage of over-privileged accounts."""
        priv_users = set(
            f["user_id"] for f in findings if "privilege" in f.get("rule", "")
        )
        return round(len(priv_users) / max(1, total) * 100, 1)
