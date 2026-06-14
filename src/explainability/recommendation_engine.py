"""
Recommendation Engine for SentinelIQ.
Generates actionable remediation recommendations.
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generates contextual remediation recommendations."""

    # Recommendation templates by risk type
    RECOMMENDATIONS = {
        "stale_privileged_account": [
            "Immediately suspend admin privileges pending review",
            "Contact user's manager to verify if role is still active",
            "Schedule access recertification within 5 business days",
            "If no response in 48h, disable account per policy",
        ],
        "orphaned_service_account": [
            "Identify service account owner through CMDB lookup",
            "If no owner found, disable account immediately",
            "Audit all access logs for past 90 days",
            "Document in service account registry",
        ],
        "excessive_privileges": [
            "Apply least-privilege principle - remove unused access",
            "Conduct access review with user's manager",
            "Implement just-in-time (JIT) access for elevated permissions",
            "Set up periodic access recertification (quarterly)",
        ],
        "off_hours_admin_activity": [
            "Verify against on-call schedule and change management records",
            "If unauthorized, immediately rotate credentials",
            "Enable MFA step-up for off-hours admin access",
            "Implement session recording for admin sessions",
        ],
        "cross_department_access": [
            "Verify business justification for cross-department access",
            "Implement approval workflow for cross-boundary access",
            "Review if access was granted through proper channels",
            "Consider time-limited access grants",
        ],
        "privilege_escalation": [
            "URGENT: Investigate for potential account compromise",
            "Check for lateral movement indicators",
            "Rotate all credentials immediately",
            "Enable enhanced monitoring on affected systems",
            "Notify SOC for incident response",
        ],
        "service_account_misuse": [
            "Verify automation patterns against known workflows",
            "Check if service account credentials were shared",
            "Implement credential vault for service accounts",
            "Enable API key rotation on 30-day cycle",
        ],
        "bulk_data_export": [
            "CRITICAL: Block further exports immediately",
            "Identify what data was exported and classification",
            "Check DLP logs for data leaving organization",
            "Initiate incident response if sensitive data involved",
            "Preserve all audit logs for forensic analysis",
        ],
    }

    def get_recommendations(self, findings: List[Dict], risk_level: str) -> List[Dict]:
        """
        Generate prioritized recommendations.
        
        Args:
            findings: List of risk findings for a user.
            risk_level: Overall risk level.
            
        Returns:
            List of recommendation dictionaries.
        """
        all_recommendations = []

        for finding in findings:
            rule = finding.get("rule", "")
            severity = finding.get("severity", "MEDIUM")

            # Get specific recommendations
            specific_recs = self.RECOMMENDATIONS.get(rule, [
                "Investigate user activity",
                "Review access permissions",
                "Confirm with manager",
            ])

            all_recommendations.append({
                "rule": rule,
                "severity": severity,
                "actions": specific_recs,
                "priority": self._get_priority(severity, risk_level),
                "sla": self._get_sla(severity),
            })

        # Sort by priority
        all_recommendations.sort(key=lambda x: x["priority"])

        return all_recommendations

    def _get_priority(self, severity: str, risk_level: str) -> int:
        """Calculate priority (1 = highest)."""
        severity_map = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}
        return severity_map.get(severity, 3)

    def _get_sla(self, severity: str) -> str:
        """Get SLA for remediation."""
        sla_map = {
            "CRITICAL": "4 hours",
            "HIGH": "24 hours",
            "MEDIUM": "7 days",
            "LOW": "30 days",
        }
        return sla_map.get(severity, "7 days")
