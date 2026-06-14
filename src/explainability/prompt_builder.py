"""
LLM Prompt Builder for SentinelIQ.
Constructs context-rich prompts for LLM-based investigation.
"""

from typing import Dict, List
import json


class PromptBuilder:
    """Builds structured prompts for LLM risk analysis."""

    SYSTEM_PROMPT = """You are a Senior Identity Security Analyst at a Fortune 500 company.
You specialize in detecting identity sprawl, privilege abuse, and insider threats.
Analyze the following identity risk finding and provide actionable intelligence.
Return ONLY valid JSON. No markdown, no code blocks."""

    ANALYSIS_TEMPLATE = """
Analyze this identity security finding:

=== USER PROFILE ===
User ID: {user_id}
Username: {username}
Department: {department}
Job Title: {job_title}
Privilege Level: {privilege_level}
Systems Access: {systems_access}
Days Inactive: {days_inactive}
Hire Date: {hire_date}

=== RISK FINDINGS ===
Risk Score: {risk_score}/100
Risk Level: {risk_level}
Rules Triggered: {rules_triggered}
ML Anomaly Detected: {is_anomaly}

=== EVIDENCE ===
{evidence}

=== CONTEXT ===
{context}

Generate a JSON response with:
{{
    "executive_summary": "2-3 sentence summary for CISO",
    "why_risky": "Technical explanation of why this is risky",
    "business_impact": "Potential business impact if not addressed",
    "confidence": "HIGH/MEDIUM/LOW with reasoning",
    "recommended_actions": ["action1", "action2", "action3"],
    "escalation_level": "IMMEDIATE/24H/WEEKLY/MONITOR",
    "compliance_impact": ["NIST AC-2 violation", "GDPR Article 32 risk"],
    "investigation_steps": ["step1", "step2", "step3"]
}}
"""

    def build_analysis_prompt(self, user_data: Dict, findings: List[Dict],
                               risk_score: float, risk_level: str,
                               context: str = "") -> str:
        """
        Build analysis prompt for LLM.
        
        Args:
            user_data: User profile dictionary.
            findings: List of risk findings for this user.
            risk_score: Final risk score.
            risk_level: Risk level string.
            context: Additional context.
            
        Returns:
            Formatted prompt string.
        """
        # Format evidence
        evidence_lines = []
        for finding in findings:
            evidence_lines.append(f"- Rule: {finding.get('rule', 'unknown')}")
            evidence_lines.append(f"  Severity: {finding.get('severity', 'unknown')}")
            evidence_lines.append(f"  Description: {finding.get('description', '')}")
            if finding.get("evidence"):
                evidence_lines.append(f"  Evidence: {json.dumps(finding['evidence'], default=str)}")
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "No specific evidence"

        # Format rules triggered
        rules = list(set(f.get("rule", "") for f in findings))

        prompt = self.ANALYSIS_TEMPLATE.format(
            user_id=user_data.get("user_id", ""),
            username=user_data.get("username", ""),
            department=user_data.get("department", ""),
            job_title=user_data.get("job_title", ""),
            privilege_level=user_data.get("privilege_level", ""),
            systems_access=user_data.get("systems_access", ""),
            days_inactive=user_data.get("days_inactive", 0),
            hire_date=user_data.get("hire_date", ""),
            risk_score=risk_score,
            risk_level=risk_level,
            rules_triggered=", ".join(rules),
            is_anomaly=user_data.get("is_anomaly", False),
            evidence=evidence_text,
            context=context or "No additional context",
        )

        return prompt

    def build_summary_prompt(self, total_users: int, critical_count: int,
                              high_count: int, top_findings: List[Dict]) -> str:
        """Build executive summary prompt."""
        return f"""
Generate an executive security briefing:

Total Users Analyzed: {total_users}
Critical Risk Users: {critical_count}
High Risk Users: {high_count}

Top Findings:
{json.dumps(top_findings[:5], indent=2, default=str)}

Generate JSON:
{{
    "executive_briefing": "3-4 sentence overview for board/CISO",
    "top_risks": ["risk1", "risk2", "risk3"],
    "immediate_actions": ["action1", "action2"],
    "trend_assessment": "Improving/Stable/Declining",
    "resource_recommendation": "Additional resources needed"
}}
"""
