"""
LLM Client for SentinelIQ.
Handles API calls to OpenAI/Claude for risk explanations.
Falls back to rule-based explanations when LLM is unavailable.
"""

import json
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM-based risk analysis and explanation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize LLM Client.
        
        Args:
            api_key: OpenAI API key. Falls back to env var.
            model: Model to use.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.is_available = bool(self.api_key)

        if not self.is_available:
            logger.warning(
                "No LLM API key configured. Using rule-based explanations. "
                "Set OPENAI_API_KEY environment variable for LLM-powered analysis."
            )

    def analyze(self, prompt: str, system_prompt: str = "") -> Dict:
        """
        Send prompt to LLM and get structured analysis.
        
        Falls back to rule-based analysis if LLM unavailable.
        """
        if self.is_available:
            try:
                return self._call_api(prompt, system_prompt)
            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                return self._generate_fallback_analysis(prompt)
        else:
            return self._generate_fallback_analysis(prompt)

    def _call_api(self, prompt: str, system_prompt: str) -> Dict:
        """Call OpenAI API."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
            )

            content = response.choices[0].message.content
            # Try to parse JSON from response
            return json.loads(content)
        except json.JSONDecodeError:
            # If response isn't valid JSON, wrap it
            return {"analysis": content, "raw": True}
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._generate_fallback_analysis(prompt)

    def _generate_fallback_analysis(self, prompt: str) -> Dict:
        """Generate rule-based fallback analysis when LLM is unavailable."""
        # Extract key info from prompt for intelligent fallback
        risk_score = self._extract_risk_score(prompt)
        rules = self._extract_rules(prompt)

        severity = "CRITICAL" if risk_score >= 81 else "HIGH" if risk_score >= 61 else "MEDIUM"

        actions = []
        if "stale" in prompt.lower():
            actions.append("Review account activity and confirm continued need")
            actions.append("Disable account if no business justification within 48 hours")
        if "privilege" in prompt.lower() or "admin" in prompt.lower():
            actions.append("Apply least-privilege principle immediately")
            actions.append("Revoke unnecessary admin/elevated access")
        if "export" in prompt.lower():
            actions.append("Review exported data for sensitive content")
            actions.append("Enable DLP controls on data exports")
        if "night" in prompt.lower() or "hours" in prompt.lower():
            actions.append("Verify if after-hours access was authorized")
            actions.append("Check on-call schedules and shift patterns")

        if not actions:
            actions = [
                "Investigate user activity in the past 30 days",
                "Review access permissions against job requirements",
                "Confirm findings with user's manager",
            ]

        return {
            "executive_summary": (
                f"Identity risk detected (Score: {risk_score}/100, Level: {severity}). "
                f"Rules triggered: {', '.join(rules) if rules else 'Multiple risk indicators'}. "
                f"Immediate review recommended."
            ),
            "why_risky": (
                f"This account shows {len(rules)} risk indicators including "
                f"{', '.join(rules[:3]) if rules else 'anomalous behavior patterns'}. "
                f"Combined signals suggest potential security exposure."
            ),
            "business_impact": (
                "Potential unauthorized access to sensitive systems. "
                "Risk of data exposure, compliance violations, and insider threat."
            ),
            "confidence": f"{severity} - Based on {len(rules)} corroborating signals",
            "recommended_actions": actions,
            "escalation_level": (
                "IMMEDIATE" if risk_score >= 81 else
                "24H" if risk_score >= 61 else "WEEKLY"
            ),
            "compliance_impact": [
                "NIST AC-2: Account Management",
                "GDPR Article 32: Security of Processing",
            ],
            "investigation_steps": [
                "Pull full activity log for past 30 days",
                "Cross-reference with HR records",
                "Check for corresponding tickets/approvals",
                "Verify with direct manager",
            ],
        }

    def _extract_risk_score(self, prompt: str) -> float:
        """Extract risk score from prompt text."""
        import re
        match = re.search(r"Risk Score:\s*(\d+)", prompt)
        return float(match.group(1)) if match else 50.0

    def _extract_rules(self, prompt: str) -> list:
        """Extract triggered rules from prompt text."""
        import re
        match = re.search(r"Rules Triggered:\s*(.+?)(?:\n|$)", prompt)
        if match:
            return [r.strip() for r in match.group(1).split(",") if r.strip()]
        return []
