"""
SentinelIQ Main Pipeline - Core Analysis Engine.
Orchestrates all components for end-to-end risk analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator
from src.features.user_features import UserFeatureEngine
from src.features.event_features import EventFeatureEngine
from src.features.aggregations import FeatureAggregator
from src.rules.stale_accounts import StaleAccountRule
from src.rules.excessive_privileges import ExcessivePrivilegesRule
from src.rules.off_hours_activity import OffHoursActivityRule
from src.rules.cross_department import CrossDepartmentRule
from src.rules.privilege_escalation import PrivilegeEscalationRule
from src.rules.service_accounts import ServiceAccountRule
from src.rules.bulk_export import BulkExportRule
from src.ml.isolation_forest import AnomalyDetector
from src.ml.scorer import MLScorer
from src.scoring.risk_score import RiskScorer
from src.graph.privilege_graph import PrivilegeGraph
from src.explainability.prompt_builder import PromptBuilder
from src.explainability.llm_client import LLMClient
from src.explainability.recommendation_engine import RecommendationEngine
from src.reporting.report_generator import ReportGenerator
from src.evaluation.metrics import MetricsCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SentinelIQ")


class SentinelIQPipeline:
    """Main orchestration pipeline for SentinelIQ analysis."""

    def __init__(self, data_dir: str = "data", output_dir: str = "outputs"):
        """
        Initialize pipeline with all components.
        
        Args:
            data_dir: Path to data directory.
            output_dir: Path for output files.
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = DataLoader(data_dir)
        self.validator = DataValidator()
        self.user_feature_engine = UserFeatureEngine()
        self.event_feature_engine = EventFeatureEngine()
        self.aggregator = FeatureAggregator()
        self.anomaly_detector = AnomalyDetector(contamination=0.15)
        self.ml_scorer = MLScorer()
        self.risk_scorer = RiskScorer()
        self.privilege_graph = PrivilegeGraph()
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient()
        self.recommendation_engine = RecommendationEngine()
        self.report_generator = ReportGenerator(str(self.output_dir / "reports"))
        self.metrics_calculator = MetricsCalculator()

        # Pipeline state
        self.users_df: Optional[pd.DataFrame] = None
        self.events_df: Optional[pd.DataFrame] = None
        self.user_features: Optional[pd.DataFrame] = None
        self.event_features: Optional[pd.DataFrame] = None
        self.rule_findings: List[Dict] = []
        self.combined_features: Optional[pd.DataFrame] = None
        self.scores_df: Optional[pd.DataFrame] = None
        self.graph: Optional[object] = None

    def run_full_pipeline(self, users_path: Optional[str] = None,
                           events_path: Optional[str] = None) -> Dict:
        """
        Execute the complete analysis pipeline.
        
        Returns:
            Dictionary with all pipeline results.
        """
        logger.info("=" * 60)
        logger.info("SentinelIQ Analysis Pipeline - Starting")
        logger.info("=" * 60)

        # Step 1: Data Ingestion
        logger.info("Step 1: Loading data...")
        self.users_df, self.events_df = self.loader.load_all(users_path, events_path)

        # Step 2: Data Validation
        logger.info("Step 2: Validating data...")
        self.users_df, user_report = self.validator.validate_users(self.users_df)
        self.events_df, event_report = self.validator.validate_events(self.events_df)
        data_quality = self.validator.get_data_quality_score()
        logger.info(f"Data quality score: {data_quality:.1f}%")

        # Step 3: Feature Engineering
        logger.info("Step 3: Engineering features...")
        self.user_features = self.user_feature_engine.generate_features(
            self.users_df, self.events_df
        )
        self.event_features = self.event_feature_engine.generate_features(
            self.events_df, self.users_df
        )

        # Step 3b: Aggregate user + event features into combined matrix
        logger.info("Step 3b: Aggregating features...")
        self.combined_features = self.aggregator.aggregate(
            self.user_features, self.event_features
        )

        # Step 4: Rule Engine
        logger.info("Step 4: Running rule engine...")
        self.rule_findings = self._run_rule_engine()

        # Step 5: ML Anomaly Detection
        logger.info("Step 5: Running anomaly detection...")
        ml_results = self.anomaly_detector.fit_predict(self.combined_features)
        ml_scored = self.ml_scorer.score(ml_results)

        # Step 6: Risk Scoring
        logger.info("Step 6: Calculating risk scores...")
        self.scores_df = self.risk_scorer.calculate_scores(
            self.combined_features, self.rule_findings, ml_scored
        )

        # Step 7: Build Privilege Graph
        logger.info("Step 7: Building privilege graph...")
        self.graph = self.privilege_graph.build_graph(self.users_df, self.events_df)

        # Step 8: Generate Reports
        logger.info("Step 8: Generating reports...")
        executive_summary = self.report_generator.generate_executive_summary(
            self.scores_df, self.rule_findings
        )

        # Step 9: Evaluate
        logger.info("Step 9: Evaluating detection quality...")
        eval_metrics = self.metrics_calculator.evaluate(self.scores_df)
        eval_report = self.metrics_calculator.generate_report(eval_metrics)
        logger.info("\n" + eval_report)

        # Step 10: Export Results
        logger.info("Step 10: Exporting results...")
        self._export_results(executive_summary, eval_metrics)

        # Final summary
        results = {
            "data_quality": data_quality,
            "total_users": len(self.users_df),
            "total_events": len(self.events_df),
            "total_findings": len(self.rule_findings),
            "risk_distribution": self.scores_df["risk_level"].value_counts().to_dict(),
            "executive_summary": executive_summary,
            "evaluation_metrics": eval_metrics,
            "graph_stats": self.privilege_graph.get_graph_stats(),
        }

        logger.info("=" * 60)
        logger.info("SentinelIQ Analysis Pipeline - Complete")
        logger.info(f"Results: {results['total_findings']} findings across {results['total_users']} users")
        logger.info("=" * 60)

        return results

    def _run_rule_engine(self) -> List[Dict]:
        """Execute all detection rules."""
        findings = []

        # Rule 1: Stale Accounts
        stale_rule = StaleAccountRule()
        findings.extend(stale_rule.evaluate(self.user_features))

        # Rule 2: Excessive Privileges
        priv_rule = ExcessivePrivilegesRule()
        findings.extend(priv_rule.evaluate(self.user_features))

        # Rule 3: Off-Hours Activity
        hours_rule = OffHoursActivityRule()
        findings.extend(hours_rule.evaluate(self.events_df, self.users_df))

        # Rule 4: Cross-Department Access
        cross_rule = CrossDepartmentRule()
        findings.extend(cross_rule.evaluate(self.events_df, self.users_df))

        # Rule 5: Privilege Escalation
        esc_rule = PrivilegeEscalationRule()
        findings.extend(esc_rule.evaluate(self.events_df, self.users_df))

        # Rule 6: Service Account Misuse
        svc_rule = ServiceAccountRule()
        findings.extend(svc_rule.evaluate(self.events_df, self.users_df))

        # Rule 7: Bulk Export
        export_rule = BulkExportRule()
        findings.extend(export_rule.evaluate(self.events_df, self.users_df))

        logger.info(f"Rule engine complete: {len(findings)} total findings")
        return findings

    def get_user_investigation(self, user_id: str) -> Dict:
        """
        Get full investigation report for a specific user.
        
        Args:
            user_id: User to investigate.
            
        Returns:
            Complete investigation report.
        """
        if self.scores_df is None:
            return {"error": "Pipeline not yet executed. Run run_full_pipeline() first."}

        # Get user data
        user_row = self.scores_df[self.scores_df["user_id"] == user_id]
        if user_row.empty:
            return {"error": f"User {user_id} not found"}

        user = user_row.iloc[0]
        user_findings = [f for f in self.rule_findings if f.get("user_id") == user_id]

        # Get LLM analysis
        user_data = user.to_dict()
        prompt = self.prompt_builder.build_analysis_prompt(
            user_data, user_findings,
            user.get("final_risk_score", 0),
            user.get("risk_level", "LOW")
        )
        llm_analysis = self.llm_client.analyze(prompt, self.prompt_builder.SYSTEM_PROMPT)

        # Get recommendations
        recommendations = self.recommendation_engine.get_recommendations(
            user_findings, user.get("risk_level", "LOW")
        )

        # Get blast radius
        blast_radius = self.privilege_graph.get_blast_radius(user_id)

        # Generate full report
        report = self.report_generator.generate_user_report(
            user_id, self.scores_df, self.rule_findings,
            llm_analysis, recommendations, blast_radius
        )

        return report

    def get_top_risks(self, n: int = 10) -> List[Dict]:
        """Get top N highest risk users."""
        if self.scores_df is None:
            return []

        top = self.scores_df.nlargest(n, "final_risk_score")
        results = []
        for _, row in top.iterrows():
            results.append({
                "user_id": row["user_id"],
                "username": row.get("username", ""),
                "department": row.get("department", ""),
                "risk_score": float(row["final_risk_score"]),
                "risk_level": row["risk_level"],
                "confidence": float(row.get("confidence", 0.5)),
            })
        return results

    def _export_results(self, summary: Dict, metrics: Dict):
        """Export all results to files."""
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Export executive summary
        self.report_generator.export_json(
            summary, "executive_summary.json"
        )

        # Export scores
        if self.scores_df is not None:
            scores_path = reports_dir / "risk_scores.csv"
            self.scores_df.to_csv(scores_path, index=False)

        # Export findings
        findings_path = reports_dir / "all_findings.json"
        with open(findings_path, "w") as f:
            json.dump(self.rule_findings, f, indent=2, default=str)

        # Export metrics
        metrics_path = reports_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Export graph data
        graph_path = reports_dir / "privilege_graph.json"
        with open(graph_path, "w") as f:
            json.dump(self.privilege_graph.export_for_visualization(), f, indent=2)

        logger.info(f"All results exported to {reports_dir}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SentinelIQ Identity Risk Analysis")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--output-dir", default="outputs", help="Output directory path")
    args = parser.parse_args()

    pipeline = SentinelIQPipeline(data_dir=args.data_dir, output_dir=args.output_dir)
    results = pipeline.run_full_pipeline()

    print("\n" + "=" * 60)
    print("SENTINELIQ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Users Analyzed: {results['total_users']}")
    print(f"Events Processed: {results['total_events']}")
    print(f"Findings: {results['total_findings']}")
    print(f"Risk Distribution: {results['risk_distribution']}")
    print(f"Data Quality: {results['data_quality']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
