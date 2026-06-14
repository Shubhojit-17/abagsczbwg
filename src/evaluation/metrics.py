"""
Evaluation Metrics for SentinelIQ.
Measures detection quality against ground truth labels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics for risk detection."""

    def evaluate(self, predictions_df: pd.DataFrame,
                 risk_threshold: float = 60.0) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Uses the risk scores themselves as the ground truth proxy
        since we're working with the same dataset.
        
        Args:
            predictions_df: DataFrame with final_risk_score and risk_level.
            risk_threshold: Score threshold for binary classification.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Binary classification: HIGH/CRITICAL vs LOW/MEDIUM
        y_pred = (predictions_df["final_risk_score"] >= risk_threshold).astype(int)

        # For self-evaluation, use rule+ML agreement as proxy ground truth
        y_true = self._generate_ground_truth(predictions_df, risk_threshold)

        metrics = {}

        if y_true.sum() > 0 and (1 - y_true).sum() > 0:
            metrics["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 4)
            metrics["recall"] = round(recall_score(y_true, y_pred, zero_division=0), 4)
            metrics["f1_score"] = round(f1_score(y_true, y_pred, zero_division=0), 4)

            try:
                scores_normalized = predictions_df["final_risk_score"] / 100
                metrics["roc_auc"] = round(roc_auc_score(y_true, scores_normalized), 4)
            except Exception:
                metrics["roc_auc"] = 0.0

            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = {
                "true_negatives": int(cm[0][0]),
                "false_positives": int(cm[0][1]),
                "false_negatives": int(cm[1][0]),
                "true_positives": int(cm[1][1]),
            }
        else:
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1_score"] = 0.0
            metrics["roc_auc"] = 0.0

        # Additional metrics
        metrics["total_users"] = len(predictions_df)
        metrics["flagged_users"] = int(y_pred.sum())
        metrics["flag_rate"] = round(y_pred.mean() * 100, 1)
        metrics["risk_distribution"] = predictions_df["risk_level"].value_counts().to_dict()
        metrics["avg_risk_score"] = round(predictions_df["final_risk_score"].mean(), 1)

        # Score distribution stats
        metrics["score_stats"] = {
            "mean": round(predictions_df["final_risk_score"].mean(), 1),
            "median": round(predictions_df["final_risk_score"].median(), 1),
            "std": round(predictions_df["final_risk_score"].std(), 1),
            "min": round(predictions_df["final_risk_score"].min(), 1),
            "max": round(predictions_df["final_risk_score"].max(), 1),
        }

        logger.info(
            f"Evaluation: Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}"
        )

        return metrics

    def _generate_ground_truth(self, df: pd.DataFrame, threshold: float) -> pd.Series:
        """
        Generate ground truth labels based on multiple signal agreement.
        A user is truly risky if both rule engine AND ML agree.
        """
        rule_flag = df.get("rule_score", pd.Series(0, index=df.index)) > 30
        ml_flag = df.get("ml_risk_score", pd.Series(0, index=df.index)) > 40

        # Both signals must agree for ground truth positive
        ground_truth = (rule_flag & ml_flag).astype(int)

        # Also include obvious cases
        if "is_anomaly" in df.columns:
            ground_truth = ground_truth | df["is_anomaly"].astype(int)

        return ground_truth

    def generate_report(self, metrics: Dict) -> str:
        """Generate human-readable evaluation report."""
        report_lines = [
            "=" * 60,
            "SentinelIQ - Detection Quality Report",
            "=" * 60,
            "",
            f"Total Users Analyzed: {metrics.get('total_users', 0)}",
            f"Users Flagged: {metrics.get('flagged_users', 0)} ({metrics.get('flag_rate', 0)}%)",
            "",
            "--- Classification Metrics ---",
            f"Precision: {metrics.get('precision', 0):.4f}",
            f"Recall:    {metrics.get('recall', 0):.4f}",
            f"F1 Score:  {metrics.get('f1_score', 0):.4f}",
            f"ROC-AUC:   {metrics.get('roc_auc', 0):.4f}",
            "",
            "--- Risk Distribution ---",
        ]

        for level, count in metrics.get("risk_distribution", {}).items():
            report_lines.append(f"  {level}: {count}")

        report_lines.extend([
            "",
            "--- Score Statistics ---",
            f"  Mean:   {metrics.get('score_stats', {}).get('mean', 0)}",
            f"  Median: {metrics.get('score_stats', {}).get('median', 0)}",
            f"  Std:    {metrics.get('score_stats', {}).get('std', 0)}",
            f"  Range:  {metrics.get('score_stats', {}).get('min', 0)} - {metrics.get('score_stats', {}).get('max', 0)}",
            "",
            "=" * 60,
        ])

        return "\n".join(report_lines)
