"""
Unit Tests for SentinelIQ.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestDataLoader:
    """Tests for data ingestion module."""

    def test_loader_initialization(self):
        from src.ingestion.loader import DataLoader
        loader = DataLoader("data")
        assert loader.data_dir.name == "data"

    def test_load_users(self):
        from src.ingestion.loader import DataLoader
        loader = DataLoader("data")
        try:
            df = loader.load_users()
            assert "user_id" in df.columns
            assert "username" in df.columns
            assert "privilege_level" in df.columns
            assert "systems_count" in df.columns
            assert len(df) > 0
        except FileNotFoundError:
            pytest.skip("Data file not available")

    def test_load_events(self):
        from src.ingestion.loader import DataLoader
        loader = DataLoader("data")
        try:
            df = loader.load_events()
            assert "timestamp" in df.columns
            assert "user_id" in df.columns
            assert "action" in df.columns
            assert "hour" in df.columns
            assert len(df) > 0
        except FileNotFoundError:
            pytest.skip("Data file not available")


class TestValidator:
    """Tests for data validation module."""

    def test_validate_users(self):
        from src.ingestion.validator import DataValidator
        validator = DataValidator()

        # Create test DataFrame
        df = pd.DataFrame({
            "user_id": ["USR001", "USR002", "USR001"],  # duplicate
            "username": ["test1", "test2", "test1"],
            "email": ["a@b.com", "c@d.com", "a@b.com"],
            "department": ["IT", None, "IT"],  # null
            "job_title": ["Dev", "Admin", "Dev"],
            "privilege_level": ["user", "admin", "user"],
            "systems_access": ["AD", "AWS", "AD"],
            "last_login": [datetime.now(), None, datetime.now()],
            "days_inactive": [5, 30, 5],
            "is_active": [True, True, True],
            "hire_date": [datetime.now(), datetime.now(), datetime.now()],
        })

        cleaned, report = validator.validate_users(df)
        assert report["dropped_records"] == 1  # duplicate removed
        assert len(cleaned) == 2


class TestRuleEngine:
    """Tests for rule-based detection."""

    def test_stale_account_detection(self):
        from src.rules.stale_accounts import StaleAccountRule
        rule = StaleAccountRule()

        users = pd.DataFrame({
            "user_id": ["USR001", "USR002", "USR003"],
            "username": ["admin1", "user1", "svc_bot"],
            "privilege_level": ["admin", "user", "service-account"],
            "days_inactive": [45, 10, 35],
            "is_active": [True, True, True],
            "systems_count": [5, 2, 3],
            "last_login": ["2024-01-01", "2024-03-01", "2024-02-01"],
            "systems_access": ["AD|AWS", "AD", "PROD_DB"],
        })

        findings = rule.evaluate(users)
        assert len(findings) >= 2  # admin and service account should be flagged
        assert any(f["user_id"] == "USR001" for f in findings)

    def test_excessive_privileges_detection(self):
        from src.rules.excessive_privileges import ExcessivePrivilegesRule
        rule = ExcessivePrivilegesRule()

        users = pd.DataFrame({
            "user_id": ["USR001", "USR002"],
            "username": ["over_priv", "normal"],
            "privilege_level": ["user", "user"],
            "systems_count": [8, 2],
            "job_title": ["Coordinator", "Developer"],
            "high_sensitivity_access_count": [3, 0],
        })

        findings = rule.evaluate(users)
        assert len(findings) >= 1
        assert findings[0]["user_id"] == "USR001"

    def test_bulk_export_detection(self):
        from src.rules.bulk_export import BulkExportRule
        rule = BulkExportRule(export_threshold=2)

        events = pd.DataFrame({
            "user_id": ["USR001"] * 3 + ["USR002"],
            "username": ["exporter"] * 3 + ["normal"],
            "action": ["export_data"] * 3 + ["login"],
            "resource": ["Data_Lake", "HRIS", "GL_System", "VPN"],
            "resource_sensitivity": ["high", "high", "medium", "low"],
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
            "time_classification": ["business_hours"] * 4,
        })

        users = pd.DataFrame({
            "user_id": ["USR001", "USR002"],
            "username": ["exporter", "normal"],
        })

        findings = rule.evaluate(events, users)
        assert len(findings) == 1
        assert findings[0]["user_id"] == "USR001"


class TestMLDetection:
    """Tests for ML anomaly detection."""

    def test_isolation_forest(self):
        from src.ml.isolation_forest import AnomalyDetector

        detector = AnomalyDetector(contamination=0.2)

        # Create test features
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "user_id": [f"USR{i:03d}" for i in range(n)],
            "days_inactive": np.random.randint(0, 100, n),
            "systems_count": np.random.randint(1, 10, n),
            "privilege_score": np.random.choice([1, 3, 5], n),
            "high_sensitivity_access_count": np.random.randint(0, 5, n),
            "department_risk_score": np.random.randint(2, 9, n),
            "total_events": np.random.randint(0, 50, n),
            "after_hours_ratio": np.random.random(n),
            "failed_login_count": np.random.randint(0, 10, n),
            "night_event_count": np.random.randint(0, 10, n),
            "high_sensitivity_event_count": np.random.randint(0, 10, n),
            "unique_resources": np.random.randint(1, 8, n),
            "admin_operations_count": np.random.randint(0, 5, n),
            "export_count": np.random.randint(0, 5, n),
        })

        result = detector.fit_predict(df)
        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns
        assert result["anomaly_score"].between(0, 100).all()
        assert detector.is_fitted


class TestRiskScoring:
    """Tests for risk scoring."""

    def test_risk_levels(self):
        from src.scoring.risk_score import RiskScorer
        scorer = RiskScorer()

        assert scorer._get_risk_level(90) == "CRITICAL"
        assert scorer._get_risk_level(70) == "HIGH"
        assert scorer._get_risk_level(50) == "MEDIUM"
        assert scorer._get_risk_level(20) == "LOW"


class TestContextIntelligence:
    """Tests for context-aware adjustments."""

    def test_role_exceptions(self):
        from src.context.role_exceptions import RoleExceptionEngine
        engine = RoleExceptionEngine()

        cto = pd.Series({"job_title": "CTO", "department": "Executive"})
        score, reason = engine.apply_exceptions(cto, 80)
        assert score < 80  # Should reduce score for CTO

    def test_new_hire_exception(self):
        from src.context.new_hire_rules import NewHireRules
        rules = NewHireRules()

        new_hire = pd.Series({
            "hire_date": pd.Timestamp.now() - pd.Timedelta(days=10),
            "privilege_level": "user",
        })
        score, reason = rules.apply_new_hire_context(new_hire, 60)
        assert score < 60  # Should reduce for new hire

    def test_contractor_rules(self):
        from src.context.contractor_rules import ContractorRules
        rules = ContractorRules()

        contractor = pd.Series({
            "job_title": "contractor",
            "username": "ext_john",
            "email": "john@vendor.com",
            "privilege_level": "admin",
            "days_inactive": 20,
            "systems_count": 5,
        })
        assert rules.is_contractor(contractor)
        score, reason = rules.apply_contractor_context(contractor, 50)
        assert score > 50  # Should increase for contractor with admin


class TestPrivilegeGraph:
    """Tests for privilege graph."""

    def test_graph_construction(self):
        from src.graph.privilege_graph import PrivilegeGraph
        graph = PrivilegeGraph()

        users = pd.DataFrame({
            "user_id": ["USR001", "USR002"],
            "username": ["admin1", "user1"],
            "department": ["IT", "Finance"],
            "privilege_level": ["admin", "user"],
            "systems_access": ["AD|AWS_IAM|PROD_DB", "AD"],
        })

        events = pd.DataFrame({
            "user_id": ["USR001", "USR001", "USR002"],
            "resource": ["HRIS", "GL_System", "File_Share"],
            "timestamp": pd.date_range("2024-01-01", periods=3),
        })

        G = graph.build_graph(users, events)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_blast_radius(self):
        from src.graph.privilege_graph import PrivilegeGraph
        graph = PrivilegeGraph()

        users = pd.DataFrame({
            "user_id": ["USR001"],
            "username": ["admin1"],
            "department": ["IT"],
            "privilege_level": ["admin"],
            "systems_access": ["AD|AWS_IAM|PROD_DB|SIEM"],
        })
        events = pd.DataFrame(columns=["user_id", "resource", "timestamp"])

        graph.build_graph(users, events)
        blast = graph.get_blast_radius("USR001")
        assert blast["blast_radius_score"] > 0
        assert len(blast["systems_at_risk"]) > 0


class TestEvaluation:
    """Tests for evaluation metrics."""

    def test_metrics_calculation(self):
        from src.evaluation.metrics import MetricsCalculator
        calc = MetricsCalculator()

        df = pd.DataFrame({
            "final_risk_score": [85, 72, 45, 30, 90, 20, 65, 15, 80, 10],
            "risk_level": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "CRITICAL",
                          "LOW", "HIGH", "LOW", "HIGH", "LOW"],
            "rule_score": [70, 50, 20, 10, 80, 5, 40, 0, 60, 0],
            "ml_risk_score": [80, 60, 30, 15, 85, 10, 55, 5, 70, 5],
            "is_anomaly": [True, True, False, False, True, False, True, False, True, False],
        })

        metrics = calc.evaluate(df)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert metrics["total_users"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
