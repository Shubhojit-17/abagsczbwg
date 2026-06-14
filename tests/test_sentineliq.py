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


class TestOffHoursActivity:
    """Tests for off-hours activity detection."""

    def test_admin_night_activity(self):
        from src.rules.off_hours_activity import OffHoursActivityRule
        rule = OffHoursActivityRule()

        users = pd.DataFrame({
            "user_id": ["USR001", "USR002"],
            "username": ["admin1", "user1"],
            "privilege_level": ["admin", "user"],
        })

        events = pd.DataFrame({
            "user_id": ["USR001"] * 3 + ["USR002"],
            "username": ["admin1"] * 3 + ["user1"],
            "action": ["file_access"] * 4,
            "resource": ["PROD_DB"] * 4,
            "resource_sensitivity": ["high"] * 4,
            "status": ["success"] * 4,
            "timestamp": pd.to_datetime([
                "2024-01-01 02:00", "2024-01-02 03:00",
                "2024-01-03 01:00", "2024-01-01 14:00"
            ]),
            "time_classification": ["night", "night", "night", "business_hours"],
            "hour": [2, 3, 1, 14],
            "is_night": [True, True, True, False],
        })

        findings = rule.evaluate(events, users)
        assert len(findings) >= 1
        assert any(f["user_id"] == "USR001" for f in findings)

    def test_weekend_activity_detected(self):
        from src.rules.off_hours_activity import OffHoursActivityRule
        rule = OffHoursActivityRule()

        users = pd.DataFrame({
            "user_id": ["USR001"],
            "username": ["admin1"],
            "privilege_level": ["admin"],
        })

        events = pd.DataFrame({
            "user_id": ["USR001"] * 3,
            "username": ["admin1"] * 3,
            "action": ["file_access"] * 3,
            "resource": ["PROD_DB"] * 3,
            "resource_sensitivity": ["high"] * 3,
            "status": ["success"] * 3,
            "timestamp": pd.to_datetime([
                "2024-01-06 10:00", "2024-01-07 11:00", "2024-01-13 09:00"
            ]),
            "time_classification": ["weekend", "weekend", "weekend"],
            "hour": [10, 11, 9],
            "is_night": [False, False, False],
        })

        findings = rule.evaluate(events, users)
        assert len(findings) >= 1


class TestCrossDepartmentRule:
    """Tests for cross-department access detection."""

    def test_cross_dept_access(self):
        from src.rules.cross_department import CrossDepartmentRule
        rule = CrossDepartmentRule()

        users = pd.DataFrame({
            "user_id": ["USR001", "USR002"],
            "username": ["sales_user", "hr_user"],
            "department": ["Sales", "HR"],
            "privilege_level": ["user", "user"],
        })

        events = pd.DataFrame({
            "user_id": ["USR001", "USR001", "USR001", "USR002"],
            "username": ["sales_user", "sales_user", "sales_user", "hr_user"],
            "action": ["file_access", "sql_query", "api_call", "file_access"],
            "resource": ["HRIS", "GL_System", "SIEM", "HRIS"],
            "resource_sensitivity": ["high", "high", "medium", "high"],
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
            "time_classification": ["business_hours"] * 4,
        })

        findings = rule.evaluate(events, users)
        # Sales user accessing HR, Finance, and Security resources
        assert len(findings) >= 1
        assert any(f["user_id"] == "USR001" for f in findings)

    def test_shared_resources_not_flagged(self):
        from src.rules.cross_department import CrossDepartmentRule
        rule = CrossDepartmentRule()

        users = pd.DataFrame({
            "user_id": ["USR001"],
            "username": ["user1"],
            "department": ["Sales"],
            "privilege_level": ["user"],
        })

        events = pd.DataFrame({
            "user_id": ["USR001", "USR001"],
            "username": ["user1", "user1"],
            "action": ["file_access", "file_access"],
            "resource": ["File_Share", "Email_Archive"],
            "resource_sensitivity": ["low", "medium"],
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
            "time_classification": ["business_hours"] * 2,
        })

        findings = rule.evaluate(events, users)
        assert len(findings) == 0


class TestPrivilegeEscalationRule:
    """Tests for privilege escalation detection."""

    def test_escalation_detected(self):
        from src.rules.privilege_escalation import PrivilegeEscalationRule
        rule = PrivilegeEscalationRule()

        users = pd.DataFrame({
            "user_id": ["USR001"],
            "username": ["user1"],
            "privilege_level": ["user"],
        })

        events = pd.DataFrame({
            "user_id": ["USR001"] * 4,
            "username": ["user1"] * 4,
            "action": ["admin_operation", "admin_operation", "admin_operation", "admin_operation"],
            "resource": ["PROD_DB", "ADMIN_SYS", "AWS_IAM", "SIEM"],
            "resource_sensitivity": ["high", "high", "high", "medium"],
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
            "time_classification": ["business_hours"] * 4,
        })

        findings = rule.evaluate(events, users)
        assert len(findings) >= 1
        assert findings[0]["user_id"] == "USR001"


class TestServiceAccountRule:
    """Tests for service account misuse detection."""

    def test_interactive_login_detected(self):
        from src.rules.service_accounts import ServiceAccountRule
        rule = ServiceAccountRule()

        users = pd.DataFrame({
            "user_id": ["USR001", "USR002"],
            "username": ["svc_bot", "normal_user"],
            "privilege_level": ["service-account", "user"],
        })

        events = pd.DataFrame({
            "user_id": ["USR001", "USR001", "USR002"],
            "username": ["svc_bot", "svc_bot", "normal_user"],
            "action": ["login", "file_access", "login"],
            "resource": ["VPN", "PROD_DB", "VPN"],
            "resource_sensitivity": ["low", "high", "low"],
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "time_classification": ["business_hours"] * 3,
            "source_ip": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
        })

        findings = rule.evaluate(events, users)
        assert len(findings) >= 1
        assert any(f["user_id"] == "USR001" for f in findings)


class TestStaleAccountReturnType:
    """Test that stale account returns None for non-stale users."""

    def test_returns_none_for_normal_user(self):
        from src.rules.stale_accounts import StaleAccountRule
        rule = StaleAccountRule()

        user = pd.Series({
            "user_id": "USR001",
            "username": "normal_user",
            "privilege_level": "user",
            "days_inactive": 5,
            "is_active": True,
            "systems_count": 2,
            "last_login": "2024-03-01",
            "systems_access": "AD",
        })

        result = rule._check_user(user)
        assert result is None  # Not empty dict


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_analyze_not_run(self, client):
        """Test summary endpoint before analysis."""
        response = client.get("/summary")
        assert response.status_code == 400

    def test_findings_not_run(self, client):
        """Test findings endpoint before analysis."""
        response = client.get("/findings")
        assert response.status_code == 400

    def test_feedback_submit(self, client):
        """Test feedback submission."""
        response = client.post("/feedback", json={
            "user_id": "USR001",
            "action": "approve",
            "reason": "Verified legitimate access"
        })
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_feedback_get(self, client):
        """Test feedback retrieval."""
        response = client.get("/feedback")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
