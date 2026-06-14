"""
Data Schema Definitions for SentinelIQ.
Defines expected columns, data types, and validation rules.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class UserSchema:
    """Schema definition for identity_users.csv"""
    required_columns: List[str] = field(default_factory=lambda: [
        "user_id", "username", "email", "department", "job_title",
        "privilege_level", "systems_access", "last_login",
        "days_inactive", "is_active", "hire_date"
    ])

    column_types: Dict[str, str] = field(default_factory=lambda: {
        "user_id": "string",
        "username": "string",
        "email": "string",
        "department": "string",
        "job_title": "string",
        "privilege_level": "string",
        "systems_access": "string",
        "last_login": "datetime",
        "days_inactive": "int",
        "is_active": "bool",
        "hire_date": "datetime"
    })

    valid_privilege_levels: List[str] = field(default_factory=lambda: [
        "user", "power-user", "admin", "service-account"
    ])

    valid_departments: List[str] = field(default_factory=lambda: [
        "Engineering", "Finance", "HR", "IT", "Legal", "Marketing",
        "Operations", "Sales", "Security", "Support", "Compliance", "Executive"
    ])


@dataclass
class EventSchema:
    """Schema definition for identity_events.csv"""
    required_columns: List[str] = field(default_factory=lambda: [
        "timestamp", "user_id", "username", "action", "resource",
        "resource_sensitivity", "status", "source_ip", "time_classification"
    ])

    column_types: Dict[str, str] = field(default_factory=lambda: {
        "timestamp": "datetime",
        "user_id": "string",
        "username": "string",
        "action": "string",
        "resource": "string",
        "resource_sensitivity": "string",
        "status": "string",
        "source_ip": "string",
        "time_classification": "string"
    })

    valid_actions: List[str] = field(default_factory=lambda: [
        "login", "file_access", "admin_operation", "export_data",
        "sql_query", "api_call", "iam_policy_change", "privilege_escalation"
    ])

    valid_sensitivity_levels: List[str] = field(default_factory=lambda: [
        "low", "medium", "high"
    ])

    valid_time_classifications: List[str] = field(default_factory=lambda: [
        "business_hours", "unusual_hours", "night", "weekend"
    ])
