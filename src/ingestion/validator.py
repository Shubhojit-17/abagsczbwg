"""
Data Validator Module for SentinelIQ.
Validates data integrity, schema compliance, and handles data quality issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from .schemas import UserSchema, EventSchema

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates identity data against defined schemas."""

    def __init__(self):
        self.user_schema = UserSchema()
        self.event_schema = EventSchema()
        self.validation_report: Dict = {}

    def validate_users(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate users DataFrame against schema.
        
        Args:
            df: Users DataFrame to validate.
            
        Returns:
            Tuple of (cleaned DataFrame, validation report).
        """
        report = {
            "total_records": len(df),
            "issues": [],
            "dropped_records": 0,
            "fixed_records": 0,
        }

        # Check required columns
        missing_cols = set(self.user_schema.required_columns) - set(df.columns)
        if missing_cols:
            report["issues"].append(f"Missing columns: {missing_cols}")
            logger.warning(f"Missing columns in users data: {missing_cols}")

        # Check for duplicates
        duplicates = df.duplicated(subset=["user_id"], keep="first")
        if duplicates.any():
            count = duplicates.sum()
            report["issues"].append(f"Duplicate user_ids: {count}")
            report["dropped_records"] += count
            df = df[~duplicates].copy()
            logger.warning(f"Removed {count} duplicate users")

        # Handle missing values
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                report["issues"].append(f"Null values in {col}: {count}")

        # Fill missing departments
        if "department" in df.columns:
            df["department"] = df["department"].fillna("Unknown")

        # Fill missing privilege levels
        if "privilege_level" in df.columns:
            df["privilege_level"] = df["privilege_level"].fillna("user")

        # Validate date ranges
        if "last_login" in df.columns:
            invalid_dates = df["last_login"].isna()
            if invalid_dates.any():
                report["issues"].append(f"Invalid last_login dates: {invalid_dates.sum()}")
                report["fixed_records"] += invalid_dates.sum()

        # Validate privilege levels
        if "privilege_level" in df.columns:
            invalid_privs = ~df["privilege_level"].isin(self.user_schema.valid_privilege_levels)
            if invalid_privs.any():
                report["issues"].append(
                    f"Invalid privilege levels: {df[invalid_privs]['privilege_level'].unique().tolist()}"
                )

        report["valid_records"] = len(df)
        self.validation_report["users"] = report
        logger.info(f"Users validation complete: {report['valid_records']} valid records")
        return df, report

    def validate_events(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate events DataFrame against schema.
        
        Args:
            df: Events DataFrame to validate.
            
        Returns:
            Tuple of (cleaned DataFrame, validation report).
        """
        report = {
            "total_records": len(df),
            "issues": [],
            "dropped_records": 0,
            "fixed_records": 0,
        }

        # Check required columns
        missing_cols = set(self.event_schema.required_columns) - set(df.columns)
        if missing_cols:
            report["issues"].append(f"Missing columns: {missing_cols}")

        # Check for invalid timestamps
        if "timestamp" in df.columns:
            invalid_ts = df["timestamp"].isna()
            if invalid_ts.any():
                count = invalid_ts.sum()
                report["issues"].append(f"Invalid timestamps: {count}")
                report["dropped_records"] += count
                df = df[~invalid_ts].copy()

        # Validate sensitivity levels
        if "resource_sensitivity" in df.columns:
            df["resource_sensitivity"] = df["resource_sensitivity"].fillna("low")

        # Validate time_classification
        if "time_classification" in df.columns:
            df["time_classification"] = df["time_classification"].fillna("business_hours")

        # Validate status
        if "status" in df.columns:
            df["status"] = df["status"].fillna("unknown")

        report["valid_records"] = len(df)
        self.validation_report["events"] = report
        logger.info(f"Events validation complete: {report['valid_records']} valid records")
        return df, report

    def get_data_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)."""
        if not self.validation_report:
            return 0.0

        scores = []
        for key, report in self.validation_report.items():
            if report["total_records"] > 0:
                valid_ratio = report["valid_records"] / report["total_records"]
                scores.append(valid_ratio * 100)

        return np.mean(scores) if scores else 0.0
