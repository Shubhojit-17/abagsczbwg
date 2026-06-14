"""
Data Loader Module for SentinelIQ.
Handles CSV loading with error handling, encoding detection, and schema awareness.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads identity data from CSV files with robust error handling."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to directory containing CSV files.
        """
        self.data_dir = Path(data_dir)

    def load_users(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load identity_users.csv with proper type handling.
        
        Args:
            filepath: Optional custom path to users CSV.
            
        Returns:
            DataFrame with user identity data.
        """
        path = Path(filepath) if filepath else self.data_dir / "identity_users.csv"
        logger.info(f"Loading users data from {path}")

        try:
            df = pd.read_csv(
                path,
                parse_dates=["last_login", "hire_date"],
                dtype={
                    "user_id": str,
                    "username": str,
                    "email": str,
                    "department": str,
                    "job_title": str,
                    "privilege_level": str,
                    "systems_access": str,
                    "is_active": str,
                }
            )

            # Convert is_active to boolean
            df["is_active"] = df["is_active"].map(
                {"true": True, "True": True, "false": False, "False": False}
            ).fillna(True)

            # Parse days_inactive as numeric
            df["days_inactive"] = pd.to_numeric(df["days_inactive"], errors="coerce").fillna(0).astype(int)

            # Parse systems_access into list
            df["systems_list"] = df["systems_access"].fillna("").str.split("|")
            df["systems_count"] = df["systems_list"].apply(len)

            logger.info(f"Loaded {len(df)} users successfully")
            return df

        except FileNotFoundError:
            logger.error(f"Users file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            raise

    def load_events(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load identity_events.csv with proper type handling.
        
        Args:
            filepath: Optional custom path to events CSV.
            
        Returns:
            DataFrame with identity event data.
        """
        path = Path(filepath) if filepath else self.data_dir / "identity_events.csv"
        logger.info(f"Loading events data from {path}")

        try:
            df = pd.read_csv(
                path,
                parse_dates=["timestamp"],
                dtype={
                    "user_id": str,
                    "username": str,
                    "action": str,
                    "resource": str,
                    "resource_sensitivity": str,
                    "status": str,
                    "source_ip": str,
                    "time_classification": str,
                }
            )

            # Extract time features
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6])
            df["is_night"] = df["hour"].apply(lambda h: h >= 22 or h <= 5)

            logger.info(f"Loaded {len(df)} events successfully")
            return df

        except FileNotFoundError:
            logger.error(f"Events file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading events: {e}")
            raise

    def load_all(self, users_path: Optional[str] = None,
                 events_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both users and events data.
        
        Returns:
            Tuple of (users_df, events_df)
        """
        users_df = self.load_users(users_path)
        events_df = self.load_events(events_path)
        return users_df, events_df
