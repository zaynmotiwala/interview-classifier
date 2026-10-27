"""
Simplified logging utility for the interview project.
In production, this would log to Elasticsearch/monitoring systems.
"""
import logging
from datetime import datetime
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)


def log_usage_data_background(usage_data: Dict[str, Any]):
    """
    Log usage data in the background (simplified version).
    In production, this sends to Elasticsearch/metrics system.
    """
    def _log():
        logger.info(f"[USAGE] {usage_data}")

    _executor.submit(_log)


def log_custom_index(data: Dict[str, Any], index_name: str):
    """
    Log custom data to a specific index (simplified version).
    In production, this sends to Elasticsearch.
    """
    def _log():
        logger.info(f"[{index_name.upper()}] {data}")

    _executor.submit(_log)
