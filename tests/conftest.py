"""Pytest configuration and fixtures."""
import pytest
import os
import logging
import dspy
from dotenv import load_dotenv

from app.classes.modules.classification.classifier_optimized import RevReplyClassifier

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_classifier():
    """Find the latest optimized classifier model."""
    search_dir = "app/classes/modules/classification"
    optimized_models = []

    if os.path.exists(search_dir):
        for item in os.listdir(search_dir):
            item_path = os.path.join(search_dir, item)
            if os.path.isdir(item_path):
                if item.startswith("revclassifier-optimized-"):
                    optimized_models.append(item)

    if optimized_models:
        latest = sorted(optimized_models)[-1]
        return os.path.join(search_dir, latest)
    else:
        raise FileNotFoundError("No classifier model found")


@pytest.fixture(scope="session")
def classifier():
    """
    Create a reusable classifier instance for testing.
    Loads once per test session to improve performance.
    """
    # Initialize Azure OpenAI LM
    azure_lm = dspy.LM(
        "azure/gpt-5-mini",
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
        temperature=1.0,
        max_tokens=16000,
        cache=False,
    )

    # Find and load classifier
    classifier_path = find_latest_classifier()
    logger.info(f"Using classifier: {classifier_path}")

    try:
        classifier_instance = RevReplyClassifier(
            json_path=classifier_path,
            lm=azure_lm,
            enable_delegation_detection=True
        )
        return classifier_instance
    except FileNotFoundError as e:
        pytest.skip(f"Classifier model not found: {e}")
