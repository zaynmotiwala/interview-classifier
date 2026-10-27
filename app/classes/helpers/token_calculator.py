"""
Token usage calculator for LLM API calls.
Aggregates token counts from multiple modules/calls.
"""
from typing import Dict, Any


def calculate_total_tokens(usage_data: Dict[str, Any]) -> Dict[str, int]:
    """
    Calculate total input/output tokens from usage data.

    Args:
        usage_data: Dictionary containing LM usage from various modules.
                   Expected structure:
                   {
                       "user_id": int,
                       "persona_id": int,
                       "classifier": {"completion_tokens": int, "prompt_tokens": int, ...},
                       "generator": {"completion_tokens": int, "prompt_tokens": int, ...},
                       ...
                   }

    Returns:
        Dictionary with aggregated token counts:
        {
            "input": total_prompt_tokens,
            "output": total_completion_tokens
        }
    """
    total_input = 0
    total_output = 0

    # Iterate through all keys looking for module usage data
    for key, value in usage_data.items():
        # Skip non-module keys (user_id, persona_id, etc.)
        if not isinstance(value, dict):
            continue

        # Extract token counts from this module
        if "prompt_tokens" in value:
            total_input += value.get("prompt_tokens", 0)
        if "completion_tokens" in value:
            total_output += value.get("completion_tokens", 0)

    return {
        "input": total_input,
        "output": total_output
    }
