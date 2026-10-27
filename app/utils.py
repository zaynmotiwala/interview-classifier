"""Utility functions for the classifier."""
import re


def clean_phone_number(phone: str) -> str:
    """
    Clean and format a phone number.
    Removes non-numeric characters and formats consistently.
    """
    if not phone:
        return ""

    # Remove all non-numeric characters
    digits = re.sub(r'\D', '', phone)

    # Return formatted phone number
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    else:
        return digits  # Return as-is if format is unclear
