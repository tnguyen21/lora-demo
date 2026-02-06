"""Parsers for teacher model output across different output formats."""

import json
import re


def parse_single_label(response: str, labels: list[str], regex: str | None = None) -> str | None:
    """Parse a single label from teacher output.

    Tries regex first (if provided), then looks for a label match in the response.
    Returns None if no valid label is found.
    """
    if regex:
        match = re.search(regex, response)
        if match:
            candidate = match.group(1).strip().lower()
            if candidate in [lab.lower() for lab in labels]:
                return candidate

    # Fallback: look for any label appearing in the response
    response_lower = response.strip().lower()
    for label in labels:
        if label.lower() in response_lower:
            return label.lower()

    return None


def parse_json_output(response: str) -> dict | list | None:
    """Parse JSON from teacher output.

    Handles responses that may have text before/after the JSON block.
    """
    # Try direct parse first
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON array or object
    for pattern in [r"(\[.*\])", r"(\{.*\})"]:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


def parse_free_text(response: str, regex: str | None = None) -> str | None:
    """Parse free-text output (e.g., SQL) from teacher response.

    If regex is provided, extracts the first capture group.
    Otherwise returns the stripped response.
    """
    if regex:
        match = re.search(regex, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try to extract from code blocks
    match = re.search(r"```(?:sql)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response.strip() if response.strip() else None


def parse_sql(response: str) -> str | None:
    """Parse SQL from teacher output. Specialized version of parse_free_text."""
    return parse_free_text(response, regex=r"(?:```sql\s*\n?(.*?)\n?\s*```|(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b.*?;)")
