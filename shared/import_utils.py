"""Utility for importing config from sibling modules in numbered directories."""

import importlib.util
from pathlib import Path


def load_config(caller_file: str):
    """Load CONFIG from a config.py in the same directory as the caller.

    Usage in any use case script:
        from shared.import_utils import load_config
        CONFIG = load_config(__file__)
    """
    config_path = Path(caller_file).parent / "config.py"
    spec = importlib.util.spec_from_file_location("_use_case_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG
