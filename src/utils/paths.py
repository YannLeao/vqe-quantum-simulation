from pathlib import Path

def get_project_root() -> Path:
    """Return the repository root inferred from this module location."""
    return Path(__file__).resolve().parents[2]
