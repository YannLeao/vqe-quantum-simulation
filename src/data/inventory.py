from pathlib import Path
from typing import Optional

from src.data.paths import get_data_dir


def classify_data_file(path: Path, data_dir: Optional[Path] = None) -> str:
    """Classify a file in the project data directory by cache/data kind."""
    root = get_data_dir(data_dir)
    rel = path.relative_to(root)
    parts = rel.parts

    if parts[0] == "noise_cache":
        return "noise_cache"

    if len(parts) >= 4:
        section = parts[2]
        if section == "vqe_cache":
            return "vqe_cache"
        if section == "figures":
            return "figure"
        if section == "reference":
            return "reference"

    if path.name.startswith("fci_"):
        return "fci_cache"

    return "other"


def summarize_data_dir(data_dir: Optional[Path] = None) -> list[dict]:
    """Return a compact inventory of files stored under data/."""
    root = get_data_dir(data_dir)
    summary: dict[tuple[str, str, str, str], dict] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(root)
        parts = rel.parts
        kind = classify_data_file(path, root)
        molecule = "" if kind == "noise_cache" else parts[0] if parts else ""
        basis = "" if kind == "noise_cache" else parts[1] if len(parts) > 1 else ""
        extension = path.suffix.lower() or "<none>"
        key = (molecule, basis, kind, extension)

        item = summary.setdefault(
            key,
            {
                "molecule": molecule,
                "basis": basis,
                "kind": kind,
                "extension": extension,
                "count": 0,
                "bytes": 0,
            },
        )
        item["count"] += 1
        item["bytes"] += path.stat().st_size

    return list(summary.values())
