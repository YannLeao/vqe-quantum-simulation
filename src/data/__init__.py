from data.cache import cache_fci, deduplicate_fci_cache
from src.data.inventory import classify_data_file, summarize_data_dir
from src.data.paths import (
    get_data_dir,
    get_fci_cache_dir,
    get_figure_dir,
    get_molecule_basis_dir,
    get_noise_cache_dir,
    get_reference_data_dir,
    get_vqe_cache_dir,
)

__all__ = [
    "cache_fci",
    "classify_data_file",
    "deduplicate_fci_cache",
    "get_data_dir",
    "get_fci_cache_dir",
    "get_figure_dir",
    "get_molecule_basis_dir",
    "get_noise_cache_dir",
    "get_reference_data_dir",
    "get_vqe_cache_dir",
    "summarize_data_dir",
]


def __getattr__(name: str):
    """Lazily import cache helpers to avoid importing PySCF during package import."""
    if name in {"cache_fci", "deduplicate_fci_cache"}:
        from src.data.cache import cache_fci, deduplicate_fci_cache

        return {
            "cache_fci": cache_fci,
            "deduplicate_fci_cache": deduplicate_fci_cache,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
