from pathlib import Path

from matplotlib.figure import Figure


def save_figure(fig: Figure, path: str, dpi: int = 300) -> None:
    """Save a matplotlib figure with standardized layout and resolution."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
