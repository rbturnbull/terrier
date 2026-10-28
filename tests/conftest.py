pytest_plugins = ("torchapp.testing",)

from pathlib import Path
import pandas as pd
import pytest


TEST_DATA_DIR = Path(__file__).parent / "test-data"


@pytest.fixture
def results_df() -> pd.DataFrame:
    """Results in the format written by Terrier's output_results."""
    rows = [
        # accession, original, prediction, probability, DNA, LTR
        ("r1", "DNA/hAT", "DNA/hAT", 0.90, 0.90, 0.10),
        ("r2", "DNA/TcMar", "DNA/hAT", 0.60, 0.70, 0.30),
        ("r3", "LTR/Gypsy", "LTR/Gypsy", 0.95, 0.05, 0.95),
        ("r4", "Unknown", "DNA/TcMar", 0.80, 0.80, 0.20),
        ("r5", "LTR/Gypsy", "LTR", 0.50, 0.40, 0.60),
    ]
    return pd.DataFrame(
        {
            "file": "example.fasta",
            "accession": [r[0] for r in rows],
            "prediction": [r[2] for r in rows],
            "probability": [r[3] for r in rows],
            "original_id": [f"{r[0]}#{r[1]}" for r in rows],
            "original_classification": [r[1] for r in rows],
            "description": "",
            "DNA": [r[4] for r in rows],
            "LTR": [r[5] for r in rows],
        }
    )


@pytest.fixture
def results_csv(results_df, tmp_path) -> Path:
    path = tmp_path / "results.csv"
    results_df.to_csv(path, index=False)
    return path


@pytest.fixture
def stub_image_export(monkeypatch):
    """Static image export needs a browser, so write a placeholder file instead."""
    import plotly.graph_objects as go

    def write_image(self, file, *args, **kwargs):
        Path(file).write_bytes(b"image")

    monkeypatch.setattr(go.Figure, "write_image", write_image)
