"""Re-execute the tutorial notebooks (about a minute in total)."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = sorted((Path(__file__).parent.parent / "examples").glob("*.ipynb"))


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_runs(path):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
