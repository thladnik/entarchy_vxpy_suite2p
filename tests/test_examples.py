"""Execute the example notebooks, so they cannot silently go stale.

The notebook is documentation that runs, which is the only kind that stays
true. It builds its own entarchy in a temporary directory, so nothing here
depends on a dataset being present.
"""
import pathlib

import pytest

nbformat = pytest.importorskip('nbformat', reason='notebook execution requires nbformat')
nbclient = pytest.importorskip('nbclient', reason='notebook execution requires nbclient')

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / 'examples'
NOTEBOOKS = sorted(EXAMPLES.glob('*.ipynb'))


def test_examples_exist():
    assert NOTEBOOKS, f'no example notebooks found in {EXAMPLES}'


@pytest.mark.parametrize('notebook_path', NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_stored_outputs(notebook_path):
    """Outputs make diffs unreadable, so the committed notebooks stay clean."""
    notebook = nbformat.read(notebook_path, as_version=4)

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            assert not cell.get('outputs'), f'cell {index} has stored outputs'
            assert cell.get('execution_count') is None, f'cell {index} has an execution count'


@pytest.mark.parametrize('notebook_path', NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cells_have_stable_ids(notebook_path):
    """Ids churn on every save unless they are set deliberately, which buries
    a one-cell change in a whole-file diff."""
    notebook = nbformat.read(notebook_path, as_version=4)
    ids = [cell.get('id') for cell in notebook.cells]

    assert all(ids), 'every cell needs an id'
    assert len(set(ids)) == len(ids), 'cell ids must be unique'


@pytest.mark.slow
@pytest.mark.parametrize('notebook_path', NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(notebook_path):
    """Run every cell; any exception fails the test.

    Executed from the repository root rather than a temporary directory, so the
    package is importable without having been installed - which is also how
    someone reading a fresh checkout would run it. The notebook writes only into
    a directory of its own from `tempfile`, so nothing lands in the repository.
    """
    from nbclient import NotebookClient

    repository_root = EXAMPLES.parent
    notebook = nbformat.read(notebook_path, as_version=4)

    client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                            resources={'metadata': {'path': str(repository_root)}})
    client.execute()
