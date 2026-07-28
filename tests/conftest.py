import pathlib
import sys

# Make the repository root importable without installation
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pytest


@pytest.fixture()
def rng():
    return np.random.default_rng(20260720)


@pytest.fixture()
def unit_sphere_points():
    """Deterministic set of unit vectors covering the sphere."""
    from entarchy_vxpy_suite2p.analysis.cmn.helper import IcosahedronSphere

    return IcosahedronSphere(subdiv_lvl=1).get_vertices().astype(np.float64)
