"""So that `python -m entarchy_vxpy_suite2p` reaches the command line tool."""
import sys

from .cli import main

sys.exit(main())
