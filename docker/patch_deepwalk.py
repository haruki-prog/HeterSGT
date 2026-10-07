"""Make pinned DeepWalk 1.0.3 importable on Python 3.11."""

from pathlib import Path

import deepwalk

module = Path(deepwalk.__file__).with_name('graph.py')
source = module.read_text()
old = 'from collections import defaultdict, Iterable'
new = 'from collections import defaultdict\nfrom collections.abc import Iterable'
if source.count(old) != 1:
    raise RuntimeError(f'Unexpected DeepWalk source in {module}')
module.write_text(source.replace(old, new, 1))
