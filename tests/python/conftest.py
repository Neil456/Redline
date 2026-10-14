import glob
import os
import shutil
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))


def _find_binary(name: str) -> str | None:
    env = os.environ.get("REDLINE_BUILD_DIR")
    candidates = [os.path.join(env, name)] if env else []
    candidates += sorted(glob.glob(os.path.join(ROOT, "build*", name)))
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return shutil.which(name)


@pytest.fixture(scope="session")
def ref_bin() -> str:
    path = _find_binary("redline_ref")
    if path is None:
        pytest.skip("redline_ref not built (cmake -DREDLINE_BUILD_CUDA=OFF ... && cmake --build)")
    return path
