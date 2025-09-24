import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

if importlib.util.find_spec("rich") is None:
    pytest.skip("rich is required for CLI smoke tests", allow_module_level=True)

_CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None

SCRIPTS = [
    Path("scripts/pull_avatar.py"),
    Path("scripts/build_index.py"),
    Path("scripts/query.py"),
    Path("scripts/live_search.py"),
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_expose_help(script: Path) -> None:
    if script.name == "live_search.py" and not _CV2_AVAILABLE:
        pytest.skip("OpenCV not available in test environment")

    env = os.environ.copy()
    env.setdefault("FAKE_EMB", "1")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
