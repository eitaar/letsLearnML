import subprocess
import sys
from pathlib import Path


def test_tlnet_script_runs_successfully():
    project_root = Path(__file__).parents[2]

    result = subprocess.run(
        [sys.executable, str(project_root / "playground" / "TLnet.py")],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
