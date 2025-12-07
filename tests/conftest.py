import sys
from pathlib import Path

# Ensure the src package is importable when running tests directly from repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
