"""Run the root smoke test from either pytest or this directory."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smoke_test import main  # noqa: E402


def test_offline_api_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
