import os
from pathlib import Path

MPLCONFIGDIR = Path("/tmp/visionplay-matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

__all__: list[str] = []
