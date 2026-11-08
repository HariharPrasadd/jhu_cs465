#!/usr/bin/env python3
"""build and run the optimized rust parser"""

import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile


folder = Path(__file__).resolve().parent
source = folder / "parse2.rs"
bundled = folder / "parse2-linux-x86_64"

# gradescope runs linux without exposing rustc to student programs
if sys.platform.startswith("linux") and platform.machine() in {"x86_64", "amd64"}:
    binary = bundled
    binary.chmod(binary.stat().st_mode | 0o111)
else:
    binary = Path(tempfile.gettempdir()) / "hw4_parse2_rust"
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(
            ["rustc", "--edition=2021", "-O", str(source), "-o", str(binary)],
            check=True,
        )

os.execv(binary, [str(binary), *sys.argv[1:]])
