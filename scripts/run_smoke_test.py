from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "uvicorn", "backend.app", "--app-dir", ".", "--host", "127.0.0.1", "--port", "8000"]
    proc = subprocess.Popen(cmd)
    print(proc.pid)


if __name__ == "__main__":
    main()

