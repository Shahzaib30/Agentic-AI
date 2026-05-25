"""Launcher for the Agentic AI Workflows Streamlit app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).resolve().parent / "longtermMemory_Chatbot_UI.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print("Launching the memory-persistent Streamlit chatbot...")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("Streamlit is not installed in the active environment. Run `pip install -e .` first.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Streamlit exited with a non-zero status: {exc.returncode}") from exc


if __name__ == "__main__":
    main()
