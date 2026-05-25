"""Reorganize the Agentic AI Workflows repository.

Moves notebooks into `notebooks/` and screenshots into `assets/`.
The script is safe to re-run and only handles files from the repository root.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def move_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Moving {source.name} -> {destination.parent.name}/")
        shutil.move(str(source), str(destination))


def main() -> None:
    notebooks_dir = ROOT / "notebooks"
    projects_dir = ROOT / "projects"
    assets_dir = ROOT / "assets"
    notebooks_dir.mkdir(exist_ok=True)
    projects_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    for item in ROOT.iterdir():
        if item.suffix == ".ipynb":
            move_if_exists(item, notebooks_dir / item.name)
        elif item.name in {
            "streamlit_ui_longtermMemory.png",
            "streamlit_ui_longtermMemory_db.png",
        }:
            move_if_exists(item, assets_dir / item.name)
        elif item.suffix == ".py" and item.name not in {
            "main.py",
            "longtermMemory_Chatbot_UI.py",
            "tools/reorganize_agentic.py",
        }:
            move_if_exists(item, projects_dir / item.name)

    print("Reorganization complete.")


if __name__ == "__main__":
    main()
