from __future__ import annotations

import json
from pathlib import Path

from datacenter_modeler.models import DataCenterLayout


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_layout(path: str | Path) -> DataCenterLayout:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return DataCenterLayout.from_dict(data)


def save_layout(layout: DataCenterLayout, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(layout.to_dict(), f, indent=2, ensure_ascii=False)
