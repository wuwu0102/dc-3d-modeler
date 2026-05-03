from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from datacenter_modeler.io import ensure_output_dir, save_layout
from datacenter_modeler.models import DataCenterLayout, Equipment

SUPPORTED = {".ply", ".obj", ".glb", ".npz", ".json"}


def load_scan_geometry(input_path: str | Path) -> dict:
    path = Path(input_path)
    if not path.exists() or path.suffix.lower() not in SUPPORTED:
        raise ValueError("Scan import failed: unsupported or invalid geometry file.")

    ext = path.suffix.lower()
    try:
        if ext == ".npz":
            d = np.load(path)
            pts = d.get("points")
            if pts is None:
                raise ValueError
            return {"points": np.asarray(pts, dtype=float)}
        if ext == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            pts = np.array(data.get("points", []), dtype=float)
            return {"points": pts}
        text = path.read_text(encoding="utf-8", errors="ignore")
        pts = []
        for line in text.splitlines():
            if line.startswith("v "):
                vals = line.split()
                if len(vals) >= 4:
                    pts.append([float(vals[1]), float(vals[2]), float(vals[3])])
        return {"points": np.array(pts, dtype=float)}
    except Exception as exc:
        raise ValueError("Scan import failed: unsupported or invalid geometry file.") from exc


def extract_scan_bounds(scan: dict) -> dict[str, float]:
    pts = scan.get("points")
    if pts is None or len(pts) == 0:
        return {"min_x": -3.0, "max_x": 3.0, "min_y": -2.0, "max_y": 2.0, "min_z": 0.0, "max_z": 2.8}
    return {
        "min_x": float(np.min(pts[:, 0])), "max_x": float(np.max(pts[:, 0])),
        "min_y": float(np.min(pts[:, 1])), "max_y": float(np.max(pts[:, 1])),
        "min_z": float(np.min(pts[:, 2])), "max_z": float(np.max(pts[:, 2])),
    }


def estimate_floor_plane(scan: dict) -> dict[str, float]:
    bounds = extract_scan_bounds(scan)
    return {"z": bounds["min_z"], "normal": [0.0, 0.0, 1.0]}


def estimate_room_outline(scan: dict) -> list[list[float]]:
    b = extract_scan_bounds(scan)
    return [[b["min_x"], b["min_y"]], [b["max_x"], b["min_y"]], [b["max_x"], b["max_y"]], [b["min_x"], b["max_y"]]]


def scan_to_layout(input_path: str | Path, output_layout_path: str | Path) -> DataCenterLayout:
    scan = load_scan_geometry(input_path)
    b = extract_scan_bounds(scan)
    cx = (b["min_x"] + b["max_x"]) / 2
    cy = (b["min_y"] + b["max_y"]) / 2
    w = max(b["max_x"] - b["min_x"], 2.0)
    d = max(b["max_y"] - b["min_y"], 2.0)
    h = max(b["max_z"] - b["min_z"], 2.2)

    layout = DataCenterLayout(project_name="Scan Imported Data Center")
    layout.equipment = [
        Equipment(id="room_boundary", type="room_boundary", name="RoomBoundary", x=cx, y=cy, z=0, width=w, depth=d, height=0.05, note="placeholder from scan bounds"),
        Equipment(id="door_placeholder", type="door", name="DoorPlaceholder", x=b["min_x"] + 0.6, y=cy, z=0, width=1.0, depth=0.2, height=2.1, note="placeholder"),
        Equipment(id="cal_ref", type="wall", name="CalibrationReferencePlaceholder", x=cx, y=b["min_y"] + 0.2, z=0, width=1.0, depth=0.2, height=2.1, note="use for calibration reference"),
    ]
    save_layout(layout, output_layout_path)
    return layout
