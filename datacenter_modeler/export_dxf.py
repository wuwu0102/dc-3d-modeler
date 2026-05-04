from __future__ import annotations

from pathlib import Path

from datacenter_modeler.geometry_utils import get_centered_equipment, get_layout_bounds
from datacenter_modeler.models import DataCenterLayout, Equipment


def _layer_for(eq: Equipment) -> str:
    t = eq.type.lower()
    if t == "rack":
        return "RACK"
    if t == "crac":
        return "CRAC"
    if t == "door":
        return "DOOR"
    if "aisle" in t:
        return "AISLE"
    return "RACK"


def _text_height(eq: Equipment) -> float:
    t = eq.type.lower()
    if t == "rack":
        return 0.15
    if t == "crac":
        return 0.18
    if "aisle" in t:
        return 0.20
    return 0.15


def _label_for(eq: Equipment, index_by_type: dict[str, int]) -> str:
    t = eq.type.lower()
    index_by_type[t] = index_by_type.get(t, 0) + 1
    idx = index_by_type[t]
    if t == "rack":
        return f"R{idx:02d}"
    if t == "crac":
        return f"CRAC{idx:02d}"
    if t == "door":
        return f"Door{idx:02d}"
    if "cold" in t and "aisle" in t:
        return "ColdAisle"
    if "hot" in t and "aisle" in t:
        return "HotAisle"
    return f"EQ{idx:02d}"


def _add_line(lines: list[str], layer: str, sx: float, sy: float, ex: float, ey: float) -> None:
    lines += ["0", "LINE", "8", layer, "10", f"{sx:.6f}", "20", f"{sy:.6f}", "30", "0.0", "11", f"{ex:.6f}", "21", f"{ey:.6f}", "31", "0.0"]


def export_floorplan_dxf(layout: DataCenterLayout, path: str | Path) -> bool:
    scan_mode = layout.source_mode == "scan"
    lines = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "0", "ENDSEC", "0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LAYER", "70", "5"]
    layer_defs = (("SCAN_BOUNDARY", "7"), ("CALIBRATION", "3"), ("TEXT", "7")) if scan_mode else (("RACK", "7"), ("CRAC", "5"), ("AISLE", "4"), ("DOOR", "2"), ("TEXT", "7"))
    for layer, color in layer_defs:
        lines += ["0", "LAYER", "2", layer, "70", "0", "62", color, "6", "CONTINUOUS"]
    lines += ["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]

    type_index: dict[str, int] = {}
    for eq in get_centered_equipment(layout):
        if scan_mode and eq.type.lower() not in {"room_boundary", "wall"}:
            continue
        x0, y0 = eq.x - eq.width / 2, eq.y - eq.depth / 2
        x1, y1 = eq.x + eq.width / 2, eq.y + eq.depth / 2
        if scan_mode and eq.type.lower() == "room_boundary":
            layer, label = "SCAN_BOUNDARY", "ScanBoundary"
            _add_line(lines, layer, x0, y0, x1, y0); _add_line(lines, layer, x1, y0, x1, y1)
            _add_line(lines, layer, x1, y1, x0, y1); _add_line(lines, layer, x0, y1, x0, y0)
        elif scan_mode:
            layer, label = "CALIBRATION", "CalibrationReference"
            _add_line(lines, layer, x0, eq.y, x1, eq.y)
        else:
            layer = _layer_for(eq)
            _add_line(lines, layer, x0, y0, x1, y0); _add_line(lines, layer, x1, y0, x1, y1)
            _add_line(lines, layer, x1, y1, x0, y1); _add_line(lines, layer, x0, y1, x0, y0)
            label = _label_for(eq, type_index)
        lines += ["0", "TEXT", "8", "TEXT", "10", f"{eq.x:.6f}", "20", f"{eq.y:.6f}", "30", "0.0", "40", f"{_text_height(eq):.2f}", "1", label]
    if scan_mode:
        lines += ["0", "TEXT", "8", "TEXT", "10", "0.0", "20", "0.0", "30", "0.0", "40", "0.20", "1", f"Scan imported. Equipment annotation required. Scale: {layout.scale_factor:.4f}"]

    lines += ["0", "ENDSEC", "0", "EOF"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
    return True


def export_floorplan_svg(layout: DataCenterLayout, path: str | Path) -> None:
    scan_mode = layout.source_mode == "scan"
    centered = get_centered_equipment(layout)
    bounds = get_layout_bounds(layout)
    width = max(bounds["max_x"] - bounds["min_x"], 6.0)
    height = max(bounds["max_y"] - bounds["min_y"], 6.0)
    margin = max(width, height) * 0.15
    min_x = -width / 2 - margin
    min_y = -height / 2 - margin
    view_box = f"{min_x:.3f} {min_y:.3f} {width + 2 * margin:.3f} {height + 2 * margin + 1.6:.3f}"

    colors = {"rack": "#dbe8ff", "crac": "#d7f5e3", "door": "#ffe3c7", "cold": "#e2f3ff", "hot": "#ffdede"}
    title = "Scan Boundary Preview" if scan_mode else "Sample Data Center Room - Floor Plan"
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="0" y="-0.9" text-anchor="middle" font-size="0.28" font-family="Arial">{title}</text>']

    layers: dict[str, list[tuple[Equipment, str, str]]] = {
        "boundary": [],
        "wall": [],
        "aisle": [],
        "equipment": [],
    }
    idx: dict[str, int] = {}

    for eq in centered:
        t = eq.type.lower()
        label = _label_for(eq, idx)
        if scan_mode and t not in {"room_boundary", "wall"}:
            continue
        if t == "room_boundary":
            layers["boundary"].append((eq, label, ""))
            continue
        if t == "wall":
            layers["wall"].append((eq, label, ""))
            continue
        fill = colors["rack"]
        if t == "crac":
            fill = colors["crac"]
        elif t == "door":
            fill = colors["door"]
        elif "cold" in t and "aisle" in t:
            fill = colors["cold"]
        elif "hot" in t and "aisle" in t:
            fill = colors["hot"]

        if "aisle" in t:
            layers["aisle"].append((eq, label, fill))
        else:
            layers["equipment"].append((eq, label, fill))

    labels: list[tuple[Equipment, str]] = []
    for eq, label, _ in layers["boundary"]:
        x = eq.x - eq.width / 2
        y = eq.y - eq.depth / 2
        parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{eq.width:.3f}" height="{eq.depth:.3f}" fill="none" stroke="#999" stroke-width="0.04" stroke-dasharray="6 4"/>')
        if scan_mode:
            parts.append(f'<text x="{eq.x:.3f}" y="{(y - 0.15):.3f}" text-anchor="middle" font-size="0.12" font-family="Arial">BBox: {eq.width:.2f}m x {eq.depth:.2f}m</text>')
    for eq, label, _ in layers["wall"]:
        x = eq.x - eq.width / 2
        y = eq.y - eq.depth / 2
        if scan_mode:
            parts.append(f'<line x1="{x:.3f}" y1="{eq.y:.3f}" x2="{(x + eq.width):.3f}" y2="{eq.y:.3f}" stroke="#4ea3ff" stroke-width="0.05"/>')
        else:
            parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{eq.width:.3f}" height="{eq.depth:.3f}" fill="none" stroke="#b0b0b0" stroke-width="0.04"/>')
    for group in ("aisle", "equipment"):
        for eq, label, fill in layers[group]:
            x = eq.x - eq.width / 2
            y = eq.y - eq.depth / 2
            parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{eq.width:.3f}" height="{eq.depth:.3f}" fill="{fill}" stroke="#333" stroke-width="0.03"/>')
            labels.append((eq, label))
    for eq, label in labels:
        parts.append(f'<text x="{eq.x:.3f}" y="{eq.y:.3f}" text-anchor="middle" dominant-baseline="middle" font-size="0.11" font-family="Arial">{label}</text>')

    ly = height / 2 + margin + 0.2
    if scan_mode:
        parts.append(f'<text x="{-width / 2:.3f}" y="{ly:.3f}" font-size="0.14" font-family="Arial">Scan imported. Equipment annotation required.</text>')
        parts.append(f'<text x="{-width / 2:.3f}" y="{(ly + 0.2):.3f}" font-size="0.14" font-family="Arial">Scale factor: {layout.scale_factor:.4f}</text>')
    else:
        parts.append(f'<text x="{-width / 2:.3f}" y="{ly:.3f}" font-size="0.14" font-family="Arial">Legend: Rack / CRAC / Door / ColdAisle / HotAisle</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")
