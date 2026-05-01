from __future__ import annotations

from pathlib import Path

from datacenter_modeler.coordinates import centered_equipment
from datacenter_modeler.models import DataCenterLayout, Equipment

TEXT_HEIGHT = 0.15


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
    lines: list[str] = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "0", "ENDSEC", "0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LAYER", "70", "5"]
    for layer, color in (("RACK", "7"), ("CRAC", "5"), ("AISLE", "4"), ("DOOR", "2"), ("TEXT", "7")):
        lines += ["0", "LAYER", "2", layer, "70", "0", "62", color, "6", "CONTINUOUS"]
    lines += ["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]

    type_index: dict[str, int] = {}
    for eq in centered_equipment(layout):
        x0, y0 = eq.x - eq.width / 2, eq.y - eq.depth / 2
        x1, y1 = eq.x + eq.width / 2, eq.y + eq.depth / 2
        layer = _layer_for(eq)
        _add_line(lines, layer, x0, y0, x1, y0)
        _add_line(lines, layer, x1, y0, x1, y1)
        _add_line(lines, layer, x1, y1, x0, y1)
        _add_line(lines, layer, x0, y1, x0, y0)
        label = _label_for(eq, type_index)
        lines += ["0", "TEXT", "8", "TEXT", "10", f"{eq.x:.6f}", "20", f"{eq.y:.6f}", "30", "0.0", "40", f"{TEXT_HEIGHT:.2f}", "1", label]

    lines += ["0", "ENDSEC", "0", "EOF"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
    return True


def export_floorplan_svg(layout: DataCenterLayout, path: str | Path) -> None:
    centered = centered_equipment(layout)
    if not centered:
        min_x = min_y = -5.0
        width = height = 10.0
    else:
        min_x = min(eq.x - eq.width / 2 for eq in centered)
        min_y = min(eq.y - eq.depth / 2 for eq in centered)
        max_x = max(eq.x + eq.width / 2 for eq in centered)
        max_y = max(eq.y + eq.depth / 2 for eq in centered)
        width, height = max_x - min_x, max_y - min_y
    margin = max(max(width, height) * 0.1, 1.0)
    view_box = f"{min_x - margin:.3f} {min_y - margin:.3f} {width + 2*margin:.3f} {height + 2*margin:.3f}"

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="0" y="-0.6" text-anchor="middle" font-size="0.28">Sample Data Center Room - Floor Plan</text>',
    ]
    type_index: dict[str, int] = {}
    for eq in centered:
        x = eq.x - eq.width / 2
        y = eq.y - eq.depth / 2
        label = _label_for(eq, type_index)
        svg_parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{eq.width:.3f}" height="{eq.depth:.3f}" fill="#f2f2f2" stroke="#333" stroke-width="0.03" />')
        svg_parts.append(f'<text x="{eq.x:.3f}" y="{eq.y:.3f}" text-anchor="middle" dominant-baseline="middle" font-size="0.11">{label}</text>')

    legend_x = min_x
    legend_y = min_y + height + 0.2
    svg_parts.append(f'<text x="{legend_x:.3f}" y="{legend_y:.3f}" font-size="0.12">Legend: Rack / CRAC / Door / ColdAisle / HotAisle</text>')
    svg_parts.append("</svg>")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg_parts) + "\n", encoding="utf-8")
