from __future__ import annotations

from pathlib import Path

from datacenter_modeler.models import DataCenterLayout, Equipment

TEXT_HEIGHT = 0.12


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
    return "GEN"


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
    return (eq.id or eq.name or "EQ").encode("ascii", "ignore").decode("ascii") or "EQ"


def _add_line(lines: list[str], layer: str, sx: float, sy: float, ex: float, ey: float) -> None:
    lines += [
        "0", "LINE", "8", layer,
        "10", f"{sx:.6f}", "20", f"{sy:.6f}", "30", "0.0",
        "11", f"{ex:.6f}", "21", f"{ey:.6f}", "31", "0.0",
    ]


def export_floorplan_dxf(layout: DataCenterLayout, path: str | Path) -> bool:
    """Export conservative AutoCAD R12 ASCII DXF (LINE + TEXT only)."""
    lines: list[str] = [
        "0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES",
        "0", "TABLE", "2", "LAYER", "70", "5",
    ]
    for layer, color in (("RACK", "7"), ("CRAC", "5"), ("AISLE", "4"), ("DOOR", "2"), ("GEN", "7"), ("TEXT", "7")):
        lines += ["0", "LAYER", "2", layer, "70", "0", "62", color, "6", "CONTINUOUS"]
    lines += ["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]

    type_index: dict[str, int] = {}
    for eq in layout.equipment:
        x0 = eq.x - eq.width / 2
        y0 = eq.y - eq.depth / 2
        x1 = eq.x + eq.width / 2
        y1 = eq.y + eq.depth / 2
        layer = _layer_for(eq)

        _add_line(lines, layer, x0, y0, x1, y0)
        _add_line(lines, layer, x1, y0, x1, y1)
        _add_line(lines, layer, x1, y1, x0, y1)
        _add_line(lines, layer, x0, y1, x0, y0)

        label = _label_for(eq, type_index)
        lines += [
            "0", "TEXT", "8", "TEXT",
            "10", f"{eq.x:.6f}", "20", f"{eq.y:.6f}", "30", "0.0",
            "40", f"{TEXT_HEIGHT:.2f}", "1", label,
            "72", "1", "73", "2", "11", f"{eq.x:.6f}", "21", f"{eq.y:.6f}", "31", "0.0",
        ]

    lines += ["0", "ENDSEC", "0", "EOF"]

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
    return True


def export_floorplan_svg(layout: DataCenterLayout, path: str | Path) -> None:
    if not layout.equipment:
        view_box = "0 0 10 10"
    else:
        min_x = min(eq.x - eq.width / 2 for eq in layout.equipment)
        min_y = min(eq.y - eq.depth / 2 for eq in layout.equipment)
        max_x = max(eq.x + eq.width / 2 for eq in layout.equipment)
        max_y = max(eq.y + eq.depth / 2 for eq in layout.equipment)
        w = max_x - min_x
        h = max_y - min_y
        margin = max(max(w, h) * 0.08, 1.2)
        view_box = f"{min_x - margin} {min_y - margin} {w + margin * 2} {h + margin * 2}"

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    type_index: dict[str, int] = {}
    for eq in layout.equipment:
        x = eq.x - eq.width / 2
        y = eq.y - eq.depth / 2
        label = _label_for(eq, type_index)
        svg_parts.append(f'<rect x="{x}" y="{y}" width="{eq.width}" height="{eq.depth}" fill="#f2f2f2" stroke="#333" stroke-width="0.03" />')
        svg_parts.append(f'<text x="{eq.x}" y="{eq.y}" text-anchor="middle" dominant-baseline="middle" font-size="0.12">{label}</text>')
    svg_parts.append("</svg>")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg_parts) + "\n", encoding="utf-8")
