from __future__ import annotations

from pathlib import Path

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
    return "TEXT"


def _label_for(eq: Equipment) -> str:
    t = eq.type.lower()
    if t == "rack":
        return eq.id or "R01"
    if t == "crac":
        return eq.id or "CRAC01"
    if t == "door":
        return eq.id or "Door01"
    if "cold" in t and "aisle" in t:
        return "Cold Aisle"
    if "hot" in t and "aisle" in t:
        return "Hot Aisle"
    return eq.id or eq.name


def _fit_svg_text(eq: Equipment, label: str) -> tuple[str, float, bool]:
    # Keep 10% padding on each side => 80% drawable area.
    width_limit = max(eq.width * 0.8, 0.08)
    height_limit = max(eq.depth * 0.8, 0.08)
    rotate = eq.type.lower() == "rack" and eq.width < eq.depth
    effective_width = height_limit if rotate else width_limit

    for size in (0.2, 0.16, 0.12):
        est_w = len(label) * size * 0.62
        if est_w <= effective_width and size <= height_limit:
            return label, size, rotate

    short_label = (eq.id or label or "R01")[:4]
    return short_label, 0.12, rotate



def _export_r12_ascii_fallback(layout: DataCenterLayout, path: str | Path) -> bool:
    lines = [
        "0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES",
        "0", "TABLE", "2", "LAYER", "70", "5",
    ]
    for layer, color in (("RACK", "7"), ("CRAC", "5"), ("AISLE", "4"), ("DOOR", "2"), ("TEXT", "7")):
        lines += ["0", "LAYER", "2", layer, "70", "0", "62", color, "6", "CONTINUOUS"]
    lines += ["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]

    for eq in layout.equipment:
        x0 = eq.x - eq.width / 2
        y0 = eq.y - eq.depth / 2
        x1 = eq.x + eq.width / 2
        y1 = eq.y + eq.depth / 2
        layer = _layer_for(eq)
        pts = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        for (sx, sy), (ex, ey) in zip(pts, pts[1:] + pts[:1]):
            lines += ["0", "LINE", "8", layer, "10", f"{sx}", "20", f"{sy}", "30", "0.0", "11", f"{ex}", "21", f"{ey}", "31", "0.0"]

        label = _label_for(eq)
        text_h = max(min(min(eq.width, eq.depth) * 0.28, 0.3), 0.12)
        lines += [
            "0", "TEXT", "8", "TEXT", "10", f"{eq.x}", "20", f"{eq.y}", "30", "0.0",
            "40", f"{text_h}", "1", label, "72", "1", "73", "2", "11", f"{eq.x}", "21", f"{eq.y}", "31", "0.0",
        ]

    lines += ["0", "ENDSEC", "0", "EOF"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
    return True

def export_floorplan_dxf(layout: DataCenterLayout, path: str | Path) -> bool:
    try:
        import ezdxf
    except ImportError:
        print("ezdxf is required for DXF export. Please install it with: pip install ezdxf")
        return _export_r12_ascii_fallback(layout, path)

    doc = ezdxf.new("R12")
    msp = doc.modelspace()

    for layer in ["RACK", "CRAC", "AISLE", "DOOR", "TEXT"]:
        if layer not in doc.layers:
            doc.layers.new(name=layer)

    for eq in layout.equipment:
        x0 = eq.x - eq.width / 2
        y0 = eq.y - eq.depth / 2
        x1 = eq.x + eq.width / 2
        y1 = eq.y + eq.depth / 2
        layer = _layer_for(eq)

        edges = (
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0)),
        )
        for start, end in edges:
            msp.add_line(start, end, dxfattribs={"layer": layer})

        label = _label_for(eq)
        text_h = max(min(min(eq.width, eq.depth) * 0.28, 0.3), 0.12)
        msp.add_text(
            label,
            dxfattribs={"height": text_h, "layer": "TEXT", "insert": (eq.x, eq.y)},
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out)
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
        legend_h = max(h * 0.25, 2.5)
        view_box = f"{min_x - margin} {min_y - margin - 1.0} {w + margin * 2} {h + margin * 2 + legend_h + 1.4}"

    def fill(eq: Equipment) -> str:
        t = eq.type.lower()
        if t == "rack":
            return "#d9d9d9"
        if t == "crac":
            return "#cfe8ff"
        if t == "door":
            return "#fff5bf"
        if "cold" in t and "aisle" in t:
            return "#d7f6f8"
        if "hot" in t and "aisle" in t:
            return "#ffe5cc"
        return "#f2f2f2"

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50%" y="4%" text-anchor="middle" dominant-baseline="middle" font-size="0.75" fill="#222">Sample Data Center Room - Floor Plan</text>',
        '<g stroke="#333" stroke-width="0.03">',
    ]

    for eq in layout.equipment:
        x = eq.x - eq.width / 2
        y = eq.y - eq.depth / 2
        label = _label_for(eq)
        safe_label, font_size, rotate = _fit_svg_text(eq, label)
        rotate_attr = f' transform="rotate(90 {eq.x} {eq.y})"' if rotate else ""
        svg_parts.append(f'<rect x="{x}" y="{y}" width="{eq.width}" height="{eq.depth}" fill="{fill(eq)}" />')
        svg_parts.append(
            f'<text x="{eq.x}" y="{eq.y}" text-anchor="middle" dominant-baseline="middle" font-size="{font_size}" fill="#0d2a4d"{rotate_attr}>{safe_label}</text>'
        )

    svg_parts.append("</g>")
    svg_parts.extend([
        '<g id="legend" font-size="0.25" fill="#111">',
        '<text x="50%" y="92%" text-anchor="middle" dominant-baseline="middle">Legend</text>',
        '<text x="50%" y="94%" text-anchor="middle" dominant-baseline="middle">Rack / CRAC / Cold Aisle / Hot Aisle / Door</text>',
        '<text x="50%" y="96%" text-anchor="middle" dominant-baseline="middle">Scale factor: 1.220930</text>',
        '<text x="50%" y="98%" text-anchor="middle" dominant-baseline="middle">Calibration reference: Door01</text>',
        '</g>',
        '</svg>',
    ])

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg_parts) + "\n", encoding="utf-8")
