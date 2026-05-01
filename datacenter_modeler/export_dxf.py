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
    return "OTHER"


def _export_floorplan_dxf_fallback(layout: DataCenterLayout, path: str | Path) -> None:
    lines = ["0", "SECTION", "2", "ENTITIES"]
    for eq in layout.equipment:
        x0 = eq.x - eq.width / 2
        y0 = eq.y - eq.depth / 2
        x1 = eq.x + eq.width / 2
        y1 = eq.y + eq.depth / 2
        layer = _layer_for(eq)
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        lines += ["0", "LWPOLYLINE", "8", layer, "90", str(len(points))]
        for x, y in points:
            lines += ["10", str(x), "20", str(y)]

    lines += ["0", "ENDSEC", "0", "EOF"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_floorplan_dxf(layout: DataCenterLayout, path: str | Path) -> None:
    try:
        import ezdxf
    except ImportError:
        print("ezdxf is not installed; writing minimal fallback DXF.")
        _export_floorplan_dxf_fallback(layout, path)
        return

    doc = ezdxf.new("R2018")
    doc.units = ezdxf.units.M
    msp = doc.modelspace()

    for layer in ["RACK", "CRAC", "DOOR", "AISLE", "OTHER", "TEXT"]:
        if layer not in doc.layers:
            doc.layers.new(name=layer)

    for eq in layout.equipment:
        x0 = eq.x - eq.width / 2
        y0 = eq.y - eq.depth / 2
        x1 = eq.x + eq.width / 2
        y1 = eq.y + eq.depth / 2
        layer = _layer_for(eq)
        msp.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], dxfattribs={"layer": layer})
        msp.add_text(
            f"{eq.name} ({eq.type})",
            dxfattribs={"height": 0.2, "layer": "TEXT"},
        ).set_placement((eq.x, eq.y))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)


def export_floorplan_svg(layout: DataCenterLayout, path: str | Path) -> None:
    if not layout.equipment:
        view_box = "0 0 10 10"
    else:
        min_x = min(eq.x - eq.width / 2 for eq in layout.equipment)
        min_y = min(eq.y - eq.depth / 2 for eq in layout.equipment)
        max_x = max(eq.x + eq.width / 2 for eq in layout.equipment)
        max_y = max(eq.y + eq.depth / 2 for eq in layout.equipment)
        margin = 1.0
        view_box = f"{min_x - margin} {min_y - margin} {max_x - min_x + margin * 2} {max_y - min_y + margin * 2}"

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">',
        '<g fill="none" stroke="black" stroke-width="0.03">',
    ]

    for eq in layout.equipment:
        x = eq.x - eq.width / 2
        y = eq.y - eq.depth / 2
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{eq.width}" height="{eq.depth}" />'
        )
        svg_parts.append(
            f'<text x="{eq.x}" y="{eq.y}" font-size="0.2" fill="blue">{eq.name} ({eq.type})</text>'
        )

    svg_parts.append("</g>")
    svg_parts.append("</svg>")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg_parts) + "\n", encoding="utf-8")
