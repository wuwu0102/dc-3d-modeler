from __future__ import annotations

from pathlib import Path

from datacenter_modeler.coordinates import centered_equipment
from datacenter_modeler.models import DataCenterLayout, Equipment


MATERIALS = {
    "rack": "rack_mat",
    "crac": "crac_mat",
    "door": "door_mat",
    "cold_aisle": "aisle_cold_mat",
    "hot_aisle": "aisle_hot_mat",
    "default": "rack_mat",
}


def _equipment_label(eq: Equipment, index_by_type: dict[str, int]) -> str:
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


def _material_for(eq: Equipment) -> str:
    t = eq.type.lower()
    if "cold" in t and "aisle" in t:
        return MATERIALS["cold_aisle"]
    if "hot" in t and "aisle" in t:
        return MATERIALS["hot_aisle"]
    return MATERIALS.get(t, MATERIALS["default"])


def export_layout_obj(layout: DataCenterLayout, obj_path: str | Path, mtl_path: str | Path) -> None:
    obj_out = Path(obj_path)
    mtl_out = Path(mtl_path)
    obj_out.parent.mkdir(parents=True, exist_ok=True)

    mtl_lines = [
        "# Units: meter",
        "newmtl rack_mat", "Kd 0.70 0.70 0.75",
        "newmtl crac_mat", "Kd 0.50 0.70 0.90",
        "newmtl aisle_cold_mat", "Kd 0.60 0.85 1.00",
        "newmtl aisle_hot_mat", "Kd 1.00 0.75 0.55",
        "newmtl door_mat", "Kd 0.70 0.55 0.40",
    ]
    mtl_out.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")

    lines = [f"mtllib {mtl_out.name}", "# Units: meter"]
    type_index: dict[str, int] = {}
    v = 1
    for eq in centered_equipment(layout):
        name = _equipment_label(eq, type_index)
        mat = _material_for(eq)
        x0, x1 = eq.x - eq.width / 2, eq.x + eq.width / 2
        y0, y1 = eq.y - eq.depth / 2, eq.y + eq.depth / 2
        h = max(eq.height, 0.01)
        z0, z1 = 0.0, h
        verts = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        lines.append(f"o {name}")
        lines.append(f"usemtl {mat}")
        lines.extend([f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts])
        faces = [(1, 2, 3, 4), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]
        for a, b, c, d in faces:
            lines.append(f"f {v + a - 1} {v + b - 1} {v + c - 1} {v + d - 1}")
        v += 8

    obj_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
