from __future__ import annotations

from pathlib import Path

from datacenter_modeler.models import DataCenterLayout, Equipment

IFC_MISSING_MESSAGE = "IfcOpenShell is not installed; IFC export skipped, OBJ fallback can be used."


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
    return eq.id or eq.name or f"EQ{idx:02d}"


def export_layout_ifc(layout: DataCenterLayout, path: str | Path) -> bool:
    try:
        import ifcopenshell
        import ifcopenshell.guid
    except ImportError:
        print(IFC_MISSING_MESSAGE)
        return False

    f = ifcopenshell.file(schema="IFC4")
    owner_history = f.create_entity("IfcOwnerHistory")

    project = f.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new(), Name=layout.project_name, OwnerHistory=owner_history)
    site = f.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new(), Name="DefaultSite", OwnerHistory=owner_history)
    building = f.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new(), Name="DataCenter", OwnerHistory=owner_history)
    storey = f.create_entity("IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(), Name="Level1", OwnerHistory=owner_history)

    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history, RelatingObject=project, RelatedObjects=[site])
    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history, RelatingObject=site, RelatedObjects=[building])
    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history, RelatingObject=building, RelatedObjects=[storey])

    proxies = []
    type_index: dict[str, int] = {}
    for eq in layout.equipment:
        label = _equipment_label(eq, type_index)
        desc = (
            f"box=({eq.width:.3f},{eq.depth:.3f},{eq.height:.3f});"
            f"position=({eq.x:.3f},{eq.y:.3f},{eq.z:.3f});"
            f"type={eq.type};power_kw={eq.power_kw};cooling_kw={eq.cooling_kw};note={eq.note}"
        )
        proxy = f.create_entity(
            "IfcBuildingElementProxy",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner_history,
            Name=label,
            ObjectType=eq.type,
            Description=desc,
        )
        proxies.append(proxy)

    if proxies:
        f.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner_history,
            RelatedElements=proxies,
            RelatingStructure=storey,
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(out))
    return True


def export_layout_obj(layout: DataCenterLayout, obj_path: str | Path, mtl_path: str | Path) -> None:
    obj_out = Path(obj_path)
    mtl_out = Path(mtl_path)
    obj_out.parent.mkdir(parents=True, exist_ok=True)

    mtl_out.write_text("newmtl default\nKd 0.80 0.80 0.80\n", encoding="utf-8")

    lines = [f"mtllib {mtl_out.name}", "usemtl default"]
    type_index: dict[str, int] = {}
    v = 1
    for eq in layout.equipment:
        name = _equipment_label(eq, type_index)
        x0, x1 = eq.x - eq.width / 2, eq.x + eq.width / 2
        y0, y1 = eq.y - eq.depth / 2, eq.y + eq.depth / 2
        z0, z1 = eq.z, eq.z + max(eq.height, 0.01)
        verts = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        lines.append(f"o {name}")
        lines.extend([f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts])
        faces = [(1,2,3,4),(5,6,7,8),(1,2,6,5),(2,3,7,6),(3,4,8,7),(4,1,5,8)]
        for a,b,c,d in faces:
            lines.append(f"f {v+a-1} {v+b-1} {v+c-1} {v+d-1}")
        v += 8

    obj_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
