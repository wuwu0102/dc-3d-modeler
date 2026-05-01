from __future__ import annotations

from pathlib import Path

from datacenter_modeler.models import DataCenterLayout

IFC_MISSING_MESSAGE = (
    "IfcOpenShell is required for IFC export. Please install it with: pip install ifcopenshell"
)


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
    site = f.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new(), Name="Default Site", OwnerHistory=owner_history)
    building = f.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new(), Name="Data Center Building", OwnerHistory=owner_history)
    storey = f.create_entity("IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(), Name="Ground Floor", OwnerHistory=owner_history)

    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history, RelatingObject=project, RelatedObjects=[site])
    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history, RelatingObject=site, RelatedObjects=[building])
    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history, RelatingObject=building, RelatedObjects=[storey])

    proxies = []
    for eq in layout.equipment:
        proxy = f.create_entity(
            "IfcBuildingElementProxy",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner_history,
            Name=eq.name,
            ObjectType=eq.type,
            Description=f"semantic placeholder; size=({eq.width},{eq.depth},{eq.height}), power_kw={eq.power_kw}, cooling_kw={eq.cooling_kw}, note={eq.note}",
        )
        proxies.append(proxy)

    if proxies:
        f.create_entity("IfcRelContainedInSpatialStructure", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history, RelatedElements=proxies, RelatingStructure=storey)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(out))
    return True
