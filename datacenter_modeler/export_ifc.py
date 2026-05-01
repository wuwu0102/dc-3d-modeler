from __future__ import annotations

from pathlib import Path

from datacenter_modeler.coordinates import centered_equipment
from datacenter_modeler.models import DataCenterLayout, Equipment

IFC_MISSING_MESSAGE = "IFC skipped, OBJ generated instead"


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


def _shape_for(eq: Equipment) -> tuple[float, float, float]:
    t = eq.type.lower()
    if t == "rack":
        return 0.6, 1.2, 2.2
    if t == "crac":
        return eq.width, eq.depth, max(eq.height, 2.0)
    if t == "door":
        return eq.width, eq.depth, max(eq.height, 2.1)
    if "aisle" in t:
        return eq.width, eq.depth, 0.05
    return eq.width, eq.depth, max(eq.height, 0.1)


def export_layout_ifc(layout: DataCenterLayout, path: str | Path) -> bool:
    try:
        import ifcopenshell
        import ifcopenshell.api
        import ifcopenshell.guid
    except Exception:
        return False

    try:
        f = ifcopenshell.api.run("project.create_file")
        project = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcProject", name=layout.project_name)
        ifcopenshell.api.run("unit.assign_unit", f)
        model = ifcopenshell.api.run("context.add_context", f, context_type="Model")
        body = ifcopenshell.api.run("context.add_context", f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model)
        site = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcSite", name="Site")
        building = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuilding", name="DataCenter")
        storey = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Level1")
        ifcopenshell.api.run("aggregate.assign_object", f, products=[site], relating_object=project)
        ifcopenshell.api.run("aggregate.assign_object", f, products=[building], relating_object=site)
        ifcopenshell.api.run("aggregate.assign_object", f, products=[storey], relating_object=building)

        products = []
        type_index: dict[str, int] = {}
        for eq in centered_equipment(layout):
            w, d, h = _shape_for(eq)
            product = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingElementProxy", name=_equipment_label(eq, type_index))
            rep = ifcopenshell.api.run("geometry.add_wall_representation", f, context=body, length=w, height=h, thickness=d)
            ifcopenshell.api.run("geometry.assign_representation", f, product=product, representation=rep)
            ifcopenshell.api.run("geometry.edit_object_placement", f, product=product, matrix=((1,0,0,eq.x),(0,1,0,eq.y),(0,0,1,0),(0,0,0,1)))
            products.append(product)

        if products:
            ifcopenshell.api.run("spatial.assign_container", f, products=products, relating_structure=storey)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        f.write(str(out))
        return True
    except Exception:
        return False
