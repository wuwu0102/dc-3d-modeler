from __future__ import annotations

from pathlib import Path

from datacenter_modeler.geometry_utils import get_centered_equipment
from datacenter_modeler.models import DataCenterLayout, Equipment


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
    w = eq.width if eq.width > 0 else 0.6
    d = eq.depth if eq.depth > 0 else 1.2
    h = eq.height
    if t == "rack":
        h = h if h > 0 else 2.2
    elif t == "crac":
        h = h if h > 0 else 2.0
    elif t == "door":
        h = h if h > 0 else 2.1
    elif "aisle" in t:
        h = 0.05
    else:
        h = h if h > 0 else 1.0
    return w, d, h


def export_layout_ifc(layout: DataCenterLayout, path: str | Path) -> bool:
    try:
        import ifcopenshell
        import ifcopenshell.api
    except Exception:
        return False

    try:
        f = ifcopenshell.api.run("project.create_file", version="IFC2X3")
        project = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcProject", name=layout.project_name)
        ifcopenshell.api.run("unit.assign_unit", f)

        model = ifcopenshell.api.run("context.add_context", f, context_type="Model")
        body = ifcopenshell.api.run("context.add_context", f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model)

        site = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcSite", name="Site")
        building = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuilding", name="DataCenter")
        storey = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Level1")
        storey.Elevation = 0.0

        for spatial in (site, building, storey):
            ifcopenshell.api.run("geometry.edit_object_placement", f, product=spatial, matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

        ifcopenshell.api.run("aggregate.assign_object", f, products=[site], relating_object=project)
        ifcopenshell.api.run("aggregate.assign_object", f, products=[building], relating_object=site)
        ifcopenshell.api.run("aggregate.assign_object", f, products=[storey], relating_object=building)

        products = []
        idx: dict[str, int] = {}
        for eq in get_centered_equipment(layout):
            name = _equipment_label(eq, idx)
            w, d, h = _shape_for(eq)
            product = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingElementProxy", name=name)
            rep = ifcopenshell.api.run("geometry.add_profile_representation", f, context=body, profile=ifcopenshell.api.run("profile.create_rectangle_profile", f, x_dim=w, y_dim=d), depth=h)
            ifcopenshell.api.run("geometry.assign_representation", f, product=product, representation=rep)
            ifcopenshell.api.run("geometry.edit_object_placement", f, product=product, matrix=((1, 0, 0, eq.x), (0, 1, 0, eq.y), (0, 0, 1, eq.z), (0, 0, 0, 1)))
            ifcopenshell.api.run("pset.add_pset", f, product=product, name="Pset_DCModeler")
            ifcopenshell.api.run("pset.edit_pset", f, pset=product.IsDefinedBy[-1].RelatingPropertyDefinition, properties={"type": eq.type, "power_kw": eq.power_kw, "cooling_kw": eq.cooling_kw, "note": eq.note})
            products.append(product)

        if products:
            ifcopenshell.api.run("spatial.assign_container", f, products=products, relating_structure=storey)

        f.write(str(Path(path)))
        txt = Path(path).read_text(encoding="utf-8", errors="ignore").upper()
        if "IFCPROJECT" not in txt or "IFCBUILDINGELEMENTPROXY" not in txt or "R01" not in txt:
            return False
        if "IFCEXTRUDEDAREASOLID" not in txt and "IFCFACETEDBREP" not in txt:
            return False
        return True
    except Exception:
        return False
