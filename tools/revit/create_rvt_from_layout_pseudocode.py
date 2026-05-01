"""Pseudocode for creating RVT content from datacenter_layout_scaled.json.

This script is intentionally a draft for Revit/Dynamo environments on Windows.
It is not expected to run in Codespaces.
"""

import json


def load_layout(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_directshape_box(doc, category, name, x, y, z, width, depth, height):
    """Pseudo API call: create DirectShape box geometry in Revit."""
    # solid = GeometryCreationUtilities.CreateExtrusionGeometry(...)
    # ds = DirectShape.CreateElement(doc, category.Id)
    # ds.Name = name
    # ds.SetShape([solid])
    # return ds
    raise NotImplementedError


def set_equipment_parameters(ds, equipment):
    """Pseudo API call: write custom parameters onto DirectShape."""
    for key in ["name", "type", "power_kw", "cooling_kw", "note"]:
        value = equipment.get(key, "")
        # ds.LookupParameter(key).Set(str(value))
        _ = value


def build_revit_model_from_layout(doc, layout_json_path):
    data = load_layout(layout_json_path)
    equipment_list = data.get("equipment", [])

    # with Transaction(doc, "Create Data Center Equipment") as tx:
    #     tx.Start()
    for equipment in equipment_list:
        name = equipment.get("name") or equipment.get("id") or "Equipment"
        x = float(equipment.get("x", 0.0))
        y = float(equipment.get("y", 0.0))
        z = float(equipment.get("z", 0.0))
        width = float(equipment.get("width", 0.6))
        depth = float(equipment.get("depth", 1.2))
        height = float(equipment.get("height", 2.0))

        ds = create_directshape_box(
            doc=doc,
            category="GenericModel",
            name=name,
            x=x,
            y=y,
            z=z,
            width=width,
            depth=depth,
            height=height,
        )
        set_equipment_parameters(ds, equipment)
    #     tx.Commit()
