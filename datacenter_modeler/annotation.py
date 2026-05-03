from __future__ import annotations

from datacenter_modeler.io import load_layout, save_layout
from datacenter_modeler.models import Equipment

ALLOWED_TYPES = {"rack", "crac", "door", "cold_aisle", "hot_aisle", "wall", "room_boundary"}


def add_equipment(layout_path: str, **kwargs) -> None:
    t = kwargs["type"].lower()
    if t not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported equipment type: {t}")
    layout = load_layout(layout_path)
    eid = kwargs.get("id") or f"{t}_{len(layout.equipment)+1:03d}"
    layout.equipment.append(Equipment(id=eid, **kwargs))
    save_layout(layout, layout_path)
