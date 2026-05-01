from __future__ import annotations

from dataclasses import replace

from datacenter_modeler.models import DataCenterLayout, Equipment


DEFAULT_SIZE = (0.6, 1.2, 2.2)


def _safe_size(eq: Equipment) -> tuple[float, float, float]:
    w = eq.width if eq.width > 0 else DEFAULT_SIZE[0]
    d = eq.depth if eq.depth > 0 else DEFAULT_SIZE[1]
    h = eq.height if eq.height > 0 else DEFAULT_SIZE[2]
    return w, d, h


def get_layout_bounds(layout: DataCenterLayout) -> dict[str, float]:
    if not layout.equipment:
        return {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0, "center_x": 0.0, "center_y": 0.0, "center_z": 0.0}

    min_x = min(eq.x - _safe_size(eq)[0] / 2 for eq in layout.equipment)
    max_x = max(eq.x + _safe_size(eq)[0] / 2 for eq in layout.equipment)
    min_y = min(eq.y - _safe_size(eq)[1] / 2 for eq in layout.equipment)
    max_y = max(eq.y + _safe_size(eq)[1] / 2 for eq in layout.equipment)
    min_z = min(max(eq.z, 0.0) for eq in layout.equipment)
    max_z = max(max(eq.z, 0.0) + _safe_size(eq)[2] for eq in layout.equipment)

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z,
        "center_x": (min_x + max_x) / 2.0,
        "center_y": (min_y + max_y) / 2.0,
        "center_z": (min_z + max_z) / 2.0,
    }


def get_centered_equipment(layout: DataCenterLayout) -> list[Equipment]:
    bounds = get_layout_bounds(layout)
    cx = bounds["center_x"]
    cy = bounds["center_y"]
    centered: list[Equipment] = []
    for eq in layout.equipment:
        centered.append(replace(eq, x=eq.x - cx, y=eq.y - cy, z=max(eq.z, 0.0)))
    return centered
