from __future__ import annotations

from dataclasses import replace

from datacenter_modeler.models import DataCenterLayout, Equipment


def compute_layout_center(layout: DataCenterLayout) -> tuple[float, float]:
    if not layout.equipment:
        return 0.0, 0.0
    min_x = min(eq.x - eq.width / 2 for eq in layout.equipment)
    max_x = max(eq.x + eq.width / 2 for eq in layout.equipment)
    min_y = min(eq.y - eq.depth / 2 for eq in layout.equipment)
    max_y = max(eq.y + eq.depth / 2 for eq in layout.equipment)
    return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0


def centered_equipment(layout: DataCenterLayout) -> list[Equipment]:
    cx, cy = compute_layout_center(layout)
    centered: list[Equipment] = []
    for eq in layout.equipment:
        centered.append(replace(eq, x=eq.x - cx, y=eq.y - cy, z=0.0))
    return centered
