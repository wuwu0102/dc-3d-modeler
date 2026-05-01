from __future__ import annotations

from datacenter_modeler.geometry_utils import get_centered_equipment, get_layout_bounds
from datacenter_modeler.models import DataCenterLayout, Equipment


def compute_layout_center(layout: DataCenterLayout) -> tuple[float, float]:
    bounds = get_layout_bounds(layout)
    return bounds["center_x"], bounds["center_y"]


def centered_equipment(layout: DataCenterLayout) -> list[Equipment]:
    return get_centered_equipment(layout)
