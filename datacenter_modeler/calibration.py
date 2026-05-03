from __future__ import annotations

from datacenter_modeler.models import CalibrationReference, DataCenterLayout

ALLOWED_REFERENCE_TYPES = {"door-height", "door-width", "rack-width"}


def calculate_scale_factor(measured: float, actual: float) -> float:
    if measured <= 0:
        raise ValueError("measured must be > 0")
    if actual <= 0:
        raise ValueError("actual must be > 0")
    return actual / measured


def apply_calibration(
    layout: DataCenterLayout,
    reference_type: str,
    measured: float,
    actual: float,
) -> DataCenterLayout:
    if reference_type not in ALLOWED_REFERENCE_TYPES:
        raise ValueError(f"unsupported reference_type: {reference_type}")

    scale_factor = calculate_scale_factor(measured, actual)
    scaled = layout.scaled_copy(scale_factor)
    scaled.calibration_reference = CalibrationReference(
        name=reference_type,
        measured_model_length=measured,
        actual_length_m=actual,
        scale_factor=scale_factor,
    )
    return scaled
