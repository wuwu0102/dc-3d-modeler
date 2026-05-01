from __future__ import annotations

from datacenter_modeler.models import CalibrationReference, DataCenterLayout


def calculate_scale_factor(measured_model_length: float, actual_length_m: float) -> float:
    if measured_model_length <= 0:
        raise ValueError("measured_model_length must be > 0")
    if actual_length_m <= 0:
        raise ValueError("actual_length_m must be > 0")
    return actual_length_m / measured_model_length


def apply_calibration(
    layout: DataCenterLayout,
    reference_name: str,
    measured_model_length: float,
    actual_length_m: float,
) -> DataCenterLayout:
    scale_factor = calculate_scale_factor(measured_model_length, actual_length_m)
    scaled = layout.scaled_copy(scale_factor)
    scaled.calibration_reference = CalibrationReference(
        name=reference_name,
        measured_model_length=measured_model_length,
        actual_length_m=actual_length_m,
        scale_factor=scale_factor,
    )
    return scaled
