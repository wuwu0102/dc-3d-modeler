from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Equipment:
    id: str
    type: str
    name: str
    x: float
    y: float
    z: float = 0
    width: float = 0
    depth: float = 0
    height: float = 0
    rotation_deg: float = 0
    note: str = ""
    power_kw: float = 0
    cooling_kw: float = 0
    status: str = "existing"


@dataclass
class CalibrationReference:
    name: str
    measured_model_length: float
    actual_length_m: float
    scale_factor: float


@dataclass
class DataCenterLayout:
    project_name: str
    unit: str = "m"
    scale_factor: float = 1.0
    calibration_reference: Optional[CalibrationReference] = None
    equipment: list[Equipment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DataCenterLayout":
        calibration_raw = data.get("calibration_reference")
        calibration = (
            CalibrationReference(**calibration_raw) if calibration_raw else None
        )
        equipment = [Equipment(**item) for item in data.get("equipment", [])]
        return cls(
            project_name=data["project_name"],
            unit=data.get("unit", "m"),
            scale_factor=data.get("scale_factor", 1.0),
            calibration_reference=calibration,
            equipment=equipment,
        )

    def scaled_copy(self, scale_factor: float) -> "DataCenterLayout":
        scaled_equipment = []
        for eq in self.equipment:
            scaled_equipment.append(
                Equipment(
                    id=eq.id,
                    type=eq.type,
                    name=eq.name,
                    x=eq.x * scale_factor,
                    y=eq.y * scale_factor,
                    z=eq.z * scale_factor,
                    width=eq.width * scale_factor,
                    depth=eq.depth * scale_factor,
                    height=eq.height * scale_factor,
                    rotation_deg=eq.rotation_deg,
                    note=eq.note,
                    power_kw=eq.power_kw,
                    cooling_kw=eq.cooling_kw,
                    status=eq.status,
                )
            )

        return DataCenterLayout(
            project_name=self.project_name,
            unit=self.unit,
            scale_factor=self.scale_factor * scale_factor,
            calibration_reference=self.calibration_reference,
            equipment=scaled_equipment,
        )
