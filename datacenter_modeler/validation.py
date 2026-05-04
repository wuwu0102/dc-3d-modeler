from __future__ import annotations

from pathlib import Path

from datacenter_modeler.models import DataCenterLayout


def validate_outputs(layout: DataCenterLayout, output_dir: str | Path) -> str:
    out_dir = Path(output_dir)
    errors, warnings = [], []
    if layout.scale_factor <= 0:
        errors.append("Missing or invalid scale_factor")
    if layout.calibration_reference is None:
        errors.append("Missing calibration_reference")

    for eq in layout.equipment:
        if abs(eq.x) > 100 or abs(eq.y) > 100:
            errors.append(f"Coordinate out of range: {eq.name}")
        if eq.type == "rack":
            if not (0.4 <= eq.width <= 0.8 and 0.8 <= eq.depth <= 1.4 and 1.8 <= eq.height <= 2.4):
                errors.append(f"Rack size unreasonable: {eq.name}")
        if eq.type == "door" and not (1.8 <= eq.height <= 2.4):
            errors.append(f"Door height unreasonable: {eq.name}")

    for req in ["datacenter_floorplan.dxf", "datacenter_floorplan.svg", "datacenter_model.obj"]:
        if not (out_dir / req).exists():
            errors.append(f"Missing output: {req}")
    if not (out_dir / "datacenter_model.ifc").exists():
        warnings.append("IFC output missing (warning only)")
    has_manual_equipment = any(eq.type in {"rack", "crac", "door"} for eq in layout.equipment)
    if layout.source_mode == "scan" and not has_manual_equipment:
        warnings.append("Scan imported but equipment annotation is not completed.")

    lines = [
        "# Validation Report",
        "",
        "## Metadata",
        f"- source_mode: {layout.source_mode}",
        f"- scan input path: {layout.scan_input_path or 'N/A'}",
        f"- equipment manually annotated: {'yes' if has_manual_equipment else 'no'}",
        "",
        "## Errors",
    ] + ([f"- {e}" for e in errors] or ["- None"]) + ["", "## Warnings"] + ([f"- {w}" for w in warnings] or ["- None"])
    return "\n".join(lines) + "\n"
