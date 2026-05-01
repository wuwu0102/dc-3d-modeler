from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

README_TEXT = """Data Center Modeling Outputs

1. AutoCAD: Open datacenter_floorplan.dxf directly, then save as DWG if needed.
2. Revit: Prefer datacenter_model.ifc.
3. If Revit cannot see IFC geometry, switch to 3D View and run Zoom Extents. If still not visible, use datacenter_model.obj + datacenter_model.mtl as fallback 3D reference.
4. heat_load_report.md is the heat-load report.
5. datacenter_layout_scaled.json is the engineering source data.
"""


def create_output_package(output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readme_path = out_dir / "README_OUTPUTS.txt"
    readme_path.write_text(README_TEXT, encoding="utf-8")

    required = [
        "datacenter_floorplan.dxf",
        "datacenter_floorplan.svg",
        "datacenter_model.obj",
        "datacenter_model.mtl",
        "heat_load_report.md",
        "heat_load_report.json",
        "datacenter_layout.json",
        "datacenter_layout_scaled.json",
        "README_OUTPUTS.txt",
    ]
    optional = ["datacenter_model.ifc"]

    zip_path = out_dir / "datacenter_modeling_outputs.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for name in required + optional:
            p = out_dir / name
            if p.exists():
                zf.write(p, arcname=name)
    return zip_path
