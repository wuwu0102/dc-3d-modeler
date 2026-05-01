from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

README_TEXT = """Data Center Modeling Outputs

1. datacenter_floorplan.dxf: AutoCAD 2D floor plan, can be saved as DWG.
2. datacenter_floorplan.svg: Browser preview image.
3. datacenter_model.ifc: Revit / BIM 3D model, preferred when available.
4. datacenter_model.obj + datacenter_model.mtl: 3D fallback model.
5. heat_load_report.md: Heat load report.
6. datacenter_layout_scaled.json: Engineering source data.
7. If Revit IFC display looks incorrect, use 3D View + Zoom Extents first; if still incorrect, use OBJ as reference.
"""


def create_output_package(output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readme_path = out_dir / "README_OUTPUTS.txt"
    readme_path.write_text(README_TEXT, encoding="utf-8")

    required = [
        "datacenter_floorplan.svg",
        "datacenter_floorplan.dxf",
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
            file_path = out_dir / name
            if file_path.exists():
                zf.write(file_path, arcname=name)
    return zip_path
