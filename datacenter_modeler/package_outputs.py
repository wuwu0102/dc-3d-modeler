from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

README_TEXT = """Data Center Modeling Outputs

This package contains CAD, BIM, and report deliverables exported from one command.
"""


def create_output_package(output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readme_path = out_dir / "README_OUTPUTS.txt"
    readme_path.write_text(README_TEXT, encoding="utf-8")

    required = [
        "datacenter_model.obj",
        "datacenter_model.mtl",
        "datacenter_floorplan.dxf",
        "datacenter_floorplan.svg",
        "heat_load_report.md",
        "datacenter_layout_scaled.json",
        "validation_report.md",
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
