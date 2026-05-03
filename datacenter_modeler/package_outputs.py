from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

README_TEXT = """Data Center Modeling Outputs

This package contains CAD, BIM, and report deliverables exported from one command.

Files:
- datacenter_floorplan.dxf: AutoCAD 2D 平面圖。
- datacenter_floorplan.svg: SVG 預覽圖。
- datacenter_model.obj + datacenter_model.mtl: 3D 模型幾何與材質。
- heat_load_report.md / heat_load_report.json: 熱負載報告（可讀版與資料版）。
- validation_report.md: 交付檢核摘要。
- datacenter_layout_scaled.json: 校正後的版面資料。
- datacenter_layout.json (optional): 原始版面輸入資料。
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
        "validation_report.md",
        "datacenter_layout_scaled.json",
        "README_OUTPUTS.txt",
    ]
    optional = ["datacenter_layout.json"]

    zip_path = out_dir / "datacenter_modeling_outputs.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for name in required + optional:
            p = out_dir / name
            if p.exists():
                zf.write(p, arcname=name)
    return zip_path
