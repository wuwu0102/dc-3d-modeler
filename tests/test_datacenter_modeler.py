from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile


def test_demo_all_generates_expected_files_and_dxf_rules():
    result = subprocess.run(
        [sys.executable, "-m", "datacenter_modeler.cli", "demo-all"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out_dir = Path("datacenter_modeler/output")
    dxf = out_dir / "datacenter_floorplan.dxf"
    svg = out_dir / "datacenter_floorplan.svg"
    obj = out_dir / "datacenter_model.obj"
    mtl = out_dir / "datacenter_model.mtl"
    heat_md = out_dir / "heat_load_report.md"
    zip_path = out_dir / "datacenter_modeling_outputs.zip"

    assert dxf.exists()
    assert svg.exists()
    assert obj.exists()
    assert mtl.exists()
    assert heat_md.exists()
    assert zip_path.exists()

    dxf_content = dxf.read_text(encoding="ascii", errors="ignore")
    assert "LWPOLYLINE" not in dxf_content

    obj_content = obj.read_text(encoding="utf-8", errors="ignore")
    assert "o R01" in obj_content
    assert "\nv " in obj_content
    assert "\nf " in obj_content

    with ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    for expected in {
        "README_OUTPUTS.txt",
        "datacenter_floorplan.dxf",
        "datacenter_floorplan.svg",
        "datacenter_model.obj",
        "datacenter_model.mtl",
        "heat_load_report.md",
    }:
        assert expected in names


def test_scan_to_cad_outputs_boundary_preview_only():
    scan_path = Path("cube.obj")
    scan_path.write_text(
        "\n".join(
            [
                "v -1 -1 0",
                "v 1 -1 0",
                "v 1 1 0",
                "v -1 1 0",
                "v -1 -1 1",
                "v 1 -1 1",
                "v 1 1 1",
                "v -1 1 1",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "datacenter_modeler.cli", "scan-to-cad", "--scan", str(scan_path), "--reference-type", "rack-width", "--measured", "1", "--actual", "600"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out_dir = Path("datacenter_modeler/output")
    svg_content = (out_dir / "datacenter_floorplan.svg").read_text(encoding="utf-8")
    validation = (out_dir / "validation_report.md").read_text(encoding="utf-8")
    assert "Scan Boundary Preview" in svg_content
    assert 'fill="white"' in svg_content
    assert "Legend: Rack / CRAC" not in svg_content
    assert "Scan imported but equipment annotation is not completed." in validation
