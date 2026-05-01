from pathlib import Path
import subprocess
import sys


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
    ifc = out_dir / "datacenter_model.ifc"
    obj = out_dir / "datacenter_model.obj"
    heat_md = out_dir / "heat_load_report.md"

    assert dxf.exists()
    content = dxf.read_text(encoding="ascii", errors="ignore")
    assert "SECTION" in content
    assert "HEADER" in content
    assert "TABLES" in content
    assert "ENTITIES" in content
    assert "EOF" in content
    assert "LWPOLYLINE" not in content

    assert svg.exists()
    assert ifc.exists() or obj.exists()
    assert heat_md.exists()
