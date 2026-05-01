from pathlib import Path

import pytest

from datacenter_modeler.export_dxf import export_floorplan_dxf, export_floorplan_svg
from datacenter_modeler.export_ifc import export_layout_ifc
from datacenter_modeler.heat_load import calculate_heat_load
from datacenter_modeler.io import load_layout
from datacenter_modeler.scale import calculate_scale_factor


def test_calculate_scale_factor():
    assert calculate_scale_factor(1.72, 2.10) == pytest.approx(2.10 / 1.72)


def test_sample_layout_loadable():
    layout = load_layout("datacenter_modeler/examples/sample_datacenter_layout.json")
    assert layout.project_name == "Sample Data Center Room"
    assert len(layout.equipment) >= 24


def test_heat_load_total_it_kw():
    layout = load_layout("datacenter_modeler/examples/sample_datacenter_layout.json")
    report = calculate_heat_load(layout)
    assert report["total_it_load_kw"] == 200


def test_svg_export(tmp_path: Path):
    layout = load_layout("datacenter_modeler/examples/sample_datacenter_layout.json")
    out = tmp_path / "floorplan.svg"
    export_floorplan_svg(layout, out)
    assert out.exists()


def test_dxf_export(tmp_path: Path):
    layout = load_layout("datacenter_modeler/examples/sample_datacenter_layout.json")
    out = tmp_path / "floorplan.dxf"
    export_floorplan_dxf(layout, out)
    assert out.exists()


def test_ifc_export_optional_dependency(tmp_path: Path, capsys):
    layout = load_layout("datacenter_modeler/examples/sample_datacenter_layout.json")
    out = tmp_path / "model.ifc"
    ok = export_layout_ifc(layout, out)
    if ok:
        assert out.exists()
    else:
        captured = capsys.readouterr()
        assert "IfcOpenShell is required" in captured.out


def test_dxf_export_r12_and_structure(tmp_path: Path):
    layout = load_layout("datacenter_modeler/examples/sample_datacenter_layout.json")
    out1 = tmp_path / "datacenter_floorplan.dxf"
    out2 = tmp_path / "datacenter_floorplan_r12.dxf"
    export_floorplan_dxf(layout, out1)
    export_floorplan_dxf(layout, out2)
    assert out1.exists()
    assert out2.exists()
    content = out1.read_text(encoding="utf-8", errors="ignore")
    assert "SECTION" in content
    assert "ENTITIES" in content
    assert "EOF" in content


def test_svg_contains_title_and_legend(tmp_path: Path):
    layout = load_layout("datacenter_modeler/examples/sample_datacenter_layout.json")
    out = tmp_path / "datacenter_floorplan.svg"
    export_floorplan_svg(layout, out)
    content = out.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "legend" in content.lower()
    assert "Sample Data Center Room" in content


def test_heat_load_markdown_contains_chinese_sections(tmp_path: Path):
    from datacenter_modeler.heat_load import save_heat_report_md

    report = {
        "project_name": "Test",
        "rack_count": 2,
        "crac_count": 1,
        "total_it_load_kw": 100.0,
        "total_cooling_required_rt": 28.43,
        "total_cooling_capacity_kw": 80.0,
        "total_cooling_capacity_rt": 22.74,
        "cooling_margin_kw": -20.0,
        "rack_loads": [],
    }
    out = tmp_path / "heat_load_report.md"
    save_heat_report_md(report, out)
    content = out.read_text(encoding="utf-8")
    assert "熱負載報告" in content
    assert "工程判讀" in content
    assert "目前空調容量不足" in content
