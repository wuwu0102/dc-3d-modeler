from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datacenter_modeler.annotation import add_equipment
from datacenter_modeler.calibration import apply_calibration
from datacenter_modeler.core_engine_adapter import find_latest_scan_output
from datacenter_modeler.export_dxf import export_floorplan_dxf, export_floorplan_svg
from datacenter_modeler.export_ifc import export_layout_ifc
from datacenter_modeler.export_obj import export_layout_obj
from datacenter_modeler.heat_load import calculate_heat_load, save_heat_report_json, save_heat_report_md
from datacenter_modeler.import_scan import scan_to_layout
from datacenter_modeler.io import ensure_output_dir, load_layout, save_layout
from datacenter_modeler.package_outputs import create_output_package
from datacenter_modeler.validation import validate_outputs

BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_LAYOUT = BASE_DIR / "examples" / "sample_datacenter_layout.json"


def _export_all(layout):
    out = ensure_output_dir()
    export_floorplan_dxf(layout, out / "datacenter_floorplan.dxf")
    export_floorplan_svg(layout, out / "datacenter_floorplan.svg")
    export_layout_obj(layout, out / "datacenter_model.obj", out / "datacenter_model.mtl")
    export_layout_ifc(layout, out / "datacenter_model.ifc")
    report = calculate_heat_load(layout)
    save_heat_report_md(report, out / "heat_load_report.md")
    save_heat_report_json(report, out / "heat_load_report.json")
    (out / "validation_report.md").write_text(validate_outputs(layout, out), encoding="utf-8")
    create_output_package(out)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="command", required=True)
    s.add_parser("demo-all")
    cal = s.add_parser("calibrate")
    cal.add_argument("--layout", required=True)
    cal.add_argument("--reference-type", required=True)
    cal.add_argument("--measured", type=float, required=True)
    cal.add_argument("--actual", type=float, required=True)
    stc = s.add_parser("scan-to-cad")
    stc.add_argument("--scan", required=True)
    stc.add_argument("--reference-type", required=True)
    stc.add_argument("--measured", type=float, required=True)
    stc.add_argument("--actual", type=float, required=True)
    s.add_parser("find-scan-output")
    ae = s.add_parser("add-equipment")
    ae.add_argument("--layout", required=True)
    for a in ["type", "name", "x", "y", "z", "width", "depth", "height", "rotation-deg", "power-kw", "cooling-kw", "note"]:
        ae.add_argument(f"--{a}", required=a in {"type", "name", "x", "y", "z", "width", "depth", "height"})
    ea = s.add_parser("export-all")
    ea.add_argument("--layout", required=True)
    args = p.parse_args()

    out = ensure_output_dir()
    if args.command == "demo-all":
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EXAMPLE_LAYOUT, out / "datacenter_layout.json")
        layout = load_layout(out / "datacenter_layout.json")
        layout = apply_calibration(layout, "door-height", 1.72, 2.10)
        save_layout(layout, out / "datacenter_layout_scaled.json")
        _export_all(layout)
        print("CAD 2D\nRevit / 3D\nReports\nPackage")
    elif args.command == "calibrate":
        layout = load_layout(args.layout)
        scaled = apply_calibration(layout, args.reference_type, args.measured, args.actual)
        save_layout(scaled, out / "datacenter_layout_scaled.json")
    elif args.command == "scan-to-cad":
        sp = Path(args.scan)
        if not sp.exists():
            print("Scan file not found.")
            return
        try:
            layout = scan_to_layout(sp, out / "datacenter_layout_from_scan.json")
        except ValueError as e:
            print(str(e)); return
        scaled = apply_calibration(layout, args.reference_type, args.measured, args.actual)
        save_layout(scaled, out / "datacenter_layout_scaled.json")
        _export_all(scaled)
    elif args.command == "find-scan-output":
        res = find_latest_scan_output(".")
        if not res:
            print("No scan output found. Please run core_engine demo first, or put scan files in datacenter_modeler/input/")
            return
        print(f"Path: {res['path']}\nFormat: {res['format']}\nModified: {res['modified_time']}")
    elif args.command == "add-equipment":
        add_equipment(
            args.layout,
            type=args.type,
            name=args.name,
            x=float(args.x), y=float(args.y), z=float(args.z),
            width=float(args.width), depth=float(args.depth), height=float(args.height),
            rotation_deg=float(args.rotation_deg or 0), power_kw=float(args.power_kw or 0), cooling_kw=float(args.cooling_kw or 0), note=args.note or "",
        )
    elif args.command == "export-all":
        layout = load_layout(args.layout)
        _export_all(layout)


if __name__ == "__main__":
    main()
