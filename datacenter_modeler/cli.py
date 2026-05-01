from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datacenter_modeler.export_dxf import export_floorplan_dxf, export_floorplan_svg
from datacenter_modeler.export_ifc import export_layout_ifc
from datacenter_modeler.export_obj import export_layout_obj
from datacenter_modeler.geometry_utils import get_centered_equipment
from datacenter_modeler.heat_load import calculate_heat_load, save_heat_report_json, save_heat_report_md
from datacenter_modeler.io import ensure_output_dir, load_layout, save_layout
from datacenter_modeler.package_outputs import create_output_package
from datacenter_modeler.scale import apply_calibration

BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_LAYOUT = BASE_DIR / "examples" / "sample_datacenter_layout.json"


def _warn_far_coordinates(layout) -> None:
    for eq in get_centered_equipment(layout):
        if abs(eq.x) > 100 or abs(eq.y) > 100:
            print(f"WARNING: {eq.name} centered coordinate is beyond 100m: ({eq.x:.3f}, {eq.y:.3f})")


def cmd_init() -> Path:
    out_dir = ensure_output_dir()
    target = out_dir / "datacenter_layout.json"
    shutil.copy2(EXAMPLE_LAYOUT, target)
    print(f"Initialized sample layout: {target}")
    return target


def cmd_calibrate(layout_path: Path, reference: str, measured: float, actual: float) -> Path:
    layout = load_layout(layout_path)
    scaled = apply_calibration(layout, reference, measured, actual)
    out_path = ensure_output_dir() / "datacenter_layout_scaled.json"
    save_layout(scaled, out_path)
    print(f"Calibrated layout saved: {out_path}")
    print(f"Scale factor: {scaled.scale_factor:.6f}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Center Modeler CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--layout", required=True)
    calibrate.add_argument("--reference", required=True)
    calibrate.add_argument("--measured", required=True, type=float)
    calibrate.add_argument("--actual", required=True, type=float)
    export_dxf = subparsers.add_parser("export-dxf")
    export_dxf.add_argument("--layout", required=True)
    export_svg = subparsers.add_parser("export-svg")
    export_svg.add_argument("--layout", required=True)
    export_ifc = subparsers.add_parser("export-ifc")
    export_ifc.add_argument("--layout", required=True)
    heat_report = subparsers.add_parser("heat-report")
    heat_report.add_argument("--layout", required=True)
    subparsers.add_parser("demo-all")
    args = parser.parse_args()

    if args.command == "demo-all":
        out_dir = ensure_output_dir()
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "init":
        cmd_init()
    elif args.command == "calibrate":
        cmd_calibrate(Path(args.layout), args.reference, args.measured, args.actual)
    elif args.command == "export-dxf":
        layout = load_layout(args.layout)
        export_floorplan_dxf(layout, ensure_output_dir() / "datacenter_floorplan.dxf")
    elif args.command == "export-svg":
        layout = load_layout(args.layout)
        export_floorplan_svg(layout, ensure_output_dir() / "datacenter_floorplan.svg")
    elif args.command == "export-ifc":
        layout = load_layout(args.layout)
        ok = export_layout_ifc(layout, ensure_output_dir() / "datacenter_model.ifc")
        if not ok:
            print("IFC failed, use OBJ fallback.")
    elif args.command == "heat-report":
        layout = load_layout(args.layout)
        report = calculate_heat_load(layout)
        out_dir = ensure_output_dir()
        save_heat_report_json(report, out_dir / "heat_load_report.json")
        save_heat_report_md(report, out_dir / "heat_load_report.md")
    elif args.command == "demo-all":
        layout_path = cmd_init()
        scaled_path = cmd_calibrate(layout_path, "Door01", 1.72, 2.10)
        layout = load_layout(scaled_path)
        _warn_far_coordinates(layout)

        export_floorplan_dxf(layout, out_dir / "datacenter_floorplan.dxf")
        export_floorplan_svg(layout, out_dir / "datacenter_floorplan.svg")

        report = calculate_heat_load(layout)
        save_heat_report_json(report, out_dir / "heat_load_report.json")
        save_heat_report_md(report, out_dir / "heat_load_report.md")

        export_layout_obj(layout, out_dir / "datacenter_model.obj", out_dir / "datacenter_model.mtl")
        if not export_layout_ifc(layout, out_dir / "datacenter_model.ifc"):
            print("IFC failed, use OBJ fallback.")

        create_output_package(out_dir)
        print("\nCAD 2D:\n- datacenter_floorplan.dxf\n- datacenter_floorplan.svg")
        print("\nRevit / 3D:\n- datacenter_model.ifc\n- datacenter_model.obj\n- datacenter_model.mtl")
        print("\nReports:\n- heat_load_report.md\n- heat_load_report.json")
        print("\nPackage:\n- datacenter_modeling_outputs.zip")


if __name__ == "__main__":
    main()
