from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datacenter_modeler.export_dxf import export_floorplan_dxf, export_floorplan_svg
from datacenter_modeler.export_ifc import export_layout_ifc
from datacenter_modeler.heat_load import calculate_heat_load, save_heat_report_json, save_heat_report_md
from datacenter_modeler.io import ensure_output_dir, load_layout, save_layout
from datacenter_modeler.scale import apply_calibration

BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_LAYOUT = BASE_DIR / "examples" / "sample_datacenter_layout.json"


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

    if args.command == "init":
        cmd_init()
    elif args.command == "calibrate":
        cmd_calibrate(Path(args.layout), args.reference, args.measured, args.actual)
    elif args.command == "export-dxf":
        layout = load_layout(args.layout)
        out_dir = ensure_output_dir()
        out = out_dir / "datacenter_floorplan.dxf"
        out_r12 = out_dir / "datacenter_floorplan_r12.dxf"
        ok_main = export_floorplan_dxf(layout, out)
        ok_r12 = export_floorplan_dxf(layout, out_r12)
        if ok_main:
            print(f"DXF exported: {out}")
        if ok_r12:
            print(f"DXF (R12) exported: {out_r12}")
    elif args.command == "export-svg":
        layout = load_layout(args.layout)
        out = ensure_output_dir() / "datacenter_floorplan.svg"
        export_floorplan_svg(layout, out)
        print(f"SVG exported: {out}")
    elif args.command == "export-ifc":
        layout = load_layout(args.layout)
        out = ensure_output_dir() / "datacenter_model.ifc"
        ok = export_layout_ifc(layout, out)
        if ok:
            print(f"IFC exported: {out}")
    elif args.command == "heat-report":
        layout = load_layout(args.layout)
        report = calculate_heat_load(layout)
        out_dir = ensure_output_dir()
        json_path = out_dir / "heat_load_report.json"
        md_path = out_dir / "heat_load_report.md"
        save_heat_report_json(report, json_path)
        save_heat_report_md(report, md_path)
        print(f"Heat report JSON: {json_path}")
        print(f"Heat report Markdown: {md_path}")
    elif args.command == "demo-all":
        output_files = []
        layout_path = cmd_init()
        output_files.append(layout_path)
        scaled_path = cmd_calibrate(layout_path, "Door01", 1.72, 2.10)
        output_files.append(scaled_path)

        layout = load_layout(scaled_path)

        out_dir = ensure_output_dir()
        dxf_path = out_dir / "datacenter_floorplan.dxf"
        dxf_r12_path = out_dir / "datacenter_floorplan_r12.dxf"
        if export_floorplan_dxf(layout, dxf_path):
            output_files.append(dxf_path)
        if export_floorplan_dxf(layout, dxf_r12_path):
            output_files.append(dxf_r12_path)

        svg_path = ensure_output_dir() / "datacenter_floorplan.svg"
        export_floorplan_svg(layout, svg_path)
        output_files.append(svg_path)

        report = calculate_heat_load(layout)
        json_path = ensure_output_dir() / "heat_load_report.json"
        md_path = ensure_output_dir() / "heat_load_report.md"
        save_heat_report_json(report, json_path)
        save_heat_report_md(report, md_path)
        output_files.extend([json_path, md_path])

        ifc_path = ensure_output_dir() / "datacenter_model.ifc"
        if export_layout_ifc(layout, ifc_path):
            output_files.append(ifc_path)
        else:
            print("Skipping IFC export due to missing dependency.")

        print("Generated output files:")
        for p in output_files:
            print(f"- {p}")


if __name__ == "__main__":
    main()
