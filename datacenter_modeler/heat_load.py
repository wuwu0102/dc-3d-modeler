from __future__ import annotations

import json
from pathlib import Path

from datacenter_modeler.models import DataCenterLayout


RT_KW = 3.517


def _load_level(power_kw: float) -> str:
    if power_kw < 5:
        return "low"
    if power_kw < 15:
        return "medium"
    if power_kw <= 40:
        return "high"
    return "extreme"


def calculate_heat_load(layout: DataCenterLayout) -> dict:
    racks = [e for e in layout.equipment if e.type.lower() == "rack"]
    cracs = [e for e in layout.equipment if e.type.lower() == "crac"]

    total_it_load_kw = sum(r.power_kw for r in racks)
    total_cooling_capacity_kw = sum(c.cooling_kw for c in cracs)

    rack_loads = [
        {
            "id": r.id,
            "name": r.name,
            "power_kw": r.power_kw,
            "load_level": _load_level(r.power_kw),
        }
        for r in racks
    ]

    return {
        "project_name": layout.project_name,
        "total_it_load_kw": total_it_load_kw,
        "total_cooling_capacity_kw": total_cooling_capacity_kw,
        "total_cooling_required_rt": total_it_load_kw / RT_KW,
        "total_cooling_capacity_rt": total_cooling_capacity_kw / RT_KW,
        "cooling_margin_kw": total_cooling_capacity_kw - total_it_load_kw,
        "rack_count": len(racks),
        "crac_count": len(cracs),
        "rack_loads": rack_loads,
    }


def save_heat_report_json(report: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def save_heat_report_md(report: dict, path: str | Path) -> None:
    margin_kw = report["cooling_margin_kw"]
    if margin_kw < 0:
        deficit_rt = abs(margin_kw) / RT_KW
        interpretation = f"目前空調容量不足，冷卻能力缺口為 {abs(margin_kw):.2f} kW。若以 RT 換算，約不足 {deficit_rt:.2f} RT。"
    else:
        interpretation = f"目前空調容量高於 IT 熱負載需求，冷卻餘裕為 {margin_kw:.2f} kW。"

    lines = [
        "# Heat Load Report / 熱負載報告",
        "",
        "## Project Summary / 專案摘要",
        f"- Project: {report['project_name']}",
        f"- Rack Count: {report['rack_count']}",
        f"- CRAC Count: {report['crac_count']}",
        f"- Total IT Load: {report['total_it_load_kw']:.2f} kW",
        f"- Required Cooling: {report['total_cooling_required_rt']:.2f} RT",
        f"- Cooling Capacity: {report['total_cooling_capacity_kw']:.2f} kW ({report['total_cooling_capacity_rt']:.2f} RT)",
        f"- Cooling Margin: {margin_kw:.2f} kW",
        "",
        "## Engineering Interpretation / 工程判讀",
        interpretation,
        "",
        "## Rack Summary / 機櫃摘要",
        "",
        "| Rack ID | Name | Power (kW) | Load Level |",
        "|---|---|---:|---|",
    ]

    for rack in report["rack_loads"]:
        lines.append(f"| {rack['id']} | {rack['name']} | {rack['power_kw']:.2f} | {rack['load_level']} |")

    lines.extend([
        "",
        "## Notes / 注意事項",
        "- 本報告為初步估算。",
        "- RT 換算使用 1 RT = 3.517 kW。",
        "- 實際設計仍需考慮顯熱比、送回風條件、冗餘架構、氣流組織與現場量測。",
    ])

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
