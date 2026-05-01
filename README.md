# DC 3D Modeler

這是一個 AI-assisted data center modeling tool。  
`core_engine` 使用 lingbot-map 作為 3D reconstruction engine。  
`datacenter_modeler` 是工程語意層，用來做：
- scale calibration
- rack / CRAC / door / aisle modeling
- heat load calculation
- CAD DXF export
- Revit IFC export

## Quick Start

```bash
pip install -r requirements.txt
python -m datacenter_modeler.cli demo-all
```

## 輸出檔案
- datacenter_modeler/output/datacenter_layout.json
- datacenter_modeler/output/datacenter_layout_scaled.json
- datacenter_modeler/output/datacenter_floorplan.dxf（R12 ASCII）
- datacenter_modeler/output/datacenter_floorplan.svg
- datacenter_modeler/output/datacenter_model.obj
- datacenter_modeler/output/datacenter_model.mtl
- datacenter_modeler/output/datacenter_model.ifc（視 ifcopenshell 是否可用）
- datacenter_modeler/output/heat_load_report.json
- datacenter_modeler/output/heat_load_report.md

## 輸出檔案等級

### 雲端可直接產生
- SVG
- DXF
- OBJ/MTL
- IFC（視 ifcopenshell）

### Windows 本機轉換
- DXF → DWG：使用 AutoCAD 另存（或 `tools/windows/convert_dxf_to_dwg.scr`）
- JSON/OBJ → RVT：需 Revit API / Dynamo / Revit Add-in

## CAD 使用方式（DWG）
1. 下載 `datacenter_floorplan.dxf`
2. 用 AutoCAD 開啟
3. 另存成 `datacenter_floorplan.dwg`，或執行 `tools/windows/convert_dxf_to_dwg.scr`

## Revit 使用方式
1. 可匯入 `datacenter_floorplan.dxf` 作為 2D 底圖
2. 可使用 `datacenter_model.obj` / `datacenter_model.mtl` 作為 3D reference model
3. 需要真正 `.rvt` 時，請參考 `tools/revit/README_RVT_WORKFLOW.md` 與 `tools/revit/create_rvt_from_layout_README.md`

## One-click output package

Run:

`python -m datacenter_modeler.cli demo-all`

Download:

`datacenter_modeler/output/datacenter_modeling_outputs.zip`

The ZIP includes CAD / Revit / 3D / report files for delivery.

- AutoCAD: open `datacenter_floorplan.dxf`, then save as DWG if needed.
- Revit: open `datacenter_model.ifc` first.
- If IFC is unstable, use `datacenter_model.obj` + `datacenter_model.mtl` as 3D reference.
- `heat_load_report.md` is the heat load report.


## One command

```bash
pip install -r requirements.txt
python -m datacenter_modeler.cli demo-all
```
