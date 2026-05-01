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
- datacenter_modeler/output/datacenter_model.ifc
- datacenter_modeler/output/datacenter_model.obj（IFC fallback）
- datacenter_modeler/output/datacenter_model.mtl（IFC fallback）
- datacenter_modeler/output/heat_load_report.json
- datacenter_modeler/output/heat_load_report.md

## CAD 使用方式
1. 下載 `datacenter_floorplan.dxf`
2. 用 AutoCAD 開啟
3. 若 AutoCAD 詢問版本，選 DXF R12 / ASCII

## Revit 使用方式
1. 優先匯入 `datacenter_model.ifc`
2. 若 IFC 無法使用，使用 `datacenter_model.obj` 作為 3D reference model
3. 第一版模型是設備 box model，不是正式 BIM family
