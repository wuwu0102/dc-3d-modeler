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
- datacenter_modeler/output/datacenter_floorplan.dxf
- datacenter_modeler/output/datacenter_floorplan.svg
- datacenter_modeler/output/datacenter_model.ifc
- datacenter_modeler/output/heat_load_report.json
- datacenter_modeler/output/heat_load_report.md

## 使用概念
1. 使用手機繞機房走一圈拍攝
2. core_engine 產生 3D reconstruction
3. datacenter_modeler 匯入或建立 layout
4. 使用門寬/門高做比例校正
5. 輸出 CAD 2D 與 Revit IFC
6. 人工複核尺寸誤差

## 限制
- 第一版不是正式 BIM / Revit / CFD 替代品
- 手機影像重建會有比例與幾何誤差
- 需要用門、柱距、地磚、機櫃標準尺寸等已知尺寸校正
- DXF 是 2D top-view floor plan
- IFC 第一版可能是 semantic placeholder，後續可升級成真實 3D box geometry
