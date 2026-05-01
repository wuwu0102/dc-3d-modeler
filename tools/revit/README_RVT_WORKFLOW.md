# Revit RVT 工作流程（Windows 本機）

1. 下載 `datacenter_model.obj` 與 `datacenter_model.mtl`。
2. Codespaces 無法直接產生真正 `.rvt`（需要 Revit 環境與 API）。
3. 在 Revit 中可使用：
   - 插入 → 匯入 CAD → 匯入 `datacenter_floorplan.dxf` 作為 2D 底圖。
   - 或使用 `datacenter_model.obj` 作為 3D reference model。
4. 若要建立真正 RVT，需在有 Revit 的 Windows 環境使用 Revit API Add-in 或 Dynamo 建模。
