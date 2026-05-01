# create_rvt_from_layout（Dynamo/Revit 草案）

此草案用於 Windows 本機 Revit/Dynamo，目標是把 `datacenter_layout_scaled.json` 轉為 Revit 內的 3D 盒狀設備模型。

流程：
1. 讀取 `datacenter_layout_scaled.json`。
2. 逐筆設備建立 DirectShape box（使用 width/depth/height + x/y/z）。
3. 寫入參數：
   - `name`
   - `type`
   - `power_kw`
   - `cooling_kw`
   - `note`

> 注意：這不是可直接在 Codespaces 執行的腳本，而是 Revit/Dynamo 實作草案。
