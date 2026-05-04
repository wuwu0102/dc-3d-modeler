# DC 3D Modeler

## 1. Current stage
- sample layout → CAD/Revit outputs.
- scan import prototype (.ply/.obj/.glb/.npz/.json).
- calibration pipeline prototype (door-height / door-width / rack-width).
- core_engine (lingbot-map) integration is currently **adapter mode**.

## 2. Quick demo
```bash
pip install -r requirements.txt
python -m datacenter_modeler.cli demo-all
```

## 3. Core engine adapter workflow
```bash
python -m datacenter_modeler.cli run-reconstruction
```

Adapter behavior:
- Detects whether `core_engine/` exists.
- Tries to execute an existing core_engine demo/reconstruction script.
- If integration execution fails, writes `datacenter_modeler/output/reconstruction_status.json`.
- If execution succeeds, finds the newest `.obj` / `.ply` / `.npz` under `core_engine/` and copies it to:
  `datacenter_modeler/input/reconstruction.obj`.
- If no scan output is found, reports a hard error (no fake fallback geometry).

## 4. Scan to CAD/Revit pipeline
```bash
python -m datacenter_modeler.cli scan-to-cad \
  --scan path/to/reconstruction.ply \
  --reference-type door-height \
  --measured 1.72 \
  --actual 2.10
```

## 5. Video to CAD pipeline
```bash
python -m datacenter_modeler.cli video-to-cad \
  --reference-type door-height \
  --measured 1.72 \
  --actual 2.10
```

Pipeline order:
1) `run-reconstruction`
2) `scan-to-cad`
3) output package zip

## 6. Useful commands
```bash
python -m datacenter_modeler.cli find-scan-output
python -m datacenter_modeler.cli add-equipment --layout datacenter_modeler/output/datacenter_layout_scaled.json --type rack --name R01 --x 0 --y 0 --z 0 --width 0.6 --depth 1.2 --height 2.2 --power-kw 10
python -m datacenter_modeler.cli export-all --layout datacenter_modeler/output/datacenter_layout_scaled.json
```
