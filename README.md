# DC 3D Modeler

## 1. Current stage
- sample layout → CAD/Revit outputs.
- scan import prototype (.ply/.obj/.glb/.npz/.json).
- calibration pipeline prototype (door-height / door-width / rack-width).

## 2. Quick demo
```bash
pip install -r requirements.txt
python -m datacenter_modeler.cli demo-all
```

## 3. Scan to CAD/Revit pipeline
```bash
python -m datacenter_modeler.cli scan-to-cad \
  --scan path/to/reconstruction.ply \
  --reference-type door-height \
  --measured 1.72 \
  --actual 2.10
```

## 4. Useful commands
```bash
python -m datacenter_modeler.cli find-scan-output
python -m datacenter_modeler.cli add-equipment --layout datacenter_modeler/output/datacenter_layout_scaled.json --type rack --name R01 --x 0 --y 0 --z 0 --width 0.6 --depth 1.2 --height 2.2 --power-kw 10
python -m datacenter_modeler.cli export-all --layout datacenter_modeler/output/datacenter_layout_scaled.json
```
