from __future__ import annotations

from pathlib import Path

SCAN_EXTS = {".ply", ".obj", ".glb", ".npz"}
SEARCH_DIRS = [
    Path("core_engine/output"),
    Path("core_engine/outputs"),
    Path("core_engine/demo_render"),
    Path("core_engine/demo_render/output"),
    Path("core_engine/example"),
    Path("datacenter_modeler/input"),
]


def find_latest_scan_output(repo_root: str | Path = ".") -> dict | None:
    root = Path(repo_root)
    candidates = []
    for rel in SEARCH_DIRS:
        p = root / rel
        if not p.exists():
            continue
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in SCAN_EXTS:
                candidates.append(f)
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return {
        "path": str(latest),
        "format": latest.suffix.lower(),
        "modified_time": latest.stat().st_mtime,
    }
