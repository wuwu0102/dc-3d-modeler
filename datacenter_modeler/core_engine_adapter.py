from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

SCAN_EXTS = {".ply", ".obj", ".npz"}
DEMO_SCRIPT_CANDIDATES = [
    Path("demo.py"),
    Path("run_demo.py"),
    Path("scripts/demo.py"),
    Path("scripts/reconstruction.py"),
    Path("reconstruction.py"),
]


class CoreEngineAdapterError(RuntimeError):
    """Raised when core_engine reconstruction cannot be completed."""


def _status_payload(ok: bool, message: str, executed: str | None = None, output: str | None = None) -> dict:
    payload = {"ok": ok, "message": message}
    if executed:
        payload["executed_script"] = executed
    if output:
        payload["output"] = output
    return payload


def write_reconstruction_status(status_path: Path, payload: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_latest_scan_output(search_root: str | Path) -> Path | None:
    root = Path(search_root)
    candidates: list[Path] = []
    for ext in SCAN_EXTS:
        candidates.extend(root.rglob(f"*{ext}"))
    files = [p for p in candidates if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _run_core_engine_demo(core_engine_dir: Path) -> tuple[bool, str, str | None]:
    for script in DEMO_SCRIPT_CANDIDATES:
        script_path = core_engine_dir / script
        if not script_path.exists() or not script_path.is_file():
            continue
        cmd = ["python", str(script_path)]
        result = subprocess.run(cmd, cwd=core_engine_dir.parent, capture_output=True, text=True, check=False)
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        if result.returncode == 0:
            return True, str(script_path.relative_to(core_engine_dir.parent)), output.strip()
        return False, str(script_path.relative_to(core_engine_dir.parent)), output.strip()
    return False, "", None


def run_reconstruction(repo_root: str | Path = ".") -> Path:
    root = Path(repo_root)
    core_engine_dir = root / "core_engine"
    status_path = root / "datacenter_modeler" / "output" / "reconstruction_status.json"
    dest_scan = root / "datacenter_modeler" / "input" / "reconstruction.obj"

    if not core_engine_dir.exists():
        payload = _status_payload(False, "core_engine not found; adapter integration is pending.")
        write_reconstruction_status(status_path, payload)
        raise CoreEngineAdapterError("core_engine folder not found.")

    ok, executed_script, output = _run_core_engine_demo(core_engine_dir)
    if not ok:
        message = "core_engine demo/reconstruction script was not found or failed; adapter integration is pending."
        payload = _status_payload(False, message, executed_script or None, output)
        write_reconstruction_status(status_path, payload)
        if executed_script:
            raise CoreEngineAdapterError(f"core_engine script failed: {executed_script}")
        raise CoreEngineAdapterError("No runnable core_engine demo/reconstruction script found.")

    latest = find_latest_scan_output(core_engine_dir)
    if latest is None:
        raise CoreEngineAdapterError("core_engine ran, but no .obj/.ply/.npz scan output was found.")

    dest_scan.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, dest_scan)
    payload = _status_payload(True, "core_engine reconstruction succeeded.", executed_script, str(latest.relative_to(root)))
    write_reconstruction_status(status_path, payload)
    return dest_scan
