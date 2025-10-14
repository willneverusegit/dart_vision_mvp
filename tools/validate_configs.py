#!/usr/bin/env python3
"""
Standalone YAML config validator for Smart-DARTS.
- Read-only: validates YAML files against JSON Schemas.
- Does NOT modify any file (no writes).
Usage:
  python tools/validate_configs.py --config-dir config
  python tools/validate_configs.py config/app.yaml config/calib.yaml
Requires:
  PyYAML (already used) and optionally 'jsonschema' (pip install jsonschema).
Exit codes:
  0 = all valid; 1 = at least one invalid or runtime error.
"""
from __future__ import annotations
import argparse, sys, json, os, glob
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml  # PyYAML
except Exception as e:
    print(f"[FATAL] PyYAML not available: {e}", file=sys.stderr); sys.exit(1)

try:
    from jsonschema import Draft2020Validator, exceptions as js_exc
    _JS_OK = True
except Exception:
    _JS_OK = False

SCHEMA_MAP = {
    "app.yaml":     "schemas/app.schema.json",
    "calib.yaml":   "schemas/calib.schema.json",
    "overlay.yaml": "schemas/overlay.schema.json",
    "game.yaml":    "schemas/game.schema.json",
}

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def infer_schema_path(yaml_path: Path) -> Path | None:
    name = yaml_path.name
    if name in SCHEMA_MAP:
        return Path(SCHEMA_MAP[name])
    return None  # unknown file -> skip schema validation (but still parse)

def validate_one(yaml_path: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    # Parse YAML first (structure sanity)
    try:
        data = load_yaml(yaml_path)
    except Exception as e:
        return False, [f"YAML parse error: {e}"]

    schema_path = infer_schema_path(yaml_path)
    if schema_path is None:
        return True, [f"No schema registered for {yaml_path.name} (parsed OK)"]

    if not schema_path.exists():
        return False, [f"Schema not found: {schema_path}"]

    if not _JS_OK:
        return False, ["jsonschema not installed. Run: pip install jsonschema"]

    schema = load_json(schema_path)
    validator = Draft2020Validator(schema)
    errs = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if not errs:
        return True, ["Valid"]
    for e in errs:
        loc = "/".join([str(p) for p in e.path]) or "<root>"
        errors.append(f"{loc}: {e.message}")
    return False, errors

def expand_targets(config_dir: Path | None, files: List[str]) -> List[Path]:
    paths: List[Path] = []
    if config_dir:
        for pat in ("*.yaml", "*.yml"):
            paths.extend([Path(p) for p in glob.glob(str(config_dir / pat))])
    for f in files:
        paths.append(Path(f))
    # unique & existing
    uniq = []
    seen = set()
    for p in paths:
        p = p.resolve()
        if p not in seen and p.exists():
            uniq.append(p); seen.add(p)
    return uniq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=Path, help="Directory with YAML configs")
    ap.add_argument("files", nargs="*", help="Explicit YAML file paths")
    args = ap.parse_args()

    targets = expand_targets(args.config_dir, args.files)
    if not targets:
        print("No YAML targets found.", file=sys.stderr)
        return 1

    any_fail = False
    print("=== Smart-DARTS Config Validation ===")
    for yp in targets:
        ok, msgs = validate_one(yp)
        status = "OK " if ok else "FAIL"
        print(f"[{status}] {yp.name}")
        for m in msgs:
            print(f"  - {m}")
        any_fail |= (not ok)
    return 0 if not any_fail else 1

if __name__ == "__main__":
    sys.exit(main())
