# Module: `src\calibration\calib_io.py`
Hash: `d467e9dd7f33` · LOC: 1 · Main guard: false

## Imports
- `numpy`\n- `yaml`

## From-Imports
- `from pathlib import Path`\n- `from typing import Any, Dict`

## Classes
—

## Functions
- `_deep_to_serializable()` (L6)\n- `save_calibration_yaml()` (L19)\n- `load_calibration_yaml()` (L26)

## Intra-module calls (heuristic)
FileNotFoundError, Path, _deep_to_serializable, exists, float, int, isinstance, items, mkdir, open, safe_dump, safe_load, tolist

## Code
```python
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict

def _deep_to_serializable(x: Any):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64, np.integer)):
        return int(x)
    if isinstance(x, dict):
        return {k: _deep_to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_deep_to_serializable(v) for v in x]
    return x

def save_calibration_yaml(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = _deep_to_serializable(data)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, sort_keys=False, allow_unicode=True)

def load_calibration_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

```
