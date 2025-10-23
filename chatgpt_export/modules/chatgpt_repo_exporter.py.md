# Module: `chatgpt_repo_exporter.py`
Hash: `3a3e1c3d2411` · LOC: 10 · Main guard: true

## Imports
- `argparse`\n- `ast`\n- `hashlib`\n- `json`\n- `os`

## From-Imports
- `from __future__ import annotations`\n- `from pathlib import Path`\n- `from typing import Dict, List, Optional, Tuple, Set`

## Classes
- `ModuleInfo` (L42)

## Functions
- `sha256_text()` (L23)\n- `should_exclude()` (L26)\n- `read_text()` (L30)\n- `relpath()` (L36)\n- `__init__()` (L43)\n- `_parse()` (L57)\n- `walk_python_files()` (L95)\n- `ensure_dir()` (L109)\n- `write_text()` (L112)\n- `make_module_markdown()` (L116)\n- `build_import_graph()` (L155)\n- `mermaid_from_import_graph()` (L175)\n- `nid()` (L177)\n- `write_index()` (L186)\n- `write_manifest_and_flow()` (L202)\n- `write_modules()` (L225)\n- `run()` (L230)\n- `main()` (L237)

## Intra-module calls (heuristic)
ArgumentParser, ModuleInfo, Path, _parse, add, add_argument, any, append, as_posix, build_import_graph, count, cwd, dumps, encode, endswith, ensure_dir, get_docstring, hexdigest, isinstance, items, join, len, lower, main, make_module_markdown, mermaid_from_import_graph, mkdir, nid, parse, parse_args, print, read_text, relative_to, relpath, replace, resolve, run, set, sha256, sha256_text, should_exclude, sorted, splitlines, startswith, str, sum, walk, walk_python_files, with_suffix, write_index, write_manifest_and_flow, write_modules, write_text

## Code
```python
#!/usr/bin/env python3
"""
ChatGPT Repo Exporter
Traverse a repository top-down, index Python modules, extract AST metadata,
and emit ChatGPT-friendly docs (Markdown) plus a simple program-flow JSON.

Language: Code & comments in English (per user preference).
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "build", "dist", ".venv", "venv", ".ruff_cache"
}

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]

def should_exclude(path: Path) -> bool:
    parts = set(path.parts)
    return any(d in EXCLUDE_DIRS for d in parts)

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(errors="ignore")

def relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)

class ModuleInfo:
    def __init__(self, path: Path, root: Path):
        self.path = path
        self.rel = relpath(path, root)
        self.code = read_text(path)
        self.hash = sha256_text(self.code)
        self.loc = self.code.count("\\n") + 1
        self.imports: List[str] = []
        self.from_imports: List[Tuple[str, List[str]]] = []
        self.classes: List[Tuple[str, int, Optional[str]]] = []
        self.functions: List[Tuple[str, int, Optional[str]]] = []
        self.has_main_guard: bool = False
        self.calls_in_module: Set[str] = set()
        self._parse()

    def _parse(self) -> None:
        try:
            tree = ast.parse(self.code, filename=self.rel)
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = [a.name for a in node.names]
                self.from_imports.append((mod, names))
            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                self.classes.append((node.name, node.lineno, doc))
            elif isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node)
                self.functions.append((node.name, node.lineno, doc))
            elif isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name) and test.left.id == "__name__"
                    and any(isinstance(op, ast.Eq) for op in test.ops)
                    and any(
                        isinstance(c, ast.Constant) and c.value == "__main__"
                        for c in test.comparators
                    )
                ):
                    self.has_main_guard = True
            elif isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    self.calls_in_module.add(f.id)
                elif isinstance(f, ast.Attribute):
                    self.calls_in_module.add(f.attr)

def walk_python_files(root: Path) -> List[Path]:
    py_files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        dpath = Path(dirpath)
        if should_exclude(dpath):
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                fpath = dpath / fn
                if not should_exclude(fpath):
                    py_files.append(fpath)
    return sorted(py_files)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")

def make_module_markdown(mi: ModuleInfo) -> str:
    imports = "\\n".join(f"- `{name}`" for name in sorted(set(mi.imports)))
    from_imps = "\\n".join(
        f"- `from {mod} import {', '.join(names)}`" for mod, names in mi.from_imports
    )
    classes = "\\n".join(
        f"- `{name}` (L{lineno})" + (f": {doc.splitlines()[0][:120]}" if doc else "")
        for name, lineno, doc in sorted(mi.classes, key=lambda t: t[1])
    )
    funcs = "\\n".join(
        f"- `{name}()` (L{lineno})" + (f": {doc.splitlines()[0][:120]}" if doc else "")
        for name, lineno, doc in sorted(mi.functions, key=lambda t: t[1])
    )
    calls = ", ".join(sorted(mi.calls_in_module)) if mi.calls_in_module else "—"

    return f"""# Module: `{mi.rel}`
Hash: `{mi.hash}` · LOC: {mi.loc} · Main guard: {str(mi.has_main_guard).lower()}

## Imports
{imports or "—"}

## From-Imports
{from_imps or "—"}

## Classes
{classes or "—"}

## Functions
{funcs or "—"}

## Intra-module calls (heuristic)
{calls}

## Code
```python
{mi.code}
```
"""

def build_import_graph(mods: List[ModuleInfo], root: Path) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {}
    mod_names = {m.rel for m in mods}
    for m in mods:
        deps: Set[str] = set()
        for mod, _names in m.from_imports:
            if not mod:
                continue
            for candidate in mod_names:
                modish = Path(candidate).with_suffix("").as_posix().replace("/", ".")
                if modish == mod or modish.startswith(mod + "."):
                    deps.add(candidate)
        for imp in m.imports:
            for candidate in mod_names:
                modish = Path(candidate).with_suffix("").as_posix().replace("/", ".")
                if modish == imp or modish.startswith(imp + "."):
                    deps.add(candidate)
        graph[m.rel] = sorted(deps)
    return graph

def mermaid_from_import_graph(graph: Dict[str, List[str]]) -> str:
    lines = ["flowchart LR"]
    def nid(p: str) -> str:
        return p.replace("/", "_").replace(".", "_").replace("-", "_")
    for src, deps in graph.items():
        if not deps:
            lines.append(f'    {nid(src)}["{src}"]')
        for dst in deps:
            lines.append(f'    {nid(src)}["{src}"] --> {nid(dst)}["{dst}"]')
    return "\\n".join(lines)

def write_index(mods: List[ModuleInfo], out_dir: Path) -> None:
    total_loc = sum(m.loc for m in mods)
    entry_points = [m for m in mods if m.has_main_guard]
    content = [
        "# ChatGPT Repository Export — INDEX",
        "",
        f"- Modules: **{len(mods)}** · Total LOC: **{total_loc}**",
        f"- Entry points (`if __name__ == '__main__'`): {len(entry_points)}",
        "",
        "## Contents",
    ]
    for m in mods:
        content.append(f"- [{m.rel}](modules/{m.rel.replace('/', '__')}.md) · LOC {m.loc} · hash `{m.hash}`")
    content.append("")
    write_text(out_dir / "INDEX.md", "\\n".join(content))

def write_manifest_and_flow(mods: List[ModuleInfo], out_dir: Path) -> None:
    manifest = {
        "modules": [
            {
                "path": m.rel,
                "hash": m.hash,
                "loc": m.loc,
                "imports": sorted(set(m.imports)),
                "from_imports": [{"module": mod, "names": names} for mod, names in m.from_imports],
                "classes": [{"name": n, "lineno": ln} for n, ln, _ in m.classes],
                "functions": [{"name": n, "lineno": ln} for n, ln, _ in m.functions],
                "has_main_guard": m.has_main_guard,
                "calls_in_module": sorted(m.calls_in_module),
            }
            for m in mods
        ]
    }
    write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2))
    graph = build_import_graph(mods, out_dir)
    write_text(out_dir / "program_flow.json", json.dumps(graph, indent=2))
    mermaid = mermaid_from_import_graph(graph)
    write_text(out_dir / "program_flow_mermaid.md", f"```mermaid\\n{mermaid}\\n```")

def write_modules(mods: List[ModuleInfo], out_dir: Path) -> None:
    for m in mods:
        path = out_dir / "modules" / (m.rel.replace("/", "__") + ".md")
        write_text(path, make_module_markdown(m))

def run(root: Path, out: Path) -> None:
    py_files = walk_python_files(root)
    mods = [ModuleInfo(p, root) for p in py_files]
    write_index(mods, out)
    write_manifest_and_flow(mods, out)
    write_modules(mods, out)

def main() -> None:
    ap = argparse.ArgumentParser(description="Export a Python repository for ChatGPT analysis.")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to scan.")
    ap.add_argument("--out", type=Path, default=Path("chatgpt_export"), help="Output directory.")
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    ensure_dir(out)
    run(root, out)
    print(f"[OK] Exported ChatGPT docs to: {out}")

if __name__ == "__main__":
    main()

```
