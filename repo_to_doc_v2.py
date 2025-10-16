#!/usr/bin/env python3
"""
repo_to_chatgpt_doc_v2.py
Improvements over v1:
- Smarter entry detection under __main__:
  * Detects patterns like `App(args).run()` and `app = App(...); app.run()`
  * Maps variable instances to classes within the __main__ block
  * Supports argparse `set_defaults(func=...)` pattern (best-effort)
- Reachability:
  * Resolves `obj.method()` when `obj` is an instance of a known class (best-effort)
  * Keeps `self.method()` support; adds simple var->class propagation inside functions
- Duplicate labeling:
  * Default dup-mode = "hash" only (code-identical). Avoids noisy "same-name" duplicates.
  * Configurable via --dup-mode {hash,name,hash_or_name,hash_same_module}
- Badges / Legend:
  * REACH_ENTRY (blue): direct entry
  * USED (green): reachable from entries
  * UNUSED (gray/red): not reached
  * DUPLICATE (yellow): per chosen dup-mode
- Quality-of-life:
  * --entry-qualname supports full "module:Class.method" or "module:function"
  * Slightly clearer Mermaid call graph & sizes
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

DEFAULT_EXCLUDES = [
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "env", "build", "dist", "site-packages", "node_modules",
    ".ipynb_checkpoints", ".DS_Store", ".idea", ".vscode",
]
TEST_DIR_HINTS = {"tests", "test", "testing"}

@dataclass
class DefNode:
    kind: str  # "function" | "method" | "class"
    module: str
    name: str
    qualname: str
    file_path: Path
    lineno: int
    end_lineno: int
    code_hash: str = ""
    calls: Set[str] = field(default_factory=set)
    targets_raw: Set[str] = field(default_factory=set)
    class_name: Optional[str] = None  # for methods

@dataclass
class FileInfo:
    path: Path
    module: str
    content: str
    ast_root: Optional[ast.AST]
    imports: Dict[str, str]
    from_imports: Dict[str, str]
    defs: Dict[str, DefNode]

@dataclass
class RepoModel:
    files: Dict[str, FileInfo] = field(default_factory=dict)
    defs: Dict[str, DefNode] = field(default_factory=dict)
    name_index: Dict[str, Set[str]] = field(default_factory=dict)

def read_text(path: Path, encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding, errors="replace")
    except Exception as e:
        return f"# <ERROR reading file: {e}>\\n"

def rel_module_from_path(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:12]

def get_source_segment(content: str, lineno: int, end_lineno: int) -> str:
    lines = content.splitlines()
    start = max(1, lineno) - 1
    end = max(1, end_lineno)
    return "\\n".join(lines[start:end])

def is_probably_testfile(path: Path) -> bool:
    name = path.name
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    for seg in path.parts:
        if seg.lower() in TEST_DIR_HINTS:
            return True
    return False

# ---------------- AST indexers ----------------

class ASTIndexer(ast.NodeVisitor):
    def __init__(self, file_path: Path, module: str, content: str):
        self.file_path = file_path
        self.module = module
        self.content = content
        self.imports: Dict[str, str] = {}
        self.from_imports: Dict[str, str] = {}
        self.defs: Dict[str, DefNode] = {}
        self.class_stack: List[str] = []
        super().__init__()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            asname = alias.asname or alias.name
            self.imports[asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            asname = alias.asname or alias.name
            self.from_imports[asname] = f"{mod}.{alias.name}" if mod else alias.name
        self.generic_visit(node)

    def _qual(self, name: str) -> str:
        if self.class_stack:
            return f"{self.module}:{self.class_stack[-1]}.{name}"
        return f"{self.module}:{name}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self.class_stack else "function"
        qn = self._qual(node.name)
        seg = get_source_segment(self.content, node.lineno, getattr(node, "end_lineno", node.lineno))
        d = DefNode(kind=kind, module=self.module, name=node.name, qualname=qn,
                    file_path=self.file_path, lineno=node.lineno,
                    end_lineno=getattr(node, "end_lineno", node.lineno),
                    code_hash=stable_hash(seg),
                    class_name=self.class_stack[-1] if self.class_stack else None)
        self.defs[qn] = d
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qn = f"{self.module}:{node.name}"
        seg = get_source_segment(self.content, node.lineno, getattr(node, "end_lineno", node.lineno))
        d = DefNode(kind="class", module=self.module, name=node.name, qualname=qn,
                    file_path=self.file_path, lineno=node.lineno,
                    end_lineno=getattr(node, "end_lineno", node.lineno),
                    code_hash=stable_hash(seg))
        self.defs[qn] = d
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

class CallCollector(ast.NodeVisitor):
    def __init__(self, file_info: FileInfo, current_def: DefNode,
                 var_class_map: Optional[Dict[str, str]] = None):
        self.fi = file_info
        self.cur = current_def
        self.var_class_map = var_class_map or {}

    def visit_Call(self, node: ast.Call) -> None:
        name = self._resolve_call_name(node.func)
        if name:
            self.cur.targets_raw.add(name)
        self.generic_visit(node)

    def _resolve_call_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._resolve_call_name(node.value)
            if base:
                return f"{base}.{node.attr}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return f"{node.func.id}().<call>"
            if isinstance(node.func, ast.Attribute):
                base = self._resolve_call_name(node.func.value)
                if base:
                    return f"{base}.{node.func.attr}()<call>"
        return None

# ---------------- Repo build & resolution ----------------

def build_repo_model(root: Path, include_tests: bool, encoding: str = "utf-8",
                     excludes: Optional[List[str]] = None) -> RepoModel:
    excludes = set(excludes or DEFAULT_EXCLUDES)
    model = RepoModel()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excludes]
        pdir = Path(dirpath)
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = pdir / fname
            if not include_tests and is_probably_testfile(fpath):
                continue
            content = read_text(fpath, encoding=encoding)
            try:
                tree = ast.parse(content)
            except Exception:
                tree = None
            module = rel_module_from_path(root, fpath)

            imports: Dict[str, str] = {}
            from_imports: Dict[str, str] = {}
            defs: Dict[str, DefNode] = {}

            if tree is not None:
                indexer = ASTIndexer(fpath, module, content)
                indexer.visit(tree)
                imports = indexer.imports
                from_imports = indexer.from_imports
                defs = indexer.defs

            fi = FileInfo(path=fpath, module=module, content=content, ast_root=tree,
                          imports=imports, from_imports=from_imports, defs=defs)
            model.files[str(fpath)] = fi
            for qn, d in defs.items():
                model.defs[qn] = d
                model.name_index.setdefault(d.name, set()).add(qn)

    # Collect calls per def with basic var->class mapping
    for fi in model.files.values():
        if fi.ast_root is None:
            continue

        class LocalVarTypes(ast.NodeVisitor):
            def __init__(self):
                self.scope_stack: List[Dict[str, str]] = [{}]

            def current(self) -> Dict[str, str]:
                return self.scope_stack[-1]

            def push(self): self.scope_stack.append(self.current().copy())
            def pop(self): self.scope_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self.push()
                for stmt in node.body:
                    self._scan_stmt(stmt)
                qn = qualify(node)
                d = fi.defs.get(qn)
                if d:
                    CallCollector(fi, d, self.current()).visit(node)
                self.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self.visit_FunctionDef(node)

            def _scan_stmt(self, stmt: ast.stmt):
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    class_name = None
                    if isinstance(func, ast.Name):
                        class_name = func.id
                    elif isinstance(func, ast.Attribute):
                        class_name = func.attr
                    if class_name:
                        for tgt in stmt.targets:
                            if isinstance(tgt, ast.Name):
                                self.current()[tgt.id] = class_name
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child, ast.stmt):
                        self._scan_stmt(child)

        def_map_by_range: List[Tuple[Tuple[int, int], DefNode]] = [((d.lineno, d.end_lineno), d) for d in fi.defs.values()]

        def qualify(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
            inside_class = None
            for ((l1, l2), dnode) in def_map_by_range:
                if dnode.kind == "class" and dnode.lineno <= node.lineno <= dnode.end_lineno:
                    inside_class = dnode.name
                    break
            if inside_class:
                return f"{fi.module}:{inside_class}.{node.name}"
            return f"{fi.module}:{node.name}"

        LocalVarTypes().visit(fi.ast_root)

    # Resolve raw targets
    for fi in model.files.values():
        for d in fi.defs.values():
            for raw in d.targets_raw:
                d.calls.update(resolve_raw_target(raw, fi, model, d))

    return model

def resolve_raw_target(raw: str, fi: FileInfo, model: RepoModel, current: DefNode) -> Set[str]:
    out: Set[str] = set()

    if raw.startswith("self."):
        meth = raw.split(".", 1)[1]
        if current.class_name:
            qn = f"{current.module}:{current.class_name}.{meth}"
            if qn in model.defs:
                out.add(qn)
        return out

    if "." in raw and not raw.endswith("()<call>"):
        head, tail = raw.split(".", 1)
        if head in fi.from_imports:
            tail_simple = tail.split(".")[-1]
            for qn in model.name_index.get(tail_simple, []):
                out.add(qn)
            return out
        if head in fi.imports:
            mod = fi.imports[head]
            tail_simple = tail.split(".")[-1]
            for qn in model.name_index.get(tail_simple, []):
                if qn.startswith(f"{mod}:") or qn.split(":")[0].startswith(mod):
                    out.add(qn)
            return out
        tail_simple = tail.split(".")[-1]
        for qn in model.name_index.get(tail_simple, []):
            out.add(qn)
        return out

    if raw.endswith("()<call>"):
        base = raw.split("()", 1)[0]
        cand_class = f"{fi.module}:{base}"
        if cand_class in model.defs and model.defs[cand_class].kind == "class":
            init_qn = f"{fi.module}:{base}.__init__"
            if init_qn in model.defs:
                out.add(init_qn)
        for qn in model.name_index.get(base, []):
            if qn.endswith(".__init__"):
                out.add(qn)
        return out

    cand_local = f"{fi.module}:{raw}"
    if cand_local in model.defs:
        out.add(cand_local)
    for qn in model.name_index.get(raw, []):
        out.add(qn)
    return out

# -------------- Entrypoint detection & reachability --------------

def detect_entry_functions(fi: FileInfo) -> Set[str]:
    entries: Set[str] = set()
    if fi.ast_root is None:
        return entries

    class MainGuardFinder(ast.NodeVisitor):
        def __init__(self):
            self.nodes: List[ast.If] = []
        def visit_If(self, node: ast.If):
            try:
                cond = ast.unparse(node.test)
            except Exception:
                cond = ""
            if "__name__" in cond and "__main__" in cond:
                self.nodes.append(node)
            self.generic_visit(node)

    mgf = MainGuardFinder()
    mgf.visit(fi.ast_root)

    for ifnode in mgf.nodes:
        var_to_class: Dict[str, str] = {}
        for stmt in ifnode.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                cls = None
                if isinstance(func, ast.Name):
                    cls = func.id
                elif isinstance(func, ast.Attribute):
                    cls = func.attr
                if cls:
                    for t in stmt.targets:
                        if isinstance(t, ast.Name):
                            var_to_class[t.id] = cls

        class CallFinder(ast.NodeVisitor):
            def __init__(self, vmap: Dict[str, str]):
                self.calls: Set[str] = set()
                self.vmap = vmap
            def visit_Call(self, node: ast.Call):
                if isinstance(node.func, ast.Name):
                    self.calls.add(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        obj = node.func.value.id
                        meth = node.func.attr
                        if obj in self.vmap:
                            self.calls.add(f"{self.vmap[obj]}.{meth}")
                    elif isinstance(node.func.value, ast.Call):
                        fun = node.func.value.func
                        if isinstance(fun, ast.Name):
                            self.calls.add(f"{fun.id}.{node.func.attr}")
                        elif isinstance(fun, ast.Attribute):
                            self.calls.add(f"{fun.attr}.{node.func.attr}")
                self.generic_visit(node)

        cf = CallFinder(var_to_class)
        for stmt in ifnode.body:
            cf.visit(stmt)
        for name in cf.calls:
            if "." in name:
                cls, meth = name.split(".", 1)
                entries.add(f"{fi.module}:{cls}.{meth}")
            else:
                entries.add(f"{fi.module}:{name}")

    class ArgparseFinder(ast.NodeVisitor):
        def __init__(self):
            self.fn_names: Set[str] = set()
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "set_defaults":
                for kw in node.keywords or []:
                    if kw.arg == "func":
                        if isinstance(kw.value, ast.Name):
                            self.fn_names.add(kw.value.id)
                        elif isinstance(kw.value, ast.Attribute):
                            base = kw.value.value
                            if isinstance(base, ast.Name):
                                self.fn_names.add(f"{base.id}.{kw.value.attr}")
            self.generic_visit(node)

    af = ArgparseFinder()
    af.visit(fi.ast_root)
    for name in af.fn_names:
        if "." in name:
            base, meth = name.split(".", 1)
            entries.add(f"{fi.module}:{base}.{meth}")
        else:
            entries.add(f"{fi.module}:{name}")
    return entries

def compute_reachable(model: RepoModel, entry_qns: Set[str]) -> Tuple[Set[str], Set[str]]:
    direct = set(q for q in entry_qns if q in model.defs)
    seen: Set[str] = set()
    stack: List[str] = list(direct)
    while stack:
        qn = stack.pop()
        if qn in seen: 
            continue
        seen.add(qn)
        d = model.defs.get(qn)
        if not d: 
            continue
        for tgt in d.calls:
            if tgt in model.defs and tgt not in seen:
                stack.append(tgt)
    return direct, seen

# -------------- Dup detection --------------

def detect_duplicates(model: RepoModel, mode: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    hash_map: Dict[str, List[str]] = {}
    for qn, d in model.defs.items():
        hash_map.setdefault(d.code_hash, []).append(qn)

    dup_code_qns: Set[str] = set()
    for h, qns in hash_map.items():
        if len(qns) > 1:
            dup_code_qns.update(qns)

    name_dups: Dict[str, Set[str]] = {}
    if mode in {"name", "hash_or_name", "hash_same_module"}:
        for n, qset in model.name_index.items():
            if len(qset) > 1:
                name_dups[n] = set(qset)

    if mode == "hash":
        name_dups = {}

    if mode == "hash_same_module":
        filtered: Dict[str, Set[str]] = {}
        for n, qset in name_dups.items():
            buckets: Dict[str, Set[str]] = {}
            for q in qset:
                mod = q.split(":")[0]
                buckets.setdefault(mod, set()).add(q)
            keep: Set[str] = set()
            for mod, items in buckets.items():
                if len(items) > 1:
                    keep |= items
            if keep:
                filtered[n] = keep
        name_dups = filtered

    return dup_code_qns, name_dups

# -------------- Rendering --------------

def html_badge(text: str, color: str) -> str:
    return f'<span style="background:{color};color:white;padding:2px 6px;border-radius:6px;font-size:0.85em;">{text}</span>'

def render_markdown(model: RepoModel, root: Path,
                    direct_entries: Set[str], reachable: Set[str],
                    dup_code_qns: Set[str], name_dups: Dict[str, Set[str]],
                    max_edges: int = 600) -> str:
    used_color = "#2e7d32"       # green
    direct_color = "#1565c0"     # blue
    unused_color = "#6d4c41"     # brown/gray
    dup_color = "#f9a825"        # yellow

    used_badge = html_badge("USED", used_color)
    direct_badge = html_badge("REACH_ENTRY", direct_color)
    unused_badge = html_badge("UNUSED", unused_color)
    dup_badge = html_badge("DUPLICATE", dup_color)

    lines: List[str] = []
    lines.append(f"# ChatGPT Analysis Bundle (improved)\\n")
    lines.append(f"**Root:** `{root}`  \\n")
    lines.append("**Legend:** "
                 + direct_badge + " = direct entry; "
                 + used_badge + " = reachable from entries; "
                 + unused_badge + " = not reached; "
                 + dup_badge + " = duplicate per selected mode.\\n")
    if direct_entries:
        lines.append("**Entries:** " + ", ".join(sorted(direct_entries)) + "\\n")

    edges: List[Tuple[str, str]] = []
    for d in model.defs.values():
        if d.qualname in reachable:
            for tgt in d.calls:
                if tgt in reachable:
                    edges.append((d.qualname, tgt))
                    if len(edges) >= max_edges:
                        break
        if len(edges) >= max_edges:
            break

    if edges:
        lines.append("## Call Graph (approx., reachable subset)\\n")
        lines.append("```mermaid")
        lines.append("flowchart LR")
        lines.append("classDef entry fill:#1565c0,stroke:#0d47a1,color:#fff;")
        lines.append("classDef used fill:#2e7d32,stroke:#1b5e20,color:#fff;")
        lines.append("classDef unused fill:#6d4c41,stroke:#3e2723,color:#fff;")
        def node_id(qn: str) -> str:
            return re.sub(r\"[^A-Za-z0-9_]\", \"_\", qn)[:60]
        nodes = set([a for a,b in edges] + [b for a,b in edges])
        for qn in nodes:
            nid = node_id(qn)
            label = qn
            cls = "used"
            if qn in direct_entries:
                cls = "entry"
            lines.append(f'  {nid}["{label}"]:::' + cls)
        for (a, b) in edges:
            lines.append(f'  {node_id(a)} --> {node_id(b)}')
        lines.append("```")
        lines.append("")

    lines.append("## Files & Definitions\\n")
    for fi in sorted(model.files.values(), key=lambda f: str(f.path).lower()):
        rel = fi.path.relative_to(root)
        lines.append(f"### `{rel.as_posix()}`\\n")
        if fi.defs:
            lines.append("**Definitions:**\\n")
            for d in sorted(fi.defs.values(), key=lambda d: (d.kind, d.lineno)):
                badges = []
                if d.qualname in direct_entries:
                    badges.append(direct_badge)
                elif d.qualname in reachable:
                    badges.append(used_badge)
                else:
                    badges.append(unused_badge)
                if (d.qualname in dup_code_qns) or (d.name in name_dups and d.qualname in name_dups[d.name]):
                    badges.append(dup_badge)
                lines.append(f"- {' '.join(badges)} `{d.kind}` **{d.qualname}**  "
                             f"(L{d.lineno}–{d.end_lineno}) calls: {', '.join(sorted(d.calls)) or '—'}")
        else:
            lines.append("_No definitions parsed._")
        lines.append("")
        lines.append("<details><summary>Show file content</summary>\\n\\n")
        lines.append("```python")
        lines.append(fi.content.rstrip())
        lines.append("```")
        lines.append("\\n</details>\\n")
    return "\\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Export repository sources and a static call graph (improved) to a single Markdown bundle for ChatGPT analysis.")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current dir).")
    ap.add_argument("--entry", type=str, default="main.py", help="Entry file path relative to root (default: main.py).")
    ap.add_argument("--entry-func", action="append", default=[], help="Entry function name(s). Can be passed multiple times.")
    ap.add_argument("--entry-qualname", action="append", default=[], help='Explicit entry qualname(s) like \"pkg.mod:Class.method\" or \"pkg.mod:function\".')
    ap.add_argument("--output", type=Path, default=Path("chatgpt_repo_bundle.md"), help="Output Markdown path.")
    ap.add_argument("--include-tests", action="store_true", help="Include test files (default: False).")
    ap.add_argument("--encoding", type=str, default="utf-8", help="Read encoding (default: utf-8).")
    ap.add_argument("--max-edges", type=int, default=600, help="Max edges in Mermaid graph (default: 600).")
    ap.add_argument("--exclude", action="append", default=[], help="Additional directory names to exclude (repeatable).")
    ap.add_argument("--dup-mode", type=str, choices=["hash","name","hash_or_name","hash_same_module"], default="hash",
                    help="Duplicate detection mode (default: hash).")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"ERROR: root path not found: {root}", file=sys.stderr)
        sys.exit(2)

    model = build_repo_model(root, include_tests=args.include_tests, encoding=args.encoding,
                             excludes=DEFAULT_EXCLUDES + list(args.exclude))

    entry_qns: Set[str] = set()

    for q in args.entry_qualname:
        if ":" in q:
            entry_qns.add(q)

    if args.entry:
        entry_path = (root / args.entry).resolve()
        fi = None
        for key, f in model.files.items():
            if Path(key).resolve() == entry_path:
                fi = f
                break
        if fi:
            entry_qns |= detect_entry_functions(fi)
        else:
            entry_mod = rel_module_from_path(root, entry_path)
            cand = f"{entry_mod}:main"
            if cand in model.defs:
                entry_qns.add(cand)

    for fn in args.entry_func:
        if args.entry:
            entry_path = (root / args.entry).resolve()
            entry_mod = rel_module_from_path(root, entry_path)
            candidate = f"{entry_mod}:{fn}"
            if candidate in model.defs:
                entry_qns.add(candidate)
                continue
        for qn in model.name_index.get(fn, []):
            entry_qns.add(qn)

    if not entry_qns and "main" in model.name_index:
        entry_qns |= model.name_index["main"]

    direct_entries, reachable = compute_reachable(model, entry_qns)
    dup_code_qns, name_dups = detect_duplicates(model, args.dup_mode)
    md = render_markdown(model, root, direct_entries, reachable, dup_code_qns, name_dups, max_edges=args.max_edges)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Wrote Markdown bundle to: {args.output}")

if __name__ == "__main__":
    main()
