#!/usr/bin/env python3
"""
repo_to_chatgpt_doc.py
Top-down repository scanner that builds a ChatGPT-ready Markdown document:
- Walks a codebase from a given root
- Parses all Python files (code + comments) and inlines their content
- Reconstructs an approximate program flow from a main entrypoint
- Builds a simple static call graph (functions & methods) and usage map
- Marks items: USED (green), UNUSED (red), DUPLICATE (yellow)
No external dependencies. Python >= 3.10 recommended.

Usage:
  python repo_to_doc.py --root /path/to/repo --entry main.py --output chatgpt_repo_bundle.md

Tip:
  Use --entry-func to override the auto-detected entry function(s). Multiple allowed.
  Use --include-tests to include tests; by default typical test folders are excluded.
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

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class DefNode:
    """Represents a function or method or class (treated as constructable callable via __init__)."""
    kind: str  # "function" | "method" | "class"
    module: str  # dotted module path (e.g., "src.module.sub")
    name: str  # simple name (e.g., "foo", "Bar", "__init__")
    qualname: str  # module + optional class + name (e.g., "a.b:Class.meth" or "a.b:func")
    file_path: Path
    lineno: int
    end_lineno: int
    code_hash: str = ""
    calls: Set[str] = field(default_factory=set)  # target qualname candidates (unresolved -> best-effort resolution)
    targets_raw: Set[str] = field(default_factory=set)  # raw names seen in AST (for debugging)
    class_name: Optional[str] = None  # if kind in {"method"}

@dataclass
class FileInfo:
    path: Path
    module: str
    content: str
    ast_root: Optional[ast.AST]
    imports: Dict[str, str]  # alias -> full module or module.object
    from_imports: Dict[str, str]  # local name -> full module.object
    defs: Dict[str, DefNode]  # qualname -> DefNode

@dataclass
class RepoModel:
    files: Dict[str, FileInfo] = field(default_factory=dict)  # key: str(path)
    defs: Dict[str, DefNode] = field(default_factory=dict)    # key: qualname
    name_index: Dict[str, Set[str]] = field(default_factory=dict)  # simple name -> set of qualnames (for duplicate-name detection)


# ----------------------------
# Helpers
# ----------------------------

DEFAULT_EXCLUDES = [
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "env", "build", "dist", "site-packages", "node_modules",
    ".ipynb_checkpoints", ".DS_Store", ".idea", ".vscode",
]

TEST_DIR_HINTS = {"tests", "test", "testing"}

def rel_module_from_path(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

def read_text(path: Path, encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding, errors="replace")
    except Exception as e:
        return f"# <ERROR reading file: {e}>\n"

def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:12]

def get_source_segment(content: str, lineno: int, end_lineno: int) -> str:
    lines = content.splitlines()
    # Python AST line numbers are 1-based and inclusive for end_lineno
    start = max(1, lineno) - 1
    end = max(1, end_lineno)
    seg = "\n".join(lines[start:end])
    return seg

def is_probably_testfile(path: Path) -> bool:
    name = path.name
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    for seg in path.parts:
        if seg.lower() in TEST_DIR_HINTS:
            return True
    return False

# ----------------------------
# AST Parsing & Indexing
# ----------------------------

class ASTIndexer(ast.NodeVisitor):
    """Collects definitions, methods, and import aliasing for a single file."""
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
            # import package.sub as ps
            asname = alias.asname or alias.name
            self.imports[asname] = alias.name  # map alias -> full module
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            asname = alias.asname or alias.name
            # from a.b import c as d  => d -> a.b.c
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
        d = DefNode(
            kind=kind,
            module=self.module,
            name=node.name,
            qualname=qn,
            file_path=self.file_path,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            code_hash=stable_hash(seg),
            class_name=self.class_stack[-1] if self.class_stack else None,
        )
        self.defs[qn] = d
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # treat like FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qn = f"{self.module}:{node.name}"
        seg = get_source_segment(self.content, node.lineno, getattr(node, "end_lineno", node.lineno))
        d = DefNode(
            kind="class",
            module=self.module,
            name=node.name,
            qualname=qn,
            file_path=self.file_path,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            code_hash=stable_hash(seg),
        )
        self.defs[qn] = d
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

class CallCollector(ast.NodeVisitor):
    """Collects calls inside a given function/method AST node."""
    def __init__(self, file_info: FileInfo, current_def: DefNode):
        self.fi = file_info
        self.cur = current_def

    def visit_Call(self, node: ast.Call) -> None:
        name = self._resolve_call_name(node.func)
        if name:
            self.cur.targets_raw.add(name)
        self.generic_visit(node)

    def _resolve_call_name(self, node: ast.AST) -> Optional[str]:
        # Simple forms:
        #   Name(id='foo')        -> "foo"
        #   Attribute(Name('m'), 'f') => "m.f"
        #   Attribute(Name('self'),'m') => "<self>.m"
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._resolve_call_name(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None

# ----------------------------
# Resolution & Graph Build
# ----------------------------

def build_repo_model(root: Path,
                     include_tests: bool,
                     encoding: str = "utf-8",
                     excludes: Optional[List[str]] = None) -> RepoModel:
    excludes = set(excludes or DEFAULT_EXCLUDES)
    model = RepoModel()

    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded directories in-place
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

            fi = FileInfo(
                path=fpath,
                module=module,
                content=content,
                ast_root=tree,
                imports=imports,
                from_imports=from_imports,
                defs=defs,
            )
            model.files[str(fpath)] = fi
            for qn, d in defs.items():
                model.defs[qn] = d
                model.name_index.setdefault(d.name, set()).add(qn)

    # Collect calls for each def
    for fi in model.files.values():
        if fi.ast_root is None:
            continue
        # Build a map from lineno->DefNode for faster association:
        def_map_by_range: List[Tuple[Tuple[int, int], DefNode]] = [
            ((d.lineno, d.end_lineno), d) for d in fi.defs.values()
        ]

        class AssocVisitor(ast.NodeVisitor):
            def __init__(self):
                self.stack: List[DefNode] = []
            def visit_FunctionDef(self, node):
                qn = qualify(node)
                d = fi.defs.get(qn)
                if d:
                    self.stack.append(d)
                    CallCollector(fi, d).visit(node)
                    self.stack.pop()
                self.generic_visit(node)
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
            def visit_ClassDef(self, node):
                self.generic_visit(node)

        def qualify(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
            # Recompute qualname like indexer did:
            # Find if inside a class by scanning parents (approx by coordinates)
            inside_class = None
            for ((l1, l2), dnode) in def_map_by_range:
                if dnode.kind == "class" and dnode.lineno <= node.lineno <= dnode.end_lineno:
                    inside_class = dnode.name
                    break
            if inside_class:
                return f"{fi.module}:{inside_class}.{node.name}"
            return f"{fi.module}:{node.name}"

        AssocVisitor().visit(fi.ast_root)

    # Resolve raw targets into qualnames best-effort
    for fi in model.files.values():
        # Reverse maps for quick lookup
        local_defs = {d.name: d for d in fi.defs.values() if d.module == fi.module and d.kind in {"function", "method"}}
        class_defs = {d.name: d for d in fi.defs.values() if d.module == fi.module and d.kind == "class"}

        for d in fi.defs.values():
            # For each raw target, try to resolve to a known qualname
            for raw in d.targets_raw:
                qns = resolve_raw_target(raw, fi, model, d)
                d.calls.update(qns)

    return model

def resolve_raw_target(raw: str, fi: FileInfo, model: RepoModel, current: DefNode) -> Set[str]:
    """
    Best-effort name resolution:
      - "foo" -> try local function/method in same module
      - "alias.func" where alias is import -> map to that module.func
      - "<self>.meth" -> same class method
      - "ClassName" (constructor call) -> class __init__
      - "module.ClassName" -> that class __init__
    """
    out: Set[str] = set()

    # handle <self>.method
    if raw.startswith("self."):
        meth = raw.split(".", 1)[1]
        if current.class_name:
            qn = f"{current.module}:{current.class_name}.{meth}"
            if qn in model.defs:
                out.add(qn)
        return out

    # qualified like alias.attr[.more]
    if "." in raw:
        head, tail = raw.split(".", 1)
        # alias to module?
        if head in fi.imports:
            mod = fi.imports[head]
            candidate = f"{mod.replace('/', '.')}.{tail}"
            # try exact match as function
            qn_func = f"{mod}.{tail}".replace("..", ".")
            # Resolve function simple tail (last attr) against defs by name if module path differs
            tail_simple = tail.split(".")[-1]
            for qn in model.name_index.get(tail_simple, []):
                out.add(qn)
            # Also add module-qualified attempt
            for qn in list(model.defs.keys()):
                if qn.startswith(f"{mod}:{tail_simple}") or qn.startswith(f"{mod}.{tail_simple}"):
                    out.add(qn)
        elif head in fi.from_imports:
            full = fi.from_imports[head]  # e.g. a.b.c
            tail2 = tail.split(".")[-1]
            # direct object import; try match by name
            for qn in model.name_index.get(tail2, []):
                out.add(qn)
        else:
            # Could be module.attr where module is in same package; attempt tail's simple name
            tail_simple = tail.split(".")[-1]
            for qn in model.name_index.get(tail_simple, []):
                out.add(qn)
        return out

    # bare name: local def?
    # function/method
    cand_local = f"{fi.module}:{raw}"
    if cand_local in model.defs:
        out.add(cand_local)
    # class constructor (ClassName -> __init__)
    cand_class = f"{fi.module}:{raw}"
    if cand_class in model.defs and model.defs[cand_class].kind == "class":
        init_qn = f"{fi.module}:{raw}.__init__"
        if init_qn in model.defs:
            out.add(init_qn)
    # by name index anywhere
    for qn in model.name_index.get(raw, []):
        out.add(qn)

    return out

# ----------------------------
# Entrypoint detection & reachability
# ----------------------------

MAIN_GUARDS = re.compile(r"""if\s+__name__\s*==\s*(['"])__main__\1\s*:\s*""")

def detect_entry_functions(fi: FileInfo) -> Set[str]:
    """Heuristic: find functions called under __main__ guard in this file."""
    entries: Set[str] = set()
    if fi.ast_root is None:
        return entries
    # Find "if __name__ == '__main__':" block siblings, collect calls inside
    class MainGuardFinder(ast.NodeVisitor):
        def __init__(self): self.nodes: List[ast.AST] = []
        def visit_If(self, node: ast.If):
            # Try to stringify the 'test' and search for pattern
            src = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
            if "__name__" in src and "__main__" in src:
                self.nodes.append(node)
            self.generic_visit(node)
    mgf = MainGuardFinder()
    mgf.visit(fi.ast_root)
    for ifnode in mgf.nodes:
        # Look for calls in body
        class CallFinder(ast.NodeVisitor):
            def __init__(self): self.calls: Set[str] = set()
            def visit_Call(self, node: ast.Call):
                if isinstance(node.func, ast.Name):
                    self.calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # main() as obj.run() etc. We take the attribute itself.
                    self.calls.add(node.func.attr)
                self.generic_visit(node)
        cf = CallFinder()
        for stmt in ifnode.body:
            cf.visit(stmt)
        entries |= cf.calls
    # Convert to qualnames in this module
    qns = set()
    for name in entries:
        qn = f"{fi.module}:{name}"
        # prefer function; also allow classes with __init__
        qns.add(qn)
        init_qn = f"{fi.module}:{name}.__init__"
        qns.add(init_qn)
    return qns

def compute_reachable(model: RepoModel, entry_qns: Set[str]) -> Set[str]:
    seen: Set[str] = set()
    stack: List[str] = [q for q in entry_qns if q in model.defs]
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
    return seen

# ----------------------------
# Duplicate detection
# ----------------------------

def detect_duplicates(model: RepoModel) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Returns:
      dup_code_qns: set of qualnames whose code hash appears more than once
      name_dups: simple name -> set[qualname] (size>1)
    """
    hash_map: Dict[str, List[str]] = {}
    for qn, d in model.defs.items():
        hash_map.setdefault(d.code_hash, []).append(qn)

    dup_code_qns: Set[str] = set()
    for h, qns in hash_map.items():
        if len(qns) > 1:
            dup_code_qns.update(qns)

    name_dups: Dict[str, Set[str]] = {n:q for n, q in model.name_index.items() if len(q) > 1}
    return dup_code_qns, name_dups

# ----------------------------
# Markdown rendering
# ----------------------------

def html_badge(text: str, color: str) -> str:
    return f'<span style="background:{color};color:white;padding:2px 6px;border-radius:6px;font-size:0.85em;">{text}</span>'

def render_markdown(model: RepoModel,
                    root: Path,
                    reachable: Set[str],
                    dup_code_qns: Set[str],
                    name_dups: Dict[str, Set[str]],
                    entry_qns: Set[str],
                    max_edges: int = 400) -> str:
    used_color = "#2e7d32"      # green
    unused_color = "#c62828"    # red
    dup_color = "#f9a825"       # yellow

    used_badge = html_badge("USED", used_color)
    unused_badge = html_badge("UNUSED", unused_color)
    dup_badge = html_badge("DUPLICATE", dup_color)

    lines: List[str] = []
    lines.append(f"# ChatGPT Analysis Bundle\n")
    lines.append(f"**Root:** `{root}`  \n")
    lines.append("**Legend:** " + used_badge + " = reachable from entry, "
                 + unused_badge + " = not reached, "
                 + dup_badge + " = duplicate (code-hash or name).\n")
    if entry_qns:
        lines.append(f"**Entries:** " + ", ".join(sorted(entry_qns)) + "\n")

    # Mermaid graph (used-only to keep size moderate)
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
        lines.append("## Call Graph (approx., reachable subset)\n")
        lines.append("```mermaid")
        lines.append("flowchart LR")
        def node_id(qn: str) -> str:
            return re.sub(r"[^A-Za-z0-9_]", "_", qn)[:60]
        for (a, b) in edges:
            lines.append(f'  {node_id(a)}["{a}"] --> {node_id(b)}["{b}"]')
        lines.append("```")
        lines.append("")

    # Files
    lines.append("## Files & Definitions\n")
    sorted_files = sorted(model.files.values(), key=lambda fi: str(fi.path).lower())
    for fi in sorted_files:
        rel = fi.path.relative_to(root)
        lines.append(f"### `{rel.as_posix()}`\n")
        # Index of defs
        if fi.defs:
            lines.append("**Definitions:**\n")
            for d in sorted(fi.defs.values(), key=lambda d: (d.kind, d.lineno)):
                badges = []
                if d.qualname in reachable:
                    badges.append(used_badge)
                else:
                    badges.append(unused_badge)
                if d.qualname in dup_code_qns or d.name in name_dups:
                    badges.append(dup_badge)
                badge_str = " ".join(badges)
                lines.append(f"- {badge_str} `{d.kind}` **{d.qualname}**  "
                             f"(L{d.lineno}–{d.end_lineno}) calls: {', '.join(sorted(d.calls)) or '—'}")
        else:
            lines.append("_No definitions parsed._")
        lines.append("")
        # Code block
        lines.append("<details><summary>Show file content</summary>\n\n")
        lines.append("```python")
        lines.append(fi.content.rstrip())
        lines.append("```")
        lines.append("\n</details>\n")

    return "\n".join(lines)

# ----------------------------
# Main CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Export repository Python sources and a static call graph to a single Markdown bundle for ChatGPT analysis.")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to scan (default: current dir).")
    ap.add_argument("--entry", type=str, default="main.py", help="Entry file path relative to root (default: main.py).")
    ap.add_argument("--entry-func", action="append", default=[], help="Entry function or method name(s). Can be passed multiple times.")
    ap.add_argument("--output", type=Path, default=Path("chatgpt_repo_bundle.md"), help="Output Markdown path.")
    ap.add_argument("--include-tests", action="store_true", help="Include tests (default: False).")
    ap.add_argument("--encoding", type=str, default="utf-8", help="File encoding to read (default: utf-8).")
    ap.add_argument("--max-edges", type=int, default=400, help="Max edges in Mermaid graph (default: 400).")
    ap.add_argument("--exclude", action="append", default=[], help="Additional directory names to exclude (can be used multiple times).")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"ERROR: root path not found: {root}", file=sys.stderr)
        sys.exit(2)

    model = build_repo_model(root, include_tests=args.include_tests, encoding=args.encoding,
                             excludes=DEFAULT_EXCLUDES + list(args.exclude))

    # Detect entrypoints
    entry_qns: Set[str] = set()
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
            # Try dotted module path form
            entry_mod = rel_module_from_path(root, entry_path)
            for qn in list(model.defs.keys()):
                if qn.startswith(f"{entry_mod}:"):
                    # use 'main' if present as a default guess
                    if qn.endswith(":main"):
                        entry_qns.add(qn)
    # User overrides
    for fn in args.entry_func:
        # try to map to module:fn in entry module first, then global by name
        if args.entry:
            entry_path = (root / args.entry).resolve()
            entry_mod = rel_module_from_path(root, entry_path)
            candidate = f"{entry_mod}:{fn}"
            if candidate in model.defs:
                entry_qns.add(candidate)
                continue
        # fallback by simple name index
        for qn in model.name_index.get(fn, []):
            entry_qns.add(qn)

    # If still empty, fallback: any top-level 'main' by name
    if not entry_qns and "main" in model.name_index:
        entry_qns |= model.name_index["main"]

    reachable = compute_reachable(model, entry_qns)
    dup_code_qns, name_dups = detect_duplicates(model)
    md = render_markdown(model, root, reachable, dup_code_qns, name_dups, entry_qns, max_edges=args.max_edges)

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Wrote Markdown bundle to: {args.output}")

if __name__ == "__main__":
    main()
