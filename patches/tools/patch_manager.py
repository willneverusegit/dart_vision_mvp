#!/usr/bin/env python3
"""
PatchManager: safe integration of `git apply` / `git am` with optional sandbox via `git worktree`.

Additions:
- --sandbox: apply in a separate worktree on a new branch
- --tests "<cmd>": run tests inside the sandbox (e.g., "pytest -q")
- --push: push sandbox branch on success
- --keep-on-success: keep worktree directory (default: removed when not kept)
- --posthook "<cmd>": run arbitrary post-apply hook

Usage:
  python tools/patch_manager.py --patch patches/x.patch --dry-run
  python tools/patch_manager.py --patch patches/x.patch --branch --commit
  python tools/patch_manager.py --patch patches/x.patch --sandbox --tests "pytest -q" --commit --push
"""
from __future__ import annotations
import argparse, subprocess, sys, time, os, shlex, textwrap, tempfile
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT_CMD = ["git", "rev-parse", "--show-toplevel"]

def run(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=check)

def shell(cmd: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, shell=True, capture_output=True, text=True)

def ensure_git_repo() -> Path:
    try:
        cp = run(REPO_ROOT_CMD)
        return Path(cp.stdout.strip())
    except subprocess.CalledProcessError as e:
        raise SystemExit("Not inside a Git repository (rev-parse failed).") from e

def is_worktree_clean(cwd: Optional[Path] = None) -> bool:
    cp = run(["git", "status", "--porcelain"], cwd=cwd, check=True)
    return cp.stdout.strip() == ""

def detect_mbox(patch_text: str) -> bool:
    head = patch_text.lstrip().splitlines()[:1]
    return bool(head and head[0].startswith("From "))

def load_patch(patch_path: str) -> str:
    if patch_path == "-":
        return sys.stdin.read()
    return Path(patch_path).read_text(encoding="utf-8")

def write_log(repo_root: Path, entry: str) -> None:
    log_dir = repo_root / "patches"
    log_dir.mkdir(exist_ok=True, parents=True)
    (log_dir / "applied.log").open("a", encoding="utf-8").write(entry + "\n")

def create_branch(name: Optional[str]) -> str:
    if not name:
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"patch/{ts}"
    run(["git", "checkout", "-b", name], check=True)
    return name

def commit_all(message: str, cwd: Optional[Path] = None) -> str:
    run(["git", "add", "-A"], cwd=cwd, check=True)
    run(["git", "commit", "-m", message], cwd=cwd, check=True)
    cp = run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True)
    return cp.stdout.strip()

def apply_with_git_apply(patch_text: str, three_way: bool, reverse: bool, index: bool, dry_run: bool, cwd: Optional[Path] = None) -> Tuple[bool, str]:
    flags = ["git", "apply", "--whitespace=fix", "--recount"]
    if three_way: flags.append("--3way")
    if reverse:   flags.append("-R")
    if index:     flags.append("--index")
    if dry_run:   flags.append("--check")
    try:
        cp = subprocess.run(flags, input=patch_text, text=True, capture_output=True, check=True, cwd=str(cwd) if cwd else None)
        return True, cp.stderr.strip() or cp.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or e.stdout or str(e))

def apply_with_git_am(patch_text: str, three_way: bool, signoff: bool, dry_run: bool, cwd: Optional[Path] = None) -> Tuple[bool, str]:
    flags = ["git", "am"]
    if signoff: flags.append("--signoff")
    if three_way: flags.append("--3way")
    if dry_run: flags.append("--dry-run")
    try:
        cp = subprocess.run(flags, input=patch_text, text=True, capture_output=True, check=True, cwd=str(cwd) if cwd else None)
        return True, cp.stderr.strip() or cp.stdout.strip()
    except subprocess.CalledProcessError as e:
        if not dry_run:
            subprocess.run(["git", "am", "--abort"], capture_output=True, cwd=str(cwd) if cwd else None)
        return False, (e.stderr or e.stdout or str(e))

def apply_patch(
    patch_source: str,
    *,
    use_am: bool = False,
    three_way: bool = True,
    reverse: bool = False,
    allow_dirty: bool = False,
    auto_branch: bool = False,
    auto_commit: bool = False,
    commit_message: Optional[str] = None,
    index: bool = False,
    dry_run: bool = False,
    cwd: Optional[Path] = None,
) -> dict:
    repo_root = ensure_git_repo()
    if not allow_dirty and not is_worktree_clean(cwd):
        raise SystemExit("Working tree is not clean. Commit/stash or pass --allow-dirty.")

    patch_text = load_patch(patch_source)
    mode = "am" if (use_am or detect_mbox(patch_text)) else "apply"
    branch_name = None
    if auto_branch and not dry_run and cwd is None:
        branch_name = create_branch(None)

    if mode == "am":
        ok, details = apply_with_git_am(patch_text, three_way=three_way, signoff=True, dry_run=dry_run, cwd=cwd)
    else:
        ok, details = apply_with_git_apply(patch_text, three_way=three_way, reverse=reverse, index=index, dry_run=dry_run, cwd=cwd)

    commit_sha = None
    if ok and auto_commit and not dry_run and mode != "am":
        msg = commit_message or f"Apply patch: {Path(patch_source).name if patch_source!='-' else 'stdin'}"
        commit_sha = commit_all(msg, cwd=cwd)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{ts}] mode={mode} ok={ok} branch={branch_name or '-'} commit={commit_sha or '-'} src={patch_source} details={(details or '').replace(os.linesep,' / ')}"
    write_log(repo_root, log_entry)

    return {"ok": ok, "mode": mode, "branch": branch_name, "commit": commit_sha, "details": details}

def apply_in_sandbox(
    patch_source: str,
    *,
    tests_cmd: Optional[str] = None,
    commit: bool = False,
    push: bool = False,
    keep_on_success: bool = False,
    posthook: Optional[str] = None,
) -> dict:
    repo_root = ensure_git_repo()
    ts = time.strftime("%Y%m%d_%H%M%S")
    branch_name = f"patch/{ts}"
    wt_dir = repo_root / ".worktrees" / f"patch_{ts}"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)

    # create worktree at HEAD on new branch
    run(["git", "worktree", "add", "-b", branch_name, str(wt_dir), "HEAD"], check=True)

    try:
        # apply inside worktree
        res = apply_patch(
            patch_source,
            three_way=True,
            reverse=False,
            allow_dirty=False,
            auto_branch=False,
            auto_commit=commit,
            commit_message=f"Apply patch in sandbox: {Path(patch_source).name if patch_source!='-' else 'stdin'}",
            index=False,
            dry_run=False,
            cwd=wt_dir,
        )

        if not res["ok"]:
            raise SystemExit(f"Patch failed in sandbox: {res['details']}")

        # run tests
        if tests_cmd:
            t = shell(tests_cmd, cwd=wt_dir)
            res["tests_stdout"] = t.stdout[-4000:]
            res["tests_stderr"] = t.stderr[-4000:]
            res["tests_rc"] = t.returncode
            if t.returncode != 0:
                raise SystemExit(f"Tests failed in sandbox (rc={t.returncode}).")

        # optional posthook
        if posthook:
            h = shell(posthook, cwd=wt_dir)
            res["posthook_stdout"] = h.stdout[-4000:]
            res["posthook_stderr"] = h.stderr[-4000:]
            res["posthook_rc"] = h.returncode
            if h.returncode != 0:
                raise SystemExit(f"Posthook failed (rc={h.returncode}).")

        # optional push
        if push:
            pr = run(["git", "push", "-u", "origin", branch_name], cwd=wt_dir, check=True)
            res["push"] = pr.stdout.strip() or pr.stderr.strip()

        res["sandbox_dir"] = str(wt_dir)
        res["branch"] = branch_name
        res["ok"] = True
        # clean up or keep
        if not keep_on_success:
            run(["git", "worktree", "remove", "--force", str(wt_dir)])
        return res

    except SystemExit as e:
        # cleanup on failure
        run(["git", "worktree", "remove", "--force", str(wt_dir)], check=False)
        run(["git", "branch", "-D", branch_name], check=False)
        write_log(repo_root, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SANDBOX FAIL branch={branch_name} src={patch_source} err={e}")
        return {"ok": False, "branch": branch_name, "error": str(e)}

def main():
    p = argparse.ArgumentParser(description="Safely integrate `git apply` (or `git am`) with optional sandbox.")
    p.add_argument("--patch", required=True, help="Path to .patch/.diff/.mbox or '-' for stdin")
    p.add_argument("--am", action="store_true", help="Force use of `git am` (email/mbox patches)")
    p.add_argument("--reverse", "-R", action="store_true", help="Reverse apply")
    p.add_argument("--no-3way", dest="three_way", action="store_false", help="Disable 3-way fallback")
    p.add_argument("--index", action="store_true", help="Update index (`git apply --index`)")
    p.add_argument("--allow-dirty", action="store_true", help="Skip clean working tree check")
    p.add_argument("--branch", action="store_true", help="Auto-create branch before applying (non-sandbox)")
    p.add_argument("--commit", action="store_true", help="Auto-commit changes (not used with `git am`)")
    p.add_argument("--message", default=None, help="Commit message if --commit")
    p.add_argument("--dry-run", action="store_true", help="Validate without changing files")

    # sandbox options
    p.add_argument("--sandbox", action="store_true", help="Apply patch in a temporary worktree & run tests")
    p.add_argument("--tests", default=None, help='Command to run tests, e.g., "pytest -q"')
    p.add_argument("--push", action="store_true", help="Push sandbox branch on success")
    p.add_argument("--keep-on-success", action="store_true", help="Keep worktree dir after success (default: remove)")
    p.add_argument("--posthook", default=None, help='Optional post-apply hook command')

    args = p.parse_args()

    if args.sandbox:
        res = apply_in_sandbox(
            patch_source=args.patch,
            tests_cmd=args.tests,
            commit=args.commit,
            push=args.push,
            keep_on_success=args.keep_on_success,
            posthook=args.posthook,
        )
        status = "OK ✅" if res.get("ok") else "FAILED ❌"
        print(textwrap.dedent(f"""
        Status: {status}
        Mode:   sandbox
        Branch: {res.get('branch','-')}
        Dir:    {res.get('sandbox_dir','-')}
        Details:
          {res.get('error','(applied & tested)')}
        """).strip())
        sys.exit(0 if res.get("ok") else 1)
    else:
        res = apply_patch(
            patch_source=args.patch,
            use_am=args.am,
            three_way=args.three_way,
            reverse=args.reverse,
            allow_dirty=args.allow_dirty,
            auto_branch=args.branch,
            auto_commit=args.commit,
            commit_message=args.message,
            index=args.index,
            dry_run=args.dry_run,
        )
        status = "OK ✅" if res["ok"] else "FAILED ❌"
        print(textwrap.dedent(f"""
        Status: {status}
        Mode:   {res['mode']}
        Branch: {res['branch'] or '-'}
        Commit: {res['commit'] or '-'}
        Details:
          {res['details']}
        """).strip())
        sys.exit(0 if res["ok"] else 1)

if __name__ == "__main__":
    main()
