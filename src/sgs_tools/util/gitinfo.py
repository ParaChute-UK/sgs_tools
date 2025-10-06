import subprocess
from pathlib import Path


def _git(args, cwd=None):
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            check=False,
            capture_output=True,
        ).stdout.strip()
    except Exception:
        return ""


def _root(start=None):
    p = Path(start or __file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _diff_tracked(root):
    diff = _git(["diff"], cwd=root)
    if not diff:
        return ""
    lines, skip = [], False
    for line in diff.splitlines():
        if line.startswith("diff --git") and line.endswith(".ipynb"):
            skip = True
            continue
        if skip and line.startswith("diff --git"):
            skip = False
        if not skip:
            lines.append(line)
    return "\n".join(lines).strip()


def _diff_untracked(root):
    files = _git(["ls-files", "--others", "--exclude-standard"], cwd=root)
    # skip .ipynb files -- too large diffs
    files = [f for f in files.splitlines() if not f.endswith(".ipynb")]
    diffs = ""
    if files:
        diff = _git(["diff", "--no-index", "/dev/null", *files], cwd=root)
        if diff:
            diffs.append(diff)
    return "\n".join(diffs).strip()


def get_git_state(level=0):
    root = _root()
    if not root:
        return "<no git repo>"

    sha = _git(["rev-parse", "--short", "HEAD"], cwd=root) or "<no commit>"
    status = _git(["status", "--porcelain"], cwd=root)
    clean = not bool(status.strip())
    out = {}
    # 1 User: Commit hash + dirty flag
    commit = [f"{sha}{'' if clean else ' +dirty'}"]
    if level >= 0:
        out["Commit"] = commit

    # 2 Development: list changed/untracked files
    if level >= 1:
        files = _git(["status", "--porcelain"], cwd=root)
        if files:
            out["Files"] = files.strip()

    # 3 Full debug/state logging
    if level >= 2:
        diff = ""
        tracked = _diff_tracked(root)
        if tracked:
            diff += tracked

        untracked = _diff_untracked(root)
        if untracked:
            diff += "\nUntracked:\n"
            diff += untracked
        # diff = 'n'.join(diff).strip()
        out["Changes"] = diff.strip()
    return out


def print_git_state(level=0):
    state = get_git_state(level)
    print("\n".join([f"{k}:\n {v}" for k, v in state.items()]))
