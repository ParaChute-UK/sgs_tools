import json
import subprocess
from datetime import datetime
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
    diff = _git(["diff", "HEAD", "--", ":(exclude)*.ipynb"], cwd=root)
    return diff.strip()


def _diff_untracked(root):
    files = _git(["ls-files", "--others", "--exclude-standard"], cwd=root)
    files = [f for f in files.splitlines() if f and not f.endswith(".ipynb")]
    diffs = []
    for f in files:
        git_diff = _git(["diff", "--no-index", "/dev/null", f], cwd=root)
        if git_diff:
            diffs.append(git_diff)
    return "\n".join(diffs).strip()


def get_git_state(verbosity=1):
    root = _root()
    if not root:
        return "<no git repo>"

    sha = _git(["rev-parse", "--short", "HEAD"], cwd=root) or "<no commit>"
    # include untracked files, exclude notebooks
    status = _git(["status", "--porcelain", "--", ":!*.ipynb"], cwd=root)
    clean = not bool(status.strip())
    out = {}
    # 1 User: Commit hash + dirty flag
    commit = f"{sha}{'' if clean else '+dirty'}"
    if verbosity > 0:
        out["Commit"] = commit

    # 2 Development: list changed/untracked files
    if verbosity > 1:
        files = _git(["status", "--porcelain"], cwd=root)
        if files:
            out["Files"] = files.strip()

    # 3 Full debug/state logging
    if verbosity > 2:
        tracked = _diff_tracked(root)
        if tracked:
            out["Tracked"] = tracked

        untracked = _diff_untracked(root)
        if untracked:
            out["Untracked"] = untracked
    return out


def print_git_state(verbosity=1):
    git_info = get_git_state(verbosity)
    print("\n".join([f"{k}:\n{v}" for k, v in git_info.items()]))


def write_git_diff_file(git_info: dict[str, str], out_path: Path | str = ".") -> str:
    """
    Write full Git info (Files + Changes) to an external JSON file.
    Relies on existence of Files and Commit keys.
    :param git_info: Dictionary returned by get_git_state().
    :param out_path: Location of output JSON file. Will create if missing.
    :return: Path to the written JSON file.
    """
    state = {k: v.strip().splitlines() for k, v in git_info.items()}
    state_is_dirty = bool(state.get("Files", ""))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    state["date-time"] = [timestamp]

    fname = f"sgs_tools_v{state['Commit'][0]}.json"
    full_path = Path(out_path) / fname
    full_path.parent.mkdir(parents=True, exist_ok=True)
    # add a run-time time-stamp for different dirty versions
    if full_path.exists() and state_is_dirty:
        full_path = full_path.with_stem(f"{full_path.stem}_{timestamp}")

    with open(full_path, "w") as f:
        json.dump(state, f, indent=2)
    return str(full_path)
