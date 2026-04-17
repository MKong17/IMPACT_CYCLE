import json
import os
import sys
from typing import Iterable, List, Optional


def load_feature_env_defaults(
    *,
    repo_root: str,
    config_env_key: str = "FEATURE_CONFIG",
    default_filename: str = "feature_defaults.json",
) -> None:
    raw_path = str(os.environ.get(config_env_key, "") or "").strip()
    if raw_path:
        cfg_path = os.path.abspath(os.path.expanduser(raw_path))
    else:
        cfg_path = os.path.abspath(os.path.join(repo_root, default_filename))
    if not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    cfg_dir = os.path.dirname(cfg_path)
    path_like_keys = {"EAST_REPO_ROOT", "EAST_CFG", "EAST_CKPT", "RUNNER_ENV_CONFIG"}
    for key, value in payload.items():
        if not key or not isinstance(key, str):
            continue
        if key in os.environ or value is None:
            continue
        text = str(value)
        if key in path_like_keys and text and not os.path.isabs(text):
            candidate = os.path.abspath(os.path.join(cfg_dir, text))
            if os.path.exists(candidate):
                text = candidate
        os.environ[key] = text


def build_runner_cmd(
    *,
    repo_root: str,
    profile: str,
    script_path: str,
    script_args: Optional[Iterable[str]] = None,
    python_executable: str = "",
) -> List[str]:
    runner = os.path.abspath(
        os.path.join(repo_root, "tools", "runners", "run_in_env.py")
    )
    script_abs = os.path.abspath(os.path.expanduser(str(script_path)))
    args = [str(x) for x in (script_args or [])]
    py = str(python_executable or sys.executable or "python")
    if os.path.isfile(runner):
        return [
            py,
            runner,
            "--profile",
            str(profile or "current"),
            "--",
            script_abs,
            *args,
        ]
    return [py, script_abs, *args]
