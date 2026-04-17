#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a Python script in a selected conda environment profile.

Usage:
  python tools/runners/run_in_env.py --profile asot -- tools/asot_full_infer_adapter.py --help
  python tools/runners/run_in_env.py --profile mobileclip -- tools/mobileclip_infer.py --video xxx.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Union


PROFILE_DEFAULT_ENVS: Dict[str, str] = {
    "ui": "current",
    "opentad": "current",
    "east": "current",
    "mobileclip": "mobileclip",
    "asot": "current",
    "qwen": "qwen",
    "sam3": "current",
    "sam3_windows": "current",
    "sam3_wsl": "current",
}

EnvSpec = Union[str, Dict[str, Any]]


_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _default_env_config_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw = str(os.environ.get("RUNNER_ENV_CONFIG", "runner_envs.json") or "").strip()
    if not raw:
        raw = "runner_envs.json"
    if not os.path.isabs(raw):
        raw = os.path.abspath(os.path.join(repo_root, raw))
    return raw


def _load_env_config() -> Dict[str, EnvSpec]:
    cfg_path = _default_env_config_path()
    if not os.path.isfile(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("profiles"), dict):
        payload = payload.get("profiles")
    if not isinstance(payload, dict):
        return {}
    resolved: Dict[str, EnvSpec] = {}
    for k, v in payload.items():
        key = str(k or "").strip().lower()
        if not key:
            continue
        if isinstance(v, dict):
            launcher = str(v.get("launcher", "") or "").strip().lower()
            if launcher:
                resolved[key] = dict(v)
            continue
        val = str(v or "").strip()
        if val:
            resolved[key] = val
    return resolved


def _resolve_env_name(profile: str, explicit_env: str) -> EnvSpec:
    if explicit_env:
        return str(explicit_env).strip()
    key = str(profile or "opentad").strip().lower() or "opentad"

    env_key = f"RUNNER_ENV_{key.upper()}"
    if os.environ.get(env_key):
        return str(os.environ.get(env_key) or "").strip()

    if key == "asot" and os.environ.get("ASOT_CONDA_ENV"):
        return str(os.environ.get("ASOT_CONDA_ENV") or "").strip()
    if key == "asot" and os.environ.get("ASOT_CONDA_PREFIX"):
        return str(os.environ.get("ASOT_CONDA_PREFIX") or "").strip()
    if key == "mobileclip" and os.environ.get("MOBILECLIP_CONDA_ENV"):
        return str(os.environ.get("MOBILECLIP_CONDA_ENV") or "").strip()
    if key == "mobileclip" and os.environ.get("MOBILECLIP_CONDA_PREFIX"):
        return str(os.environ.get("MOBILECLIP_CONDA_PREFIX") or "").strip()
    if key in {"sam3", "sam3_windows"} and os.environ.get("SAM3_CONDA_ENV"):
        return str(os.environ.get("SAM3_CONDA_ENV") or "").strip()
    if key in {"sam3", "sam3_windows"} and os.environ.get("SAM3_CONDA_PREFIX"):
        return str(os.environ.get("SAM3_CONDA_PREFIX") or "").strip()
    if key in {"opentad", "east", "ui"} and os.environ.get("OPENTAD_CONDA_ENV"):
        return str(os.environ.get("OPENTAD_CONDA_ENV") or "").strip()
    if key in {"opentad", "east", "ui"} and os.environ.get("OPENTAD_CONDA_PREFIX"):
        return str(os.environ.get("OPENTAD_CONDA_PREFIX") or "").strip()

    config_map = _load_env_config()
    if config_map.get(key):
        return config_map.get(key)  # type: ignore[return-value]

    # Common local layout fallback: /home/.../conda_envs/<name>
    home = os.path.expanduser("~")
    if key in {"asot", "mobileclip"}:
        cand = os.path.join(home, "IsaacDrive", "conda_envs", key)
        if os.path.isdir(cand):
            return cand

    return PROFILE_DEFAULT_ENVS.get(key, key)


def _windows_to_wsl_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return raw
    normalized = raw.replace("\\", "/")
    if not _WINDOWS_ABS_RE.match(raw):
        return normalized
    drive = normalized[0].lower()
    suffix = normalized[2:]
    suffix = suffix.lstrip("/")
    return f"/mnt/{drive}/{suffix}"


def _maybe_wsl_path_arg(arg: str) -> str:
    raw = str(arg or "")
    if _WINDOWS_ABS_RE.match(raw):
        return _windows_to_wsl_path(raw)
    return raw


def _build_wsl_exec_cmd(
    script_path: str,
    script_args: List[str],
    env_spec: Dict[str, Any],
    cwd: str,
    python_bin: str,
) -> List[str]:
    distro = str(env_spec.get("distro", "Ubuntu-22.04") or "Ubuntu-22.04").strip()
    wsl_exe = str(env_spec.get("wsl_exe", "wsl.exe") or "wsl.exe").strip()
    python_target = str(
        env_spec.get("python_bin", "") or env_spec.get("python", "") or python_bin or "python3"
    ).strip()
    if not python_target:
        python_target = "python3"
    script_target = _windows_to_wsl_path(os.path.abspath(os.path.expanduser(script_path)))
    arg_targets = [_maybe_wsl_path_arg(arg) for arg in script_args]
    wsl_cwd = str(env_spec.get("cwd", "") or cwd or "").strip()
    if wsl_cwd:
        wsl_cwd = _windows_to_wsl_path(os.path.abspath(os.path.expanduser(wsl_cwd)))
    cmd = [wsl_exe, "-d", distro]
    if wsl_cwd:
        cmd.extend(["--cd", wsl_cwd])
    cmd.extend(["--", python_target, script_target, *arg_targets])
    return cmd


def _build_exec_cmd(
    script_path: str,
    script_args: List[str],
    profile: str,
    env_name: str,
    conda_exe: str,
    python_bin: str,
    cwd: str,
) -> List[str]:
    target_env = _resolve_env_name(profile=profile, explicit_env=env_name)
    if isinstance(target_env, dict):
        launcher = str(target_env.get("launcher", "") or "").strip().lower()
        if launcher == "wsl":
            return _build_wsl_exec_cmd(
                script_path=script_path,
                script_args=script_args,
                env_spec=target_env,
                cwd=cwd,
                python_bin=python_bin,
            )

    target_env = str(target_env or "").strip()
    script_abs = os.path.abspath(os.path.expanduser(script_path))

    # Allow opting out of conda by setting env to current/self/none.
    if target_env.lower() in {"", "current", "self", "none"}:
        return [sys.executable, script_abs, *script_args]

    if os.path.isdir(target_env):
        return [conda_exe, "run", "--no-capture-output", "-p", target_env, python_bin, script_abs, *script_args]
    return [conda_exe, "run", "--no-capture-output", "-n", target_env, python_bin, script_abs, *script_args]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a script in a configured conda environment profile.")
    ap.add_argument("--profile", default="opentad", help="Environment profile: opentad/east/mobileclip/asot/...")
    ap.add_argument("--env-name", default="", help="Override conda env name directly.")
    ap.add_argument("--conda-exe", default=os.environ.get("CONDA_EXE", "conda"), help="conda executable path.")
    ap.add_argument("--python-bin", default="python", help="Python binary name inside target env.")
    ap.add_argument("--cwd", default="", help="Optional working directory.")
    ap.add_argument("--print-cmd", action="store_true", help="Print resolved command before running.")
    args, rest = ap.parse_known_args()

    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        print("[RUNNER][ERROR] Missing script path. Use: run_in_env.py --profile <p> -- <script.py> <args...>", file=sys.stderr)
        return 2

    script_path = rest[0]
    script_args = rest[1:]
    cmd = _build_exec_cmd(
        script_path=script_path,
        script_args=script_args,
        profile=args.profile,
        env_name=args.env_name,
        conda_exe=args.conda_exe,
        python_bin=args.python_bin,
        cwd=args.cwd,
    )

    if args.print_cmd:
        print("[RUNNER] " + " ".join(cmd))

    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        proc = subprocess.Popen(cmd, cwd=(args.cwd or None), env=child_env)
    except FileNotFoundError as exc:
        print(f"[RUNNER][ERROR] Command not found: {exc}", file=sys.stderr)
        return 127
    except Exception as exc:
        print(f"[RUNNER][ERROR] Failed to launch process: {exc}", file=sys.stderr)
        return 1
    return int(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())
