from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional


def _hash_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            sha.update(buf)
    return sha.hexdigest()


def safe_hash(path: Optional[str]) -> str:
    if not path:
        return "missing"
    ap = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(ap):
        return "missing"
    try:
        return _hash_file(ap)
    except Exception:
        return "unreadable"


def collect_model_identity(
    ckpt_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    feature_backbone_version: Optional[str] = None,
) -> Dict[str, str]:
    ckpt_abs = os.path.abspath(os.path.expanduser(ckpt_path)) if ckpt_path else ""
    cfg_abs = os.path.abspath(os.path.expanduser(cfg_path)) if cfg_path else ""
    return {
        "ckpt_path": ckpt_abs,
        "cfg_path": cfg_abs,
        "ckpt_hash": safe_hash(ckpt_abs) if ckpt_abs else "missing",
        "cfg_hash": safe_hash(cfg_abs) if cfg_abs else "missing",
        "feature_backbone_version": str(feature_backbone_version or "unknown"),
    }


def model_identity_matches(current: Dict[str, str], cached: Dict[str, str]) -> bool:
    # Hard rule: checkpoint hash and text-bank version must match.
    cur_ckpt = str(current.get("ckpt_hash") or "missing")
    old_ckpt = str(cached.get("ckpt_hash") or "missing")
    if cur_ckpt != old_ckpt:
        return False

    cur_cfg = str(current.get("cfg_hash") or "missing")
    old_cfg = str(cached.get("cfg_hash") or "missing")
    if cur_cfg != old_cfg:
        return False

    cur_backbone = str(current.get("feature_backbone_version") or "unknown")
    old_backbone = str(cached.get("feature_backbone_version") or "unknown")
    if cur_backbone != old_backbone:
        return False

    return True
