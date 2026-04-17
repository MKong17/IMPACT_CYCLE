from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_load(path: str) -> Any:
    if torch is not None:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            try:
                return torch.load(path, map_location="cpu")
            except Exception:
                pass
        except Exception:
            pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _safe_save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if torch is not None:
        try:
            torch.save(obj, path)
            return
        except Exception:
            pass
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _normalize_rows(table: np.ndarray) -> np.ndarray:
    arr = np.asarray(table, dtype=np.float32)
    if arr.ndim != 2 or arr.size <= 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return arr / norms


def _stable_hash(parts: Sequence[str]) -> str:
    blob = "||".join(str(x or "") for x in parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(blob).hexdigest()


def _runtime_dir(features_dir: str) -> str:
    return os.path.join(
        os.path.abspath(os.path.expanduser(str(features_dir or "").strip())),
        "east_runtime",
    )


def _bank_path(features_dir: str) -> str:
    return os.path.join(_runtime_dir(features_dir), "label_text_bank.pt")


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _normalize_labels(classes: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in classes or []:
        name = str(raw or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _prompt_labels(labels: Sequence[str], prompt_template: str) -> List[str]:
    tpl = str(prompt_template or "action: {}").strip() or "{}"
    out: List[str] = []
    for name in labels or []:
        if "{}" in tpl:
            out.append(tpl.format(str(name)))
        else:
            out.append(f"{tpl} {name}".strip())
    return out


def _lexical_hash_embeddings(texts: Sequence[str], dim: int) -> np.ndarray:
    dim = int(max(8, dim))
    table = np.zeros((len(texts or []), dim), dtype=np.float32)
    for row_idx, raw_text in enumerate(texts or []):
        text = str(raw_text or "").strip().lower()
        if not text:
            continue
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]
        feats = set(tokens)
        chars = text.replace(" ", "_")
        for n in (2, 3, 4):
            if len(chars) < n:
                continue
            for idx in range(len(chars) - n + 1):
                feats.add(chars[idx : idx + n])
        for feat in feats:
            digest = hashlib.sha1(feat.encode("utf-8")).digest()
            col = int.from_bytes(digest[:4], "little", signed=False) % dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            table[row_idx, int(col)] += np.float32(sign)
        if float(np.linalg.norm(table[row_idx])) <= 1e-6:
            table[row_idx, row_idx % dim] = 1.0
    return _normalize_rows(table)


def _sentence_transformer_embeddings(
    texts: Sequence[str],
    model_name: str,
) -> Optional[np.ndarray]:
    if SentenceTransformer is None:
        return None
    try:
        model = SentenceTransformer(model_name)
        arr = model.encode(
            list(texts or []),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(texts or []):
        return None
    return _normalize_rows(arr)


def _default_sentence_model_name() -> str:
    return str(
        os.environ.get("EAST_TEXT_BANK_SENTENCE_MODEL")
        or os.environ.get("EAST_TEXT_BANK_MODEL")
        or "sentence-transformers/all-MiniLM-L6-v2"
    ).strip()


def _default_clip_model_name() -> str:
    return str(
        os.environ.get("EAST_TEXT_BANK_CLIP_MODEL")
        or "apple/MobileCLIP-S2"
    ).strip()


def _allow_remote_model_fetch() -> bool:
    raw = str(os.environ.get("EAST_TEXT_BANK_ALLOW_REMOTE", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _mobileclip_runner_embeddings(
    features_dir: str,
    texts: Sequence[str],
    classes: Sequence[str],
    feature_dim: int,
    *,
    model_name: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    repo_root = _repo_root()
    runner_path = os.path.join(repo_root, "tools", "runners", "run_in_env.py")
    script_path = os.path.join(repo_root, "tools", "mobileclip_text_bank.py")
    if not (os.path.isfile(runner_path) and os.path.isfile(script_path)):
        return None

    runtime_dir = _runtime_dir(features_dir)
    try:
        os.makedirs(runtime_dir, exist_ok=True)
    except Exception:
        pass
    req_path = os.path.join(runtime_dir, "_east_text_bank_request.json")
    out_path = os.path.join(runtime_dir, "_east_text_bank_response.pkl")
    try:
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "texts": list(texts or []),
                    "classes": list(classes or []),
                    "feature_dim": int(feature_dim),
                    "model_name": str(model_name or "").strip(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        cmd = [
            sys.executable,
            runner_path,
            "--profile",
            "mobileclip",
            "--",
            script_path,
            "--input-json",
            req_path,
            "--output",
            out_path,
            "--model-name",
            str(model_name or "").strip(),
        ]
        if not _allow_remote_model_fetch():
            cmd.append("--local-only")
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0 or not os.path.isfile(out_path):
            if progress_cb is not None:
                tail = str((proc.stderr or proc.stdout or "")[-240:]).strip()
                if tail:
                    progress_cb(f"[EAST][TEXT] MobileCLIP text bank unavailable, falling back. {tail}")
            return None
        obj = _safe_load(out_path)
        if not isinstance(obj, dict):
            return None
        table = np.asarray(obj.get("text_table"), dtype=np.float32) if obj.get("text_table") is not None else np.zeros((0, 0), dtype=np.float32)
        if table.ndim != 2 or table.shape != (len(classes or []), int(feature_dim)):
            return None
        return {
            "backend": str(obj.get("backend") or "mobileclip_runner"),
            "model_name": str(obj.get("model_name") or str(model_name or "").strip()),
            "raw_dim": int(obj.get("raw_dim", table.shape[1]) or table.shape[1]),
            "text_table": _normalize_rows(table),
        }
    except Exception as exc:
        if progress_cb is not None:
            progress_cb(f"[EAST][TEXT] MobileCLIP text bank unavailable, falling back. {exc}")
        return None
    finally:
        for path in (req_path, out_path):
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass


def _project_embeddings(table: np.ndarray, out_dim: int, seed_key: str) -> np.ndarray:
    arr = np.asarray(table, dtype=np.float32)
    out_dim = int(max(8, out_dim))
    if arr.ndim != 2 or arr.shape[0] <= 0:
        return np.zeros((0, out_dim), dtype=np.float32)
    if int(arr.shape[1]) == out_dim:
        return _normalize_rows(arr.astype(np.float32))
    seed = int(_stable_hash([seed_key, str(arr.shape[1]), str(out_dim)])[:8], 16)
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((int(arr.shape[1]), out_dim), dtype=np.float32)
    proj = proj / np.sqrt(max(1, int(arr.shape[1])))
    out = np.asarray(arr @ proj, dtype=np.float32)
    return _normalize_rows(out)


def ensure_label_text_bank(
    features_dir: str,
    classes: Sequence[str],
    feature_dim: int,
    *,
    backend: str = "auto",
    model_name: str = "",
    prompt_template: str = "action: {}",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    def _emit(msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(str(msg))
            except Exception:
                pass

    features_dir = os.path.abspath(os.path.expanduser(str(features_dir or "").strip()))
    runtime_dir = _runtime_dir(features_dir)
    os.makedirs(runtime_dir, exist_ok=True)
    labels = _normalize_labels(classes)
    feature_dim = int(max(1, feature_dim))
    backend = str(backend or os.environ.get("EAST_TEXT_BANK_BACKEND", "auto")).strip().lower() or "auto"
    sentence_model_name = _default_sentence_model_name()
    clip_model_name = _default_clip_model_name()
    if model_name:
        if backend in {"mobileclip", "mobileclip_runner", "clip"}:
            clip_model_name = str(model_name).strip()
        else:
            sentence_model_name = str(model_name).strip()
    prompt_template = str(
        prompt_template
        or os.environ.get("EAST_TEXT_BANK_PROMPT", "action: {}")
        or "action: {}"
    )
    bank_path = _bank_path(features_dir)
    label_hash = _stable_hash(
        [
            backend,
            sentence_model_name,
            clip_model_name,
            prompt_template,
            str(feature_dim),
        ]
        + list(labels)
    )

    existing = _safe_load(bank_path) if os.path.isfile(bank_path) else {}
    if isinstance(existing, dict):
        same = (
            str(existing.get("label_hash", "") or "") == label_hash
            and int(existing.get("feature_dim", 0) or 0) == feature_dim
        )
        table = np.asarray(existing.get("text_table"), dtype=np.float32) if existing.get("text_table") is not None else np.zeros((0, 0), dtype=np.float32)
        if same and table.ndim == 2 and table.shape == (len(labels), feature_dim):
            return {
                "ok": True,
                "changed": False,
                "path": bank_path,
                "backend": str(existing.get("backend", "") or "unknown"),
                "classes": list(labels),
                "feature_dim": int(feature_dim),
                "text_table": _normalize_rows(table),
            }

    texts = _prompt_labels(labels, prompt_template)
    raw_table: Optional[np.ndarray] = None
    backend_used = backend
    resolved_model_name = ""
    if backend in {"auto", "mobileclip", "mobileclip_runner", "clip"}:
        clip_result = _mobileclip_runner_embeddings(
            features_dir,
            texts,
            labels,
            int(feature_dim),
            model_name=str(clip_model_name or "").strip(),
            progress_cb=progress_cb,
        )
        if clip_result is not None:
            backend_used = str(clip_result.get("backend") or "mobileclip_runner")
            resolved_model_name = str(clip_result.get("model_name") or clip_model_name)
            text_table = np.asarray(clip_result.get("text_table"), dtype=np.float32)
            raw_dim = int(clip_result.get("raw_dim", text_table.shape[1] if text_table.ndim == 2 else 0) or 0)
        else:
            text_table = None
            raw_dim = 0
    else:
        text_table = None
        raw_dim = 0

    if text_table is None and backend in {"auto", "sentence_transformers"}:
        raw_table = _sentence_transformer_embeddings(texts, model_name=sentence_model_name)
        if raw_table is not None:
            backend_used = "sentence_transformers"
            resolved_model_name = str(sentence_model_name or "")
            raw_dim = int(raw_table.shape[1]) if raw_table.ndim == 2 else 0
            _emit(f"[EAST][TEXT] Built label text bank with SentenceTransformer ({sentence_model_name}).")
            text_table = _project_embeddings(
                raw_table,
                out_dim=feature_dim,
                seed_key=f"{backend_used}:{sentence_model_name}:{prompt_template}",
            )

    if text_table is None:
        raw_dim = min(max(64, feature_dim), 512)
        raw_table = _lexical_hash_embeddings(texts, dim=raw_dim)
        backend_used = "hashed_lexical"
        resolved_model_name = ""
        _emit("[EAST][TEXT] Falling back to lexical-hash label text bank.")
        text_table = _project_embeddings(
            raw_table,
            out_dim=feature_dim,
            seed_key=f"{backend_used}:{prompt_template}",
        )

    payload = {
        "version": 1,
        "kind": "east_label_text_bank",
        "created_at": _utc_now(),
        "backend": backend_used,
        "model_name": resolved_model_name,
        "sentence_model_name": sentence_model_name,
        "clip_model_name": clip_model_name,
        "prompt_template": prompt_template,
        "classes": list(labels),
        "label_hash": label_hash,
        "raw_dim": int(raw_dim),
        "feature_dim": int(feature_dim),
        "text_table": text_table.astype(np.float32),
    }
    _safe_save(payload, bank_path)
    return {
        "ok": True,
        "changed": True,
        "path": bank_path,
        "backend": backend_used,
        "classes": list(labels),
        "feature_dim": int(feature_dim),
        "text_table": text_table.astype(np.float32),
    }


def load_label_text_bank_map(
    features_dir: str,
    classes: Sequence[str],
    feature_dim: int,
) -> Dict[str, np.ndarray]:
    bank_path = _bank_path(features_dir)
    if not os.path.isfile(bank_path):
        return {}
    obj = _safe_load(bank_path)
    if not isinstance(obj, dict):
        return {}
    names = _normalize_labels(obj.get("classes") or [])
    table = np.asarray(obj.get("text_table"), dtype=np.float32) if obj.get("text_table") is not None else np.zeros((0, 0), dtype=np.float32)
    if table.ndim != 2 or table.shape != (len(names), int(feature_dim)):
        return {}
    class_set = set(_normalize_labels(classes))
    out: Dict[str, np.ndarray] = {}
    table = _normalize_rows(table)
    for idx, name in enumerate(names):
        if name in class_set:
            out[str(name)] = table[idx]
    return out
