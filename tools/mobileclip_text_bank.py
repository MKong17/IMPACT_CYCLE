#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[MOBILECLIP][ERROR] PyTorch is required: {exc}")

try:
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[MOBILECLIP][ERROR] transformers is required: {exc}")


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


def _load_input(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _save_output(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _encode_texts(
    texts: Sequence[str],
    model_name: str,
    *,
    local_only: bool,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=bool(local_only),
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=bool(local_only),
    )
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    inputs = tokenizer(
        list(texts or []),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {
        str(k): v.to(device) if hasattr(v, "to") else v
        for k, v in dict(inputs).items()
    }
    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            feats = model.get_text_features(**inputs)
        else:
            out = model(**inputs)
            if hasattr(out, "text_embeds") and out.text_embeds is not None:
                feats = out.text_embeds
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                feats = out.last_hidden_state[:, 0]
            else:
                raise RuntimeError("Model does not expose usable text features.")
    arr = feats.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(texts or []):
        raise RuntimeError("Unexpected text feature shape.")
    return _normalize_rows(arr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a MobileCLIP/CLIP label text bank.")
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-name", default="")
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    payload = _load_input(args.input_json)
    texts = [str(x or "").strip() for x in (payload.get("texts") or [])]
    classes = [str(x or "").strip() for x in (payload.get("classes") or [])]
    feature_dim = int(payload.get("feature_dim", 0) or 0)
    model_name = str(args.model_name or payload.get("model_name") or "").strip()
    if not texts or not classes or len(texts) != len(classes) or feature_dim <= 0:
        raise SystemExit("[MOBILECLIP][ERROR] Invalid label text bank request payload.")
    if not model_name:
        raise SystemExit("[MOBILECLIP][ERROR] --model-name is required.")

    raw = _encode_texts(texts, model_name=model_name, local_only=bool(args.local_only))
    table = _project_embeddings(
        raw,
        out_dim=int(feature_dim),
        seed_key=f"hf_clip:{model_name}",
    )
    backend_name = "hf_mobileclip" if "mobileclip" in model_name.lower() else "hf_clip"
    _save_output(
        args.output,
        {
            "version": 1,
            "kind": "east_label_text_bank",
            "backend": backend_name,
            "model_name": model_name,
            "classes": list(classes),
            "raw_dim": int(raw.shape[1]) if raw.ndim == 2 else 0,
            "feature_dim": int(feature_dim),
            "text_table": table.astype(np.float32),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
