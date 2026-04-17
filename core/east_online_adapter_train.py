import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

from core.east_online_update import (
    _confirmed_kind,
    _load_feature_series,
    _load_json,
    _load_supervision_records,
    _mark_runtime_dirty,
    _supervision_weight,
    _safe_load_object,
    _safe_save_object,
    _stable_records_hash,
)
from core.label_text_bank import ensure_label_text_bank
from core.east_label_fusion import (
    SOURCE_NAMES,
    build_gate_features,
    normalize_rows,
    softmax_rows,
)
from tools.east.online_ieast_adapter import OnlineInteractiveAdapter


def _runtime_classes(runtime_dir: str, records: Sequence[Dict[str, Any]]) -> List[str]:
    seg_path = os.path.join(runtime_dir, "segments.json")
    classes: List[str] = []
    meta_obj = _load_json(seg_path)
    for name in meta_obj.get("classes") or []:
        text = str(name or "").strip()
        if text and text not in classes:
            classes.append(text)
    if classes:
        return classes
    for rec in records or []:
        for key in ("label", "old_label", "from_label", "to_label"):
            text = str(rec.get(key, "") or "").strip()
            if text and text not in classes:
                classes.append(text)
    return classes


def _source_name_from_meta(meta: Dict[str, Any]) -> str:
    return str(
        meta.get("backbone")
        or meta.get("source")
        or meta.get("feature_backbone_version")
        or "unknown"
    ).strip()


def _load_runtime_prototype_table(runtime_dir: str, classes: Sequence[str], dim: int) -> np.ndarray:
    proto_path = os.path.join(runtime_dir, "prototype.npy")
    table = np.zeros((len(classes), dim), dtype=np.float32)
    if not os.path.isfile(proto_path):
        return table
    try:
        arr = np.asarray(np.load(proto_path), dtype=np.float32)
    except Exception:
        return table
    if arr.ndim != 2:
        return table
    rows = min(int(arr.shape[0]), len(classes))
    cols = min(int(arr.shape[1]), dim)
    if rows <= 0 or cols <= 0:
        return table
    table[:rows, :cols] = arr[:rows, :cols]
    return table


def _load_delta_prototype_table(runtime_dir: str, classes: Sequence[str], dim: int) -> np.ndarray:
    table = np.zeros((len(classes), dim), dtype=np.float32)
    delta_path = os.path.join(runtime_dir, "model_delta.pt")
    if not os.path.isfile(delta_path):
        return table
    obj = _safe_load_object(delta_path)
    if not isinstance(obj, dict):
        return table
    protos = dict(obj.get("prototypes") or {})
    if not protos:
        return table
    class_to_idx = {str(name): i for i, name in enumerate(classes)}
    for name, vec in protos.items():
        idx = class_to_idx.get(str(name))
        if idx is None:
            continue
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            continue
        cols = min(int(arr.shape[0]), int(dim))
        if cols <= 0:
            continue
        row = np.zeros((dim,), dtype=np.float32)
        row[:cols] = arr[:cols]
        norm = float(np.linalg.norm(row))
        if norm > 0.0:
            row = row / norm
        table[int(idx)] = row.astype(np.float32)
    return table


def _load_delta_confusion_map(runtime_dir: str) -> Dict[str, Dict[str, float]]:
    delta_path = os.path.join(runtime_dir, "model_delta.pt")
    if not os.path.isfile(delta_path):
        return {}
    obj = _safe_load_object(delta_path)
    if not isinstance(obj, dict):
        return {}
    raw = dict(obj.get("confusion_deltas") or {})
    out: Dict[str, Dict[str, float]] = {}
    for pos, row in raw.items():
        if not isinstance(row, dict):
            continue
        clean_row: Dict[str, float] = {}
        for neg, val in row.items():
            try:
                score = float(val)
            except Exception:
                continue
            if abs(score) <= 1e-6:
                continue
            clean_row[str(neg)] = float(score)
        if clean_row:
            out[str(pos)] = clean_row
    return out


def _load_runtime_segment_payload(runtime_dir: str) -> List[Dict[str, Any]]:
    seg_path = os.path.join(runtime_dir, "segments.json")
    obj = _load_json(seg_path)
    rows = obj.get("segments") if isinstance(obj, dict) else []
    return [dict(x) for x in rows if isinstance(x, dict)]


def _load_runtime_base_label_scores(runtime_dir: str) -> np.ndarray:
    path = os.path.join(runtime_dir, "label_scores.npy")
    if not os.path.isfile(path):
        return np.zeros((0, 0), dtype=np.float32)
    try:
        arr = np.asarray(np.load(path), dtype=np.float32)
    except Exception:
        return np.zeros((0, 0), dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    row_sum = np.maximum(arr.sum(axis=1, keepdims=True), 1e-6)
    return (arr / row_sum).astype(np.float32, copy=False)


def _best_base_row_for_span(
    start: int,
    end: int,
    *,
    segments: Sequence[Dict[str, Any]],
    base_scores: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    k = int(len(classes))
    if base_scores.ndim != 2 or base_scores.shape[1] != k:
        return np.full((k,), 1.0 / max(k, 1), dtype=np.float32)
    best_idx = -1
    best_score = -1.0
    for idx, seg in enumerate(segments or []):
        if idx >= base_scores.shape[0]:
            break
        try:
            ss = int(seg.get("start", 0))
            ee = int(seg.get("end", ss))
        except Exception:
            continue
        overlap = min(int(end), int(ee)) - max(int(start), int(ss)) + 1
        if overlap <= 0:
            continue
        union = max(int(end), int(ee)) - min(int(start), int(ss)) + 1
        score = float(overlap) / float(max(1, union))
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx < 0:
        return np.full((k,), 1.0 / max(k, 1), dtype=np.float32)
    row = np.asarray(base_scores[int(best_idx)], dtype=np.float32).reshape(-1)
    if row.size != k:
        return np.full((k,), 1.0 / max(k, 1), dtype=np.float32)
    row_sum = float(np.sum(row))
    if row_sum <= 1e-6:
        return np.full((k,), 1.0 / max(k, 1), dtype=np.float32)
    return (row / row_sum).astype(np.float32, copy=False)


def _segment_rows_from_records(
    records: Sequence[Dict[str, Any]],
    classes: Sequence[str],
    confusion_map: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    finalized_video: bool = False,
) -> Tuple[
    List[Tuple[int, int, int, float]],
    List[Tuple[int, int, int, float]],
    List[Tuple[int, int, int, int, float]],
    List[Tuple[int, float]],
    List[Tuple[int, float]],
]:
    class_to_idx = {str(name): i for i, name in enumerate(classes)}
    positive_spans: List[Tuple[int, int, int, float]] = []
    negative_spans: List[Tuple[int, int, int, float]] = []
    ranking_pairs: List[Tuple[int, int, int, int, float]] = []
    pos_bounds: List[Tuple[int, float]] = []
    neg_bounds: List[Tuple[int, float]] = []
    for rec in records or []:
        point_type = str(rec.get("point_type", "") or "").strip().lower()
        action_kind = str(rec.get("action_kind", "") or "").strip().lower()
        confirmed_kind = _confirmed_kind(rec)
        rec_weight = float(_supervision_weight(rec, finalized_video=finalized_video))
        if point_type == "label":
            try:
                s = int(rec.get("feedback_start", 0))
                e = int(rec.get("feedback_end", s))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            label = str(rec.get("label", "") or "").strip()
            old_label = str(rec.get("old_label", "") or "").strip()
            pos_cid = int(class_to_idx[label]) if label and label in class_to_idx else -1
            neg_cids: List[int] = []
            if label and label in class_to_idx:
                positive_spans.append((s, e, pos_cid, rec_weight))
            if confirmed_kind != "accepted" and old_label and old_label in class_to_idx and action_kind in {"label_replace", "label_remove"}:
                neg_cids.append(int(class_to_idx[old_label]))
            if confirmed_kind != "accepted":
                for raw_name in rec.get("hard_negative_labels") or []:
                    name = str(raw_name or "").strip()
                    if not name or name not in class_to_idx:
                        continue
                    neg_cids.append(int(class_to_idx[name]))
                if label:
                    memory_row = dict((confusion_map or {}).get(label) or {})
                    if memory_row:
                        ordered = sorted(
                            (
                                (str(name).strip(), float(score))
                                for name, score in memory_row.items()
                                if str(name).strip() in class_to_idx and str(name).strip() != label
                            ),
                            key=lambda item: (-float(item[1]), str(item[0])),
                        )
                        for name, _score in ordered[:3]:
                            neg_cids.append(int(class_to_idx[str(name)]))
            for neg_cid in sorted(set(int(x) for x in neg_cids if int(x) >= 0)):
                negative_spans.append((s, e, int(neg_cid), rec_weight))
                if pos_cid >= 0 and int(neg_cid) != int(pos_cid):
                    ranking_pairs.append((s, e, int(pos_cid), int(neg_cid), rec_weight))
        elif point_type == "boundary":
            if action_kind in {"boundary_add", "boundary_move", "boundary_accept", "boundary_finalize"}:
                try:
                    pos_bounds.append((int(rec.get("boundary_frame")), rec_weight))
                except Exception:
                    pass
            if action_kind in {"boundary_remove", "boundary_move"}:
                try:
                    neg_bounds.append(
                        (
                            int(rec.get("old_boundary_frame", rec.get("boundary_frame"))),
                            rec_weight,
                        )
                    )
                except Exception:
                    pass
    return positive_spans, negative_spans, ranking_pairs, pos_bounds, neg_bounds


def rebuild_runtime_online_adapter(
    features_dir: str,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    def _emit(msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(str(msg))
            except Exception:
                pass

    if torch is None or F is None:
        return {"ok": False, "changed": False, "error": "PyTorch is not available."}
    features_dir = os.path.abspath(os.path.expanduser(str(features_dir or "")))
    runtime_dir = os.path.join(features_dir, "east_runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    adapter_path = os.path.join(runtime_dir, "online_adapter.pt")
    records, supervision_source = _load_supervision_records(runtime_dir)
    finalized_video = supervision_source == "finalized"
    record_hash = _stable_records_hash(records)
    classes = _runtime_classes(runtime_dir, records)
    if not classes:
        if os.path.isfile(adapter_path):
            try:
                os.remove(adapter_path)
            except Exception:
                pass
        _mark_runtime_dirty(
            runtime_dir,
            record_hash=record_hash,
            label_hash=record_hash,
            update_step=0,
            record_count=len(records),
            label_record_count=0,
        )
        return {
            "ok": True,
            "changed": True,
            "record_count": int(len(records)),
            "class_count": 0,
            "reason": "cleared_no_classes",
        }

    prev_obj = _safe_load_object(adapter_path) if os.path.isfile(adapter_path) else {}
    if not isinstance(prev_obj, dict):
        prev_obj = {}
    prev_hash = str(prev_obj.get("last_record_hash", "") or "")
    try:
        prev_step = int(prev_obj.get("step", 0))
    except Exception:
        prev_step = 0
    if os.path.isfile(adapter_path) and prev_hash == record_hash:
        return {
            "ok": True,
            "changed": False,
            "record_count": int(len(records)),
            "class_count": int(len(classes)),
            "step": int(prev_step),
            "reason": "unchanged",
        }

    feature_table, _, meta = _load_feature_series(features_dir)
    feature_dim = int(feature_table.shape[1])
    source_name = OnlineInteractiveAdapter.normalize_source_name(_source_name_from_meta(meta))
    delta_confusion = _load_delta_confusion_map(runtime_dir)
    pos_spans, neg_spans, ranking_pairs, pos_bounds, neg_bounds = _segment_rows_from_records(
        records,
        classes,
        confusion_map=delta_confusion,
        finalized_video=finalized_video,
    )
    confusion_memory_pair_count = int(
        sum(len(row) for row in dict(delta_confusion or {}).values() if isinstance(row, dict))
    )
    usable = bool(pos_spans or pos_bounds or neg_bounds)
    if not usable:
        if os.path.isfile(adapter_path):
            try:
                os.remove(adapter_path)
            except Exception:
                pass
        _mark_runtime_dirty(
            runtime_dir,
            record_hash=record_hash,
            label_hash=record_hash,
            update_step=0,
            record_count=len(records),
            label_record_count=0,
        )
        return {
            "ok": True,
            "changed": True,
            "record_count": int(len(records)),
            "class_count": int(len(classes)),
            "reason": "cleared_no_supervision",
        }

    _emit("[EAST][ADAPTER] Training lightweight online adapter...")
    model = OnlineInteractiveAdapter(input_dim=feature_dim, hidden_ratio=0.5)
    if isinstance(prev_obj.get("model"), dict) and int(prev_obj.get("input_dim", feature_dim)) == feature_dim:
        try:
            model.load_state_dict(prev_obj.get("model") or {}, strict=False)
        except Exception:
            pass
    model.train()

    runtime_proto = _load_runtime_prototype_table(runtime_dir, classes, feature_dim)
    delta_proto = _load_delta_prototype_table(runtime_dir, classes, feature_dim)
    text_bank = ensure_label_text_bank(
        features_dir,
        classes,
        feature_dim,
        progress_cb=progress_cb,
    )
    text_bank_table = np.asarray(text_bank.get("text_table"), dtype=np.float32) if bool(text_bank.get("ok", False)) else np.zeros((len(classes), feature_dim), dtype=np.float32)
    init_table = text_bank_table.copy() if text_bank_table.shape == (len(classes), feature_dim) else np.zeros((len(classes), feature_dim), dtype=np.float32)
    runtime_mask = np.linalg.norm(runtime_proto, axis=1) > 0.0 if runtime_proto.size else np.zeros((len(classes),), dtype=bool)
    if np.any(runtime_mask):
        init_table[runtime_mask] = runtime_proto[runtime_mask]
    if isinstance(prev_obj.get("text_table"), (list, np.ndarray)):
        try:
            prev_table = np.asarray(prev_obj.get("text_table"), dtype=np.float32)
            if prev_table.ndim == 2 and prev_table.shape[0] == len(classes) and prev_table.shape[1] == feature_dim:
                prev_mask = np.linalg.norm(prev_table, axis=1) > 0.0
                init_table[prev_mask] = prev_table[prev_mask]
        except Exception:
            pass
    if delta_proto.size:
        delta_mask = np.linalg.norm(delta_proto, axis=1) > 0.0
        if np.any(delta_mask):
            init_table[delta_mask] = delta_proto[delta_mask]
    class_table = torch.nn.Parameter(torch.from_numpy(init_table.astype(np.float32)))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [class_table],
        lr=1e-3,
        weight_decay=1e-4,
    )

    x = torch.from_numpy(np.array(feature_table, dtype=np.float32, copy=True))
    t_len = int(x.shape[0])
    cp_target = torch.full((t_len,), -1.0, dtype=torch.float32)
    cp_weight = torch.zeros((t_len,), dtype=torch.float32)
    for arr, value in ((pos_bounds, 1.0), (neg_bounds, 0.0)):
        for raw, weight in arr:
            frame = max(0, min(t_len - 1, int(raw)))
            cp_target[frame] = float(value)
            cp_weight[frame] = max(float(cp_weight[frame].item()), float(weight))
    for raw, weight in pos_bounds:
        frame = max(0, min(t_len - 1, int(raw)))
        for nb, val in ((frame - 1, 0.5), (frame + 1, 0.5)):
            if 0 <= nb < t_len and float(cp_target[nb]) < 0.0:
                cp_target[nb] = float(val)
                cp_weight[nb] = max(float(cp_weight[nb].item()), float(weight) * 0.5)

    def _span_embed(feat_2d: torch.Tensor, start: int, end: int) -> torch.Tensor:
        s = max(0, min(int(start), feat_2d.shape[0] - 1))
        e = max(s, min(int(end), feat_2d.shape[0] - 1))
        seg = feat_2d[s : e + 1]
        return F.normalize(seg.mean(dim=0), dim=0)

    steps = 60 if len(pos_spans) + len(neg_spans) + int(cp_target.ge(0).sum().item()) > 3 else 30
    steps = max(20, min(120, int(steps)))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(x, source_name=source_name)
        adapted = out["features"][0]
        cp_logits = out["cp_logits"][0]
        losses = []

        cp_mask = cp_target.ge(0.0)
        if bool(cp_mask.any()):
            losses.append(
                F.binary_cross_entropy_with_logits(
                    cp_logits[cp_mask],
                    cp_target[cp_mask],
                    weight=torch.clamp(cp_weight[cp_mask], min=0.05),
                )
            )

        norm_table = F.normalize(class_table, dim=1)
        pos_losses = []
        for s, e, cid, rec_weight in pos_spans:
            emb = _span_embed(adapted, s, e)
            logits = emb @ norm_table.t()
            pos_losses.append(
                F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([cid], dtype=torch.long),
                )
                * float(rec_weight)
            )
        if pos_losses:
            losses.append(torch.stack(pos_losses).sum() / max(sum(float(w) for *_rest, w in pos_spans), 1e-6))

        rank_losses = []
        for s, e, pos_cid, neg_cid, rec_weight in ranking_pairs:
            emb = _span_embed(adapted, s, e)
            logits = emb @ norm_table.t()
            margin = 0.2
            rank_losses.append(
                F.relu(torch.tensor(margin, dtype=logits.dtype) - logits[int(pos_cid)] + logits[int(neg_cid)])
                * float(rec_weight)
            )
        if rank_losses:
            losses.append(
                0.75
                * (torch.stack(rank_losses).sum() / max(sum(float(w) for *_rest, w in ranking_pairs), 1e-6))
            )

        neg_losses = []
        for s, e, cid, rec_weight in neg_spans:
            emb = _span_embed(adapted, s, e)
            logits = emb @ norm_table.t()
            neg_logit = logits[int(cid)]
            other_max = torch.max(torch.cat([logits[:cid], logits[cid + 1 :]]) if logits.numel() > 1 else torch.zeros_like(logits[:1]))
            neg_losses.append(F.softplus(neg_logit - other_max + 0.1) * float(rec_weight))
        if neg_losses:
            losses.append(
                0.5
                * (torch.stack(neg_losses).sum() / max(sum(float(w) for *_rest, w in neg_spans), 1e-6))
            )

        if text_bank_table.shape == (len(classes), feature_dim):
            target_table = torch.from_numpy(text_bank_table.astype(np.float32))
            losses.append(0.08 * F.mse_loss(norm_table, target_table))

        losses.append(0.05 * F.mse_loss(adapted, F.normalize(x, dim=1)))
        total = torch.stack(losses).sum()
        total.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x, source_name=source_name)
        adapted = out["features"][0].detach().cpu().numpy().astype(np.float32)
        cp_prob = torch.sigmoid(out["cp_logits"][0]).detach().cpu().numpy().astype(np.float32)
        text_table = F.normalize(class_table.detach(), dim=1).cpu().numpy().astype(np.float32)

    gate_weight = np.zeros((len(SOURCE_NAMES), 7), dtype=np.float32)
    gate_bias = np.log(np.asarray([0.45, 0.2, 0.2, 0.15], dtype=np.float32))
    runtime_segments = _load_runtime_segment_payload(runtime_dir)
    runtime_base_scores = _load_runtime_base_label_scores(runtime_dir)
    proto_gate_table = delta_proto.copy() if delta_proto.size else np.zeros((len(classes), feature_dim), dtype=np.float32)
    if proto_gate_table.shape != runtime_proto.shape:
        proto_gate_table = np.zeros((len(classes), feature_dim), dtype=np.float32)
    for idx in range(len(classes)):
        if idx >= proto_gate_table.shape[0]:
            break
        if float(np.linalg.norm(proto_gate_table[idx])) <= 1e-6 and idx < runtime_proto.shape[0]:
            proto_gate_table[idx] = runtime_proto[idx]
    text_prior_table = (
        normalize_rows(text_bank_table)
        if text_bank_table.shape == (len(classes), feature_dim)
        else np.zeros((len(classes), feature_dim), dtype=np.float32)
    )
    class_gate_table = normalize_rows(text_table) if text_table.shape == (len(classes), feature_dim) else np.zeros((len(classes), feature_dim), dtype=np.float32)
    proto_gate_table = normalize_rows(proto_gate_table) if proto_gate_table.shape == (len(classes), feature_dim) else np.zeros((len(classes), feature_dim), dtype=np.float32)
    text_available = bool(np.any(np.linalg.norm(text_prior_table, axis=1) > 0.0)) if text_prior_table.size else False
    class_available = bool(np.any(np.linalg.norm(class_gate_table, axis=1) > 0.0)) if class_gate_table.size else False
    proto_available = bool(np.any(np.linalg.norm(proto_gate_table, axis=1) > 0.0)) if proto_gate_table.size else False
    gate_targets: List[int] = []
    gate_lengths: List[int] = []
    gate_rows: Dict[str, List[np.ndarray]] = {name: [] for name in SOURCE_NAMES}
    gate_span_index: Dict[Tuple[int, int, int], int] = {}
    adapted_norm = normalize_rows(adapted)
    for row_idx, (s, e, cid, _weight) in enumerate(pos_spans):
        s = max(0, min(int(s), adapted_norm.shape[0] - 1))
        e = max(s, min(int(e), adapted_norm.shape[0] - 1))
        emb = np.asarray(np.mean(adapted_norm[s : e + 1], axis=0), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm > 0.0:
            emb = emb / norm
        base_row = _best_base_row_for_span(
            s,
            e,
            segments=runtime_segments,
            base_scores=runtime_base_scores,
            classes=classes,
        )
        text_row = (
            softmax_rows(np.expand_dims(emb @ text_prior_table.T, axis=0))[0]
            if text_available
            else np.full((len(classes),), 1.0 / max(len(classes), 1), dtype=np.float32)
        )
        class_row = (
            softmax_rows(np.expand_dims(emb @ class_gate_table.T, axis=0))[0]
            if class_available
            else np.full((len(classes),), 1.0 / max(len(classes), 1), dtype=np.float32)
        )
        proto_row = (
            softmax_rows(np.expand_dims(emb @ proto_gate_table.T, axis=0))[0]
            if proto_available
            else np.full((len(classes),), 1.0 / max(len(classes), 1), dtype=np.float32)
        )
        gate_rows["base"].append(np.asarray(base_row, dtype=np.float32))
        gate_rows["text_prior"].append(np.asarray(text_row, dtype=np.float32))
        gate_rows["class_table"].append(np.asarray(class_row, dtype=np.float32))
        gate_rows["prototype"].append(np.asarray(proto_row, dtype=np.float32))
        gate_targets.append(int(cid))
        gate_lengths.append(int(e - s + 1))
        gate_span_index[(int(s), int(e), int(cid))] = int(row_idx)

    if gate_targets:
        gate_feat_np = build_gate_features(
            np.asarray(gate_rows["base"], dtype=np.float32),
            gate_lengths,
            text_available=text_available,
            class_available=class_available,
            proto_available=proto_available,
        )
        gate_sources_np = np.stack(
            [np.asarray(gate_rows[name], dtype=np.float32) for name in SOURCE_NAMES],
            axis=1,
        )
        gate_targets_t = torch.tensor(gate_targets, dtype=torch.long)
        gate_feat_t = torch.from_numpy(gate_feat_np.astype(np.float32))
        gate_sources_t = torch.from_numpy(gate_sources_np.astype(np.float32))
        gate_w = torch.nn.Parameter(torch.zeros((len(SOURCE_NAMES), gate_feat_np.shape[1]), dtype=torch.float32))
        gate_b = torch.nn.Parameter(torch.from_numpy(gate_bias.astype(np.float32)))
        gate_opt = torch.optim.AdamW([gate_w, gate_b], lr=2e-3, weight_decay=1e-4)
        gate_steps = max(20, min(80, int(20 + len(gate_targets) * 2)))
        for _ in range(gate_steps):
            gate_opt.zero_grad(set_to_none=True)
            mix = torch.softmax(gate_feat_t @ gate_w.t() + gate_b.unsqueeze(0), dim=1)
            fused = torch.sum(mix.unsqueeze(-1) * gate_sources_t, dim=1)
            fused = fused / torch.clamp(fused.sum(dim=1, keepdim=True), min=1e-6)
            loss = F.nll_loss(torch.log(torch.clamp(fused, min=1e-6)), gate_targets_t)
            rank_terms = []
            for s, e, pos_cid, neg_cid, rec_weight in ranking_pairs:
                row_idx = gate_span_index.get((int(s), int(e), int(pos_cid)))
                if row_idx is None:
                    continue
                row = fused[int(row_idx)]
                margin = 0.12
                rank_terms.append(
                    F.relu(
                        torch.tensor(margin, dtype=row.dtype)
                        - torch.log(torch.clamp(row[int(pos_cid)], min=1e-6))
                        + torch.log(torch.clamp(row[int(neg_cid)], min=1e-6))
                    )
                    * float(rec_weight)
                )
            if rank_terms:
                loss = loss + 0.25 * (
                    torch.stack(rank_terms).sum()
                    / max(sum(float(w) for *_rest, w in ranking_pairs), 1e-6)
                )
            loss.backward()
            gate_opt.step()
        gate_weight = gate_w.detach().cpu().numpy().astype(np.float32)
        gate_bias = gate_b.detach().cpu().numpy().astype(np.float32)

    step = int(prev_step + 1)
    ckpt = {
        "version": 1,
        "step": int(step),
        "input_dim": int(feature_dim),
        "hidden_ratio": 0.5,
        "feature_source": str(source_name),
        "supervision_source": str(supervision_source),
        "video_finalized": bool(finalized_video),
        "classes": list(classes),
        "text_table": text_table,
        "text_bank_backend": str(text_bank.get("backend", "") or ""),
        "fusion_gate_weight": gate_weight,
        "fusion_gate_bias": gate_bias,
        "fusion_gate_sources": list(SOURCE_NAMES),
        "fusion_gate_feature_names": [
            "base_confidence",
            "base_margin",
            "base_entropy",
            "segment_length_norm",
            "text_available",
            "class_available",
            "proto_available",
        ],
        "model": model.state_dict(),
        "boundary_supervision_count": int(cp_target.ge(0.0).sum().item()),
        "boundary_supervision_weight": float(cp_weight[cp_target.ge(0.0)].sum().item()) if bool(cp_target.ge(0.0).any()) else 0.0,
        "positive_span_count": int(len(pos_spans)),
        "positive_span_weight": float(sum(float(w) for *_rest, w in pos_spans)),
        "negative_span_count": int(len(neg_spans)),
        "negative_span_weight": float(sum(float(w) for *_rest, w in neg_spans)),
        "ranking_pair_count": int(len(ranking_pairs)),
        "ranking_pair_weight": float(sum(float(w) for *_rest, w in ranking_pairs)),
        "confusion_memory_pair_count": int(confusion_memory_pair_count),
        "fusion_gate_trained": bool(len(gate_targets) > 0),
        "last_record_hash": str(record_hash),
        "updated_at": str(meta.get("updated_at") or ""),
        "boundary_prob_preview": cp_prob[: min(8, len(cp_prob))].astype(np.float32),
        "adapted_feature_preview": adapted[:1].astype(np.float32),
    }
    _safe_save_object(adapter_path, ckpt)
    _mark_runtime_dirty(
        runtime_dir,
        record_hash=record_hash,
        label_hash=record_hash,
        update_step=step,
        record_count=len(records),
        label_record_count=len(pos_spans),
    )
    _emit(
        f"[EAST][ADAPTER] Online adapter ready: {len(pos_spans)} positive spans, "
        f"{len(neg_spans)} negative spans, {int(cp_target.ge(0.0).sum().item())} boundary targets."
    )
    return {
        "ok": True,
        "changed": True,
        "record_count": int(len(records)),
        "supervision_source": str(supervision_source),
        "video_finalized": bool(finalized_video),
        "class_count": int(len(classes)),
        "positive_span_count": int(len(pos_spans)),
        "positive_span_weight": float(sum(float(w) for *_rest, w in pos_spans)),
        "negative_span_count": int(len(neg_spans)),
        "negative_span_weight": float(sum(float(w) for *_rest, w in neg_spans)),
        "ranking_pair_count": int(len(ranking_pairs)),
        "ranking_pair_weight": float(sum(float(w) for *_rest, w in ranking_pairs)),
        "confusion_memory_pair_count": int(confusion_memory_pair_count),
        "fusion_gate_trained": bool(len(gate_targets) > 0),
        "boundary_target_count": int(cp_target.ge(0.0).sum().item()),
        "boundary_target_weight": float(cp_weight[cp_target.ge(0.0)].sum().item()) if bool(cp_target.ge(0.0).any()) else 0.0,
        "step": int(step),
    }


def train_shared_online_adapter_from_runtime_dirs(
    runtime_dirs: Sequence[str],
    *,
    initial_online: Optional[Dict[str, Any]] = None,
    label_bank: Optional[Sequence[str]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    def _emit(msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(str(msg))
            except Exception:
                pass

    if torch is None or F is None:
        return {"ok": False, "changed": False, "error": "PyTorch is not available."}

    payloads: List[Dict[str, Any]] = []
    classes: List[str] = [str(x).strip() for x in (label_bank or []) if str(x).strip()]
    feature_dim = 0
    for runtime_dir in runtime_dirs or []:
        runtime_dir = os.path.abspath(os.path.expanduser(str(runtime_dir or "").strip()))
        if not runtime_dir or not os.path.isdir(runtime_dir):
            continue
        records, supervision_source = _load_supervision_records(runtime_dir)
        if supervision_source != "finalized" or not records:
            continue
        features_dir = os.path.dirname(runtime_dir)
        try:
            feature_table, _frame_map, meta = _load_feature_series(features_dir)
        except Exception:
            continue
        if feature_table.ndim != 2 or feature_table.shape[0] <= 0:
            continue
        cur_dim = int(feature_table.shape[1])
        if feature_dim <= 0:
            feature_dim = cur_dim
        if cur_dim != feature_dim:
            continue
        runtime_classes = _runtime_classes(runtime_dir, records)
        for name in runtime_classes:
            if name and name not in classes:
                classes.append(name)
        payloads.append(
            {
                "runtime_dir": runtime_dir,
                "features_dir": features_dir,
                "feature_table": np.asarray(feature_table, dtype=np.float32),
                "meta": dict(meta or {}),
                "records": list(records),
            }
        )
    if not payloads or feature_dim <= 0 or not classes:
        return {"ok": False, "changed": False, "reason": "no_finalized_runtimes"}

    enriched: List[Dict[str, Any]] = []
    text_targets: List[np.ndarray] = []
    for item in payloads:
        runtime_dir = str(item["runtime_dir"])
        records = list(item["records"])
        delta_confusion = _load_delta_confusion_map(runtime_dir)
        pos_spans, neg_spans, ranking_pairs, pos_bounds, neg_bounds = _segment_rows_from_records(
            records,
            classes,
            confusion_map=delta_confusion,
            finalized_video=True,
        )
        if not (pos_spans or pos_bounds or neg_bounds):
            continue
        features_dir = str(item["features_dir"])
        text_bank = ensure_label_text_bank(
            features_dir,
            classes,
            feature_dim,
            progress_cb=progress_cb,
        )
        text_bank_table = (
            np.asarray(text_bank.get("text_table"), dtype=np.float32)
            if bool(text_bank.get("ok", False))
            else np.zeros((len(classes), feature_dim), dtype=np.float32)
        )
        runtime_proto = _load_runtime_prototype_table(runtime_dir, classes, feature_dim)
        delta_proto = _load_delta_prototype_table(runtime_dir, classes, feature_dim)
        text_targets.append(text_bank_table)
        enriched.append(
            {
                **item,
                "pos_spans": pos_spans,
                "neg_spans": neg_spans,
                "ranking_pairs": ranking_pairs,
                "pos_bounds": pos_bounds,
                "neg_bounds": neg_bounds,
                "runtime_proto": runtime_proto,
                "delta_proto": delta_proto,
                "runtime_segments": _load_runtime_segment_payload(runtime_dir),
                "runtime_base_scores": _load_runtime_base_label_scores(runtime_dir),
                "source_name": OnlineInteractiveAdapter.normalize_source_name(
                    _source_name_from_meta(dict(item.get("meta") or {}))
                ),
            }
        )
    if not enriched:
        return {"ok": False, "changed": False, "reason": "no_finalized_supervision"}

    _emit(
        f"[EAST][OFFLINE] Training shared adapter from {len(enriched)} finalized runtime(s)..."
    )
    model = OnlineInteractiveAdapter(input_dim=feature_dim, hidden_ratio=0.5)
    init_obj = initial_online if isinstance(initial_online, dict) else {}
    if isinstance(init_obj.get("model"), dict) and int(init_obj.get("input_dim", feature_dim)) == feature_dim:
        try:
            model.load_state_dict(init_obj.get("model") or {}, strict=False)
        except Exception:
            pass
    model.train()

    init_table = np.zeros((len(classes), feature_dim), dtype=np.float32)
    if isinstance(init_obj.get("text_table"), (list, np.ndarray)):
        try:
            prev_table = np.asarray(init_obj.get("text_table"), dtype=np.float32)
            prev_classes = [str(x).strip() for x in (init_obj.get("classes") or []) if str(x).strip()]
            if prev_table.ndim == 2 and prev_table.shape[1] == feature_dim:
                prev_map = {name: prev_table[idx] for idx, name in enumerate(prev_classes) if idx < prev_table.shape[0]}
                for idx, name in enumerate(classes):
                    if name in prev_map:
                        init_table[idx] = np.asarray(prev_map[name], dtype=np.float32).reshape(-1)
        except Exception:
            pass
    if text_targets:
        acc = np.zeros_like(init_table)
        used = np.zeros((len(classes), 1), dtype=np.float32)
        for table in text_targets:
            if table.shape != init_table.shape:
                continue
            mask = (np.linalg.norm(table, axis=1, keepdims=True) > 0.0).astype(np.float32)
            acc += table * mask
            used += mask
        used = np.maximum(used, 1e-6)
        avg_table = acc / used
        avg_mask = np.linalg.norm(avg_table, axis=1) > 0.0
        init_table[avg_mask] = avg_table[avg_mask]
    class_table = torch.nn.Parameter(torch.from_numpy(init_table.astype(np.float32)))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [class_table],
        lr=8e-4,
        weight_decay=1e-4,
    )
    steps = max(30, min(140, 20 + 8 * len(enriched)))
    text_target_global = np.zeros((len(classes), feature_dim), dtype=np.float32)
    if text_targets:
        acc = np.zeros_like(text_target_global)
        used = np.zeros((len(classes), 1), dtype=np.float32)
        for table in text_targets:
            if table.shape != text_target_global.shape:
                continue
            mask = (np.linalg.norm(table, axis=1, keepdims=True) > 0.0).astype(np.float32)
            acc += table * mask
            used += mask
        used = np.maximum(used, 1e-6)
        text_target_global = acc / used

    def _span_embed(feat_2d: torch.Tensor, start: int, end: int) -> torch.Tensor:
        s = max(0, min(int(start), feat_2d.shape[0] - 1))
        e = max(s, min(int(end), feat_2d.shape[0] - 1))
        seg = feat_2d[s : e + 1]
        return F.normalize(seg.mean(dim=0), dim=0)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        norm_table = F.normalize(class_table, dim=1)
        for item in enriched:
            x = torch.from_numpy(np.array(item["feature_table"], dtype=np.float32, copy=True))
            out = model(x, source_name=str(item.get("source_name") or "unknown"))
            adapted = out["features"][0]
            cp_logits = out["cp_logits"][0]
            t_len = int(adapted.shape[0])
            cp_target = torch.full((t_len,), -1.0, dtype=torch.float32)
            cp_weight = torch.zeros((t_len,), dtype=torch.float32)
            for arr, value in ((item["pos_bounds"], 1.0), (item["neg_bounds"], 0.0)):
                for raw, weight in arr:
                    frame = max(0, min(t_len - 1, int(raw)))
                    cp_target[frame] = float(value)
                    cp_weight[frame] = max(float(cp_weight[frame].item()), float(weight))
            cp_mask = cp_target.ge(0.0)
            if bool(cp_mask.any()):
                losses.append(
                    F.binary_cross_entropy_with_logits(
                        cp_logits[cp_mask],
                        cp_target[cp_mask],
                        weight=torch.clamp(cp_weight[cp_mask], min=0.05),
                    )
                )
            pos_losses = []
            for s, e, cid, rec_weight in item["pos_spans"]:
                emb = _span_embed(adapted, s, e)
                logits = emb @ norm_table.t()
                pos_losses.append(
                    F.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([cid], dtype=torch.long),
                    )
                    * float(rec_weight)
                )
            if pos_losses:
                losses.append(
                    torch.stack(pos_losses).sum()
                    / max(sum(float(w) for *_rest, w in item["pos_spans"]), 1e-6)
                )
            rank_losses = []
            for s, e, pos_cid, neg_cid, rec_weight in item["ranking_pairs"]:
                emb = _span_embed(adapted, s, e)
                logits = emb @ norm_table.t()
                rank_losses.append(
                    F.relu(torch.tensor(0.2, dtype=logits.dtype) - logits[int(pos_cid)] + logits[int(neg_cid)])
                    * float(rec_weight)
                )
            if rank_losses:
                losses.append(
                    0.75
                    * (
                        torch.stack(rank_losses).sum()
                        / max(sum(float(w) for *_rest, w in item["ranking_pairs"]), 1e-6)
                    )
                )
            neg_losses = []
            for s, e, cid, rec_weight in item["neg_spans"]:
                emb = _span_embed(adapted, s, e)
                logits = emb @ norm_table.t()
                neg_logit = logits[int(cid)]
                other_max = torch.max(torch.cat([logits[:cid], logits[cid + 1 :]]) if logits.numel() > 1 else torch.zeros_like(logits[:1]))
                neg_losses.append(F.softplus(neg_logit - other_max + 0.1) * float(rec_weight))
            if neg_losses:
                losses.append(
                    0.5
                    * (
                        torch.stack(neg_losses).sum()
                        / max(sum(float(w) for *_rest, w in item["neg_spans"]), 1e-6)
                    )
                )
        if text_target_global.shape == (len(classes), feature_dim):
            losses.append(
                0.08
                * F.mse_loss(
                    norm_table,
                    torch.from_numpy(text_target_global.astype(np.float32)),
                )
            )
        if not losses:
            break
        total = torch.stack(losses).sum()
        total.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        text_table = F.normalize(class_table.detach(), dim=1).cpu().numpy().astype(np.float32)
    gate_weight = np.zeros((len(SOURCE_NAMES), 7), dtype=np.float32)
    gate_bias = np.log(np.asarray([0.45, 0.2, 0.2, 0.15], dtype=np.float32))
    text_prior_global = (
        normalize_rows(text_target_global)
        if text_target_global.shape == (len(classes), feature_dim)
        else np.zeros((len(classes), feature_dim), dtype=np.float32)
    )
    class_gate_table = (
        normalize_rows(text_table)
        if text_table.shape == (len(classes), feature_dim)
        else np.zeros((len(classes), feature_dim), dtype=np.float32)
    )
    text_available = bool(np.any(np.linalg.norm(text_prior_global, axis=1) > 0.0)) if text_prior_global.size else False
    class_available = bool(np.any(np.linalg.norm(class_gate_table, axis=1) > 0.0)) if class_gate_table.size else False
    gate_targets: List[int] = []
    gate_lengths: List[int] = []
    gate_rows: Dict[str, List[np.ndarray]] = {name: [] for name in SOURCE_NAMES}
    gate_span_index: Dict[Tuple[int, int, int, int], int] = {}
    gate_rank_refs: List[Tuple[int, int, float]] = []
    any_proto_available = False
    with torch.no_grad():
        for item_idx, item in enumerate(enriched):
            x = torch.from_numpy(np.array(item["feature_table"], dtype=np.float32, copy=True))
            adapted = model(x, source_name=str(item.get("source_name") or "unknown"))["features"][0]
            adapted_norm = normalize_rows(adapted.detach().cpu().numpy().astype(np.float32))
            proto_gate_table = np.asarray(item.get("delta_proto"), dtype=np.float32)
            runtime_proto = np.asarray(item.get("runtime_proto"), dtype=np.float32)
            if proto_gate_table.shape != runtime_proto.shape:
                proto_gate_table = np.zeros_like(runtime_proto)
            for idx in range(min(len(classes), proto_gate_table.shape[0])):
                if float(np.linalg.norm(proto_gate_table[idx])) <= 1e-6 and idx < runtime_proto.shape[0]:
                    proto_gate_table[idx] = runtime_proto[idx]
            proto_gate_table = (
                normalize_rows(proto_gate_table)
                if proto_gate_table.shape == (len(classes), feature_dim)
                else np.zeros((len(classes), feature_dim), dtype=np.float32)
            )
            proto_available = bool(np.any(np.linalg.norm(proto_gate_table, axis=1) > 0.0)) if proto_gate_table.size else False
            any_proto_available = bool(any_proto_available or proto_available)
            for row_idx, (s, e, cid, _weight) in enumerate(item["pos_spans"]):
                s = max(0, min(int(s), adapted_norm.shape[0] - 1))
                e = max(s, min(int(e), adapted_norm.shape[0] - 1))
                emb = np.asarray(np.mean(adapted_norm[s : e + 1], axis=0), dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(emb))
                if norm > 0.0:
                    emb = emb / norm
                base_row = _best_base_row_for_span(
                    s,
                    e,
                    segments=item.get("runtime_segments") or [],
                    base_scores=np.asarray(item.get("runtime_base_scores"), dtype=np.float32),
                    classes=classes,
                )
                text_row = (
                    softmax_rows(np.expand_dims(emb @ text_prior_global.T, axis=0))[0]
                    if text_available
                    else np.full((len(classes),), 1.0 / max(len(classes), 1), dtype=np.float32)
                )
                class_row = (
                    softmax_rows(np.expand_dims(emb @ class_gate_table.T, axis=0))[0]
                    if class_available
                    else np.full((len(classes),), 1.0 / max(len(classes), 1), dtype=np.float32)
                )
                proto_row = (
                    softmax_rows(np.expand_dims(emb @ proto_gate_table.T, axis=0))[0]
                    if proto_available
                    else np.full((len(classes),), 1.0 / max(len(classes), 1), dtype=np.float32)
                )
                gate_rows["base"].append(np.asarray(base_row, dtype=np.float32))
                gate_rows["text_prior"].append(np.asarray(text_row, dtype=np.float32))
                gate_rows["class_table"].append(np.asarray(class_row, dtype=np.float32))
                gate_rows["prototype"].append(np.asarray(proto_row, dtype=np.float32))
                gate_targets.append(int(cid))
                gate_lengths.append(int(e - s + 1))
                gate_span_index[(int(item_idx), int(s), int(e), int(cid))] = int(len(gate_targets) - 1)
            for s, e, pos_cid, neg_cid, rec_weight in item["ranking_pairs"]:
                row_idx = gate_span_index.get((int(item_idx), int(s), int(e), int(pos_cid)))
                if row_idx is None:
                    continue
                gate_rank_refs.append((int(row_idx), int(neg_cid), float(rec_weight)))
    if gate_targets:
        gate_feat_np = build_gate_features(
            np.asarray(gate_rows["base"], dtype=np.float32),
            gate_lengths,
            text_available=text_available,
            class_available=class_available,
            proto_available=bool(any_proto_available),
        )
        gate_sources_np = np.stack(
            [np.asarray(gate_rows[name], dtype=np.float32) for name in SOURCE_NAMES],
            axis=1,
        )
        gate_targets_t = torch.tensor(gate_targets, dtype=torch.long)
        gate_feat_t = torch.from_numpy(gate_feat_np.astype(np.float32))
        gate_sources_t = torch.from_numpy(gate_sources_np.astype(np.float32))
        gate_w = torch.nn.Parameter(torch.zeros((len(SOURCE_NAMES), gate_feat_np.shape[1]), dtype=torch.float32))
        gate_b = torch.nn.Parameter(torch.from_numpy(gate_bias.astype(np.float32)))
        gate_opt = torch.optim.AdamW([gate_w, gate_b], lr=2e-3, weight_decay=1e-4)
        gate_steps = max(24, min(96, int(24 + len(gate_targets) * 1.5)))
        for _ in range(gate_steps):
            gate_opt.zero_grad(set_to_none=True)
            mix = torch.softmax(gate_feat_t @ gate_w.t() + gate_b.unsqueeze(0), dim=1)
            fused = torch.sum(mix.unsqueeze(-1) * gate_sources_t, dim=1)
            fused = fused / torch.clamp(fused.sum(dim=1, keepdim=True), min=1e-6)
            loss = F.nll_loss(torch.log(torch.clamp(fused, min=1e-6)), gate_targets_t)
            if gate_rank_refs:
                rank_terms = []
                rank_weight_sum = 0.0
                for row_idx, neg_cid, rec_weight in gate_rank_refs:
                    row = fused[int(row_idx)]
                    pos_cid = int(gate_targets[row_idx])
                    margin = 0.12
                    rank_terms.append(
                        F.relu(
                            torch.tensor(margin, dtype=row.dtype)
                            - torch.log(torch.clamp(row[int(pos_cid)], min=1e-6))
                            + torch.log(torch.clamp(row[int(neg_cid)], min=1e-6))
                        )
                        * float(rec_weight)
                    )
                    rank_weight_sum += float(rec_weight)
                if rank_terms:
                    loss = loss + 0.25 * (torch.stack(rank_terms).sum() / max(rank_weight_sum, 1e-6))
            loss.backward()
            gate_opt.step()
        gate_weight = gate_w.detach().cpu().numpy().astype(np.float32)
        gate_bias = gate_b.detach().cpu().numpy().astype(np.float32)
    step = int(max(int((initial_online or {}).get("step", 0) or 0), 0) + 1)
    return {
        "ok": True,
        "changed": True,
        "version": 1,
        "step": int(step),
        "input_dim": int(feature_dim),
        "hidden_ratio": 0.5,
        "feature_source": "mixed_finalized_offline",
        "classes": list(classes),
        "text_table": text_table,
        "text_bank_backend": "mixed_finalized_offline",
        "fusion_gate_weight": gate_weight,
        "fusion_gate_bias": gate_bias,
        "fusion_gate_sources": list(SOURCE_NAMES),
        "fusion_gate_feature_names": [
            "base_confidence",
            "base_margin",
            "base_entropy",
            "segment_length_norm",
            "text_available",
            "class_available",
            "proto_available",
        ],
        "model": model.state_dict(),
        "boundary_supervision_count": int(
            sum(len(item["pos_bounds"]) + len(item["neg_bounds"]) for item in enriched)
        ),
        "positive_span_count": int(sum(len(item["pos_spans"]) for item in enriched)),
        "negative_span_count": int(sum(len(item["neg_spans"]) for item in enriched)),
        "ranking_pair_count": int(sum(len(item["ranking_pairs"]) for item in enriched)),
        "fusion_gate_trained": bool(len(gate_targets) > 0),
        "training_scope": "finalized_shared_offline",
        "member_count": int(len(enriched)),
        "updated_at": _load_json(os.path.join(enriched[0]["runtime_dir"], "meta.json")).get("updated_at", ""),
    }
