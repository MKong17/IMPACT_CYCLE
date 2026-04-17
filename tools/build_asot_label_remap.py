#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.asot_label_remap_utils import canonicalize_label_name
from tools.label_utils import load_label_names, resolve_label_source


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build an ASOT cluster-to-label remap JSON from GT-calibrated videos."
    )
    ap.add_argument(
        "roots",
        nargs="+",
        help="Dataset roots containing groundTruth/ and videos/.",
    )
    ap.add_argument(
        "--output-json",
        default="",
        help="Output remap JSON path. Defaults to a sidecar next to the label bank.",
    )
    ap.add_argument(
        "--pred-prefix",
        default="pred_asot",
        help="Prediction prefix to read (default: pred_asot).",
    )
    ap.add_argument(
        "--class-names",
        default="",
        help="Semantic label bank used by ASOT export. Auto-discovered when omitted.",
    )
    ap.add_argument(
        "--feature-search-root",
        action="append",
        default=[],
        help=(
            "Extra root to recursively search for <video_id>_features directories. "
            "Can be specified multiple times."
        ),
    )
    return ap.parse_args(argv)


def resolve_builder_label_source(
    *,
    explicit_path: str,
    roots: Sequence[Path],
    feature_search_roots: Sequence[str],
) -> Path:
    raw = str(explicit_path or "").strip()
    if raw:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Semantic label bank not found: {path}")
        return path

    extra_dirs = [str(root.resolve()) for root in roots]
    extra_dirs.extend(str(Path(item).expanduser().resolve()) for item in feature_search_roots)
    resolved = resolve_label_source(extra_dirs=extra_dirs, repo_root="")
    if resolved:
        return Path(resolved).resolve()
    raise SystemExit(
        "No semantic label bank found. Pass --class-names or place a label file next to the dataset/features."
    )


def resolve_output_json_path(
    raw_output_path: str,
    class_names_path: Path,
    roots: Sequence[Path],
) -> Path:
    raw = str(raw_output_path or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if class_names_path:
        return class_names_path.resolve().parent / "asot_label_remap.json"
    if roots:
        return roots[0].resolve() / "asot_label_remap.json"
    return Path("asot_label_remap.json").resolve()


def read_gt_sequence(path: Path, canonical_to_actual: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        key = canonicalize_label_name(line)
        if not key:
            continue
        actual = canonical_to_actual.get(key)
        if actual:
            out.append(actual)
        else:
            out.append(str(line).strip())
    return out


def align_length(seq: Sequence[int], target_len: int) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.int32).reshape(-1)
    if int(arr.shape[0]) == int(target_len):
        return arr.astype(np.int32, copy=False)
    if int(arr.shape[0]) > int(target_len):
        return arr[: int(target_len)].astype(np.int32, copy=False)
    pad_value = int(arr[-1]) if arr.size > 0 else 0
    if int(target_len) <= 0:
        return np.zeros((0,), dtype=np.int32)
    padded = np.full((int(target_len),), pad_value, dtype=np.int32)
    if arr.size > 0:
        padded[: int(arr.shape[0])] = arr
    return padded


def clusters_from_segments_json(path: Path, length_hint: int) -> Optional[np.ndarray]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    meta = obj.get("meta") or {}
    if bool(meta.get("label_remap_applied", False)):
        return None
    segs = obj.get("segments")
    if not isinstance(segs, list) or not segs:
        return None
    length = max(
        int(length_hint),
        max((int(seg.get("end_frame", -1)) for seg in segs), default=-1) + 1,
    )
    if length <= 0:
        return None
    out = np.zeros((int(length),), dtype=np.int32)
    for seg in segs:
        try:
            s = int(seg.get("start_frame", seg.get("start", 0)))
            e = int(seg.get("end_frame", seg.get("end", s)))
            cid = int(seg.get("cluster_id", seg.get("class_id", 0)))
        except Exception:
            continue
        s = max(0, min(int(length) - 1, s))
        e = max(s, min(int(length) - 1, e))
        out[s : e + 1] = int(cid)
    return out


def load_cluster_sequence(
    features_dir: Path,
    pred_prefix: str,
    length_hint: int,
) -> Optional[np.ndarray]:
    candidates = [
        features_dir / f"{pred_prefix}_cluster_per_frame.npy",
        features_dir / f"{pred_prefix}_per_frame.npy",
        features_dir / f"{pred_prefix}_segments.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix.lower() == ".npy":
            try:
                arr = np.asarray(np.load(path), dtype=np.int32).reshape(-1)
            except Exception:
                continue
            return align_length(arr, length_hint)
        arr = clusters_from_segments_json(path, length_hint=length_hint)
        if arr is not None:
            return align_length(arr, length_hint)
    return None


def _has_asot_prediction(features_dir: Path, pred_prefix: str) -> bool:
    candidates = [
        features_dir / f"{pred_prefix}_cluster_per_frame.npy",
        features_dir / f"{pred_prefix}_per_frame.npy",
        features_dir / f"{pred_prefix}_segments.json",
    ]
    return any(path.is_file() for path in candidates)


def build_feature_index(
    search_roots: Sequence[Path],
    pred_prefix: str,
) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    seen: set[str] = set()
    for raw_root in search_roots:
        root = Path(raw_root).resolve()
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_dir():
                continue
            name = path.name
            if name.endswith("_features"):
                video_id = name[: -len("_features")]
            else:
                continue
            if not video_id or (not _has_asot_prediction(path, pred_prefix)):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            index[video_id].append(path.resolve())
    return index


def iter_calibration_items(
    root: Path,
    pred_prefix: str,
    feature_index: Optional[Dict[str, List[Path]]] = None,
) -> Iterable[Tuple[str, Path, List[Path]]]:
    gt_dir = root / "groundTruth"
    video_dir = root / "videos"
    if not gt_dir.is_dir():
        return
    for gt_path in sorted(gt_dir.glob("*.txt")):
        video_id = gt_path.stem
        candidates: List[Path] = []
        if video_dir.is_dir():
            for features_dir in (video_dir / f"{video_id}_features", video_dir / video_id):
                if features_dir.exists():
                    candidates.append(features_dir.resolve())
        for features_dir in list((feature_index or {}).get(video_id, [])):
            resolved = features_dir.resolve()
            if resolved not in candidates:
                candidates.append(resolved)
        if candidates:
            yield video_id, gt_path, candidates


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    roots = [Path(item).resolve() for item in args.roots]
    class_names_path = resolve_builder_label_source(
        explicit_path=str(args.class_names or ""),
        roots=roots,
        feature_search_roots=list(args.feature_search_root or []),
    )
    output_json = resolve_output_json_path(
        str(args.output_json or ""),
        class_names_path,
        roots,
    )
    semantic_classes = load_label_names(str(class_names_path))
    if not semantic_classes:
        raise SystemExit(f"No semantic classes found in {class_names_path}")

    canonical_to_actual = {
        canonicalize_label_name(name): name
        for name in semantic_classes
        if canonicalize_label_name(name)
    }
    cluster_label_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    cluster_support: Counter[int] = Counter()
    used_videos: List[str] = []
    max_cluster = -1
    skipped: List[str] = []
    seen_video_ids: set[str] = set()
    search_roots = list(roots)
    search_roots.extend(Path(item).resolve() for item in list(args.feature_search_root or []))
    feature_index = build_feature_index(
        search_roots,
        pred_prefix=str(args.pred_prefix or "pred_asot"),
    )

    for root in roots:
        for video_id, gt_path, feature_candidates in iter_calibration_items(
            root,
            pred_prefix=str(args.pred_prefix or "pred_asot"),
            feature_index=feature_index,
        ):
            if video_id in seen_video_ids:
                continue
            gt_seq = read_gt_sequence(gt_path, canonical_to_actual)
            if not gt_seq:
                skipped.append(f"{video_id}: empty GT")
                continue
            cluster_seq = None
            for features_dir in feature_candidates:
                cluster_seq = load_cluster_sequence(
                    features_dir,
                    pred_prefix=str(args.pred_prefix or "pred_asot"),
                    length_hint=len(gt_seq),
                )
                if cluster_seq is not None and cluster_seq.size > 0:
                    break
            if cluster_seq is None or cluster_seq.size <= 0:
                skipped.append(f"{video_id}: missing ASOT cluster prediction")
                continue
            cluster_seq = align_length(cluster_seq, len(gt_seq))
            for cid, label in zip(cluster_seq.tolist(), gt_seq):
                cid_int = int(cid)
                cluster_label_counts[cid_int][label] += 1
                cluster_support[cid_int] += 1
                max_cluster = max(max_cluster, cid_int)
            used_videos.append(video_id)
            seen_video_ids.add(video_id)

    if not used_videos:
        raise SystemExit("No calibration videos with GT + ASOT prediction were found.")

    cluster_to_label: Dict[str, str] = {}
    cluster_purity: Dict[str, float] = {}
    cluster_counts_json: Dict[str, Dict[str, int]] = {}
    for cid in range(max(0, int(max_cluster)) + 1):
        counts = cluster_label_counts.get(cid, Counter())
        key = str(cid)
        if not counts:
            if cid < len(semantic_classes):
                cluster_to_label[key] = semantic_classes[cid]
            else:
                cluster_to_label[key] = semantic_classes[0]
            cluster_purity[key] = 0.0
            cluster_counts_json[key] = {}
            continue
        label, support = counts.most_common(1)[0]
        total = max(1, sum(int(v) for v in counts.values()))
        cluster_to_label[key] = str(label)
        cluster_purity[key] = float(support) / float(total)
        cluster_counts_json[key] = {
            str(name): int(val) for name, val in counts.most_common()
        }

    payload = {
        "version": 1,
        "source": "majority_vote_from_gt",
        "pred_prefix": str(args.pred_prefix or "pred_asot"),
        "class_names_path": str(class_names_path),
        "semantic_classes": list(semantic_classes),
        "cluster_to_label": cluster_to_label,
        "cluster_support": {str(k): int(v) for k, v in sorted(cluster_support.items())},
        "cluster_purity": cluster_purity,
        "cluster_counts": cluster_counts_json,
        "videos_used": sorted(used_videos),
        "roots": [str(root) for root in roots],
        "skipped": list(skipped),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Wrote ASOT label remap to {output_json}")
    print(f"[INFO] Videos used: {len(used_videos)}")
    for video_id in sorted(used_videos):
        print(f"  - {video_id}")
    if skipped:
        print("[INFO] Skipped:")
        for item in skipped:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
