from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple


_LABEL_FILENAMES: Tuple[str, ...] = (
    "label.txt",
    "labels.txt",
    "classes.txt",
    "mapping.txt",
)

DEFAULT_VERB_PREFIXES: Tuple[str, ...] = (
    "pick_up",
    "put_down",
    "hand_tighten",
    "hand_loosen",
    "hand_spin",
    "screw_on",
    "tighten",
    "loosen",
    "remove",
    "insert",
    "mount",
    "attach",
    "detach",
    "dismount",
    "extract",
    "place",
    "store",
    "transfer",
    "hold",
    "flip",
    "thread",
    "adjust",
    "align",
    "seat",
    "retrieve",
    "install",
    "unscrew",
    "start",
    "finish",
)


def _looks_like_int(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    if text[0] in "+-":
        return text[1:].isdigit()
    return text.isdigit()


def _dedupe_keep_order(names: Iterable[str]) -> List[str]:
    out: List[str] = []
    for raw in names:
        name = str(raw or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def parse_label_line(line: str) -> Tuple[str, Optional[int]]:
    text = str(line or "").strip()
    if not text:
        return "", None
    parts = text.split()
    if len(parts) >= 2 and _looks_like_int(parts[-1]):
        name = " ".join(parts[:-1]).strip()
        return name, int(parts[-1])
    if len(parts) >= 2 and _looks_like_int(parts[0]):
        name = " ".join(parts[1:]).strip()
        return name, int(parts[0])
    return text, None


def load_label_names(path: str, num_classes: Optional[int] = None) -> List[str]:
    if not path or not os.path.isfile(path):
        names = []
    else:
        indexed: List[Tuple[int, int, str]] = []
        plain: List[Tuple[int, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for order, line in enumerate(f):
                name, idx = parse_label_line(line)
                if not name:
                    continue
                if idx is None:
                    plain.append((order, name))
                else:
                    indexed.append((int(idx), order, name))
        if indexed:
            indexed.sort(key=lambda item: (item[0], item[1]))
            names = [name for _, _, name in indexed]
            names.extend(name for _, name in plain if name not in names)
        else:
            names = [name for _, name in plain]
    names = _dedupe_keep_order(names)
    if num_classes is None:
        return names
    padded = list(names[: int(num_classes)])
    for idx in range(len(padded), int(num_classes)):
        padded.append(f"cls_{idx}")
    return padded[: int(num_classes)]


def infer_verb_noun(
    label_name: str,
    verb_candidates: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    name = str(label_name or "").strip().lower()
    name = name.replace(" ", "_")
    name = "_".join(part for part in name.split("_") if part)
    if not name or name == "null":
        return None, None
    verbs = [str(v or "").strip().lower() for v in (verb_candidates or DEFAULT_VERB_PREFIXES)]
    verbs = [v for v in verbs if v]
    for verb in sorted(set(verbs), key=len, reverse=True):
        prefix = verb + "_"
        if name.startswith(prefix):
            noun = name[len(prefix) :]
            return verb, (noun if noun else None)
    if "_" in name:
        verb, noun = name.split("_", 1)
        return verb, (noun if noun else None)
    return name, None


def candidate_label_paths(
    *,
    features_dir: str = "",
    video_path: str = "",
    repo_root: str = "",
    extra_dirs: Optional[Sequence[str]] = None,
    filenames: Optional[Sequence[str]] = None,
) -> List[str]:
    dirs: List[str] = []

    def _add_dir(path: str) -> None:
        path = str(path or "").strip()
        if not path:
            return
        abs_path = os.path.abspath(os.path.expanduser(path))
        if abs_path not in dirs:
            dirs.append(abs_path)

    for item in extra_dirs or []:
        extra_dir = str(item or "").strip()
        if not extra_dir:
            continue
        extra_dir = os.path.abspath(os.path.expanduser(extra_dir))
        _add_dir(extra_dir)
        _add_dir(os.path.dirname(extra_dir))

    feat_dir = str(features_dir or "").strip()
    if feat_dir:
        feat_dir = os.path.abspath(os.path.expanduser(feat_dir))
        _add_dir(feat_dir)
        _add_dir(os.path.dirname(feat_dir))
        _add_dir(os.path.dirname(os.path.dirname(feat_dir)))

    vid_path = str(video_path or "").strip()
    if vid_path:
        vid_dir = os.path.dirname(os.path.abspath(os.path.expanduser(vid_path)))
        _add_dir(vid_dir)
        _add_dir(os.path.dirname(vid_dir))

    if repo_root:
        _add_dir(repo_root)

    ordered_names = tuple(filenames or _LABEL_FILENAMES)
    out: List[str] = []
    for base_dir in dirs:
        for name in ordered_names:
            cand = os.path.abspath(os.path.join(base_dir, name))
            if cand not in out:
                out.append(cand)
    return out


def resolve_label_source(
    *,
    features_dir: str = "",
    video_path: str = "",
    repo_root: str = "",
    extra_dirs: Optional[Sequence[str]] = None,
    filenames: Optional[Sequence[str]] = None,
) -> str:
    for cand in candidate_label_paths(
        features_dir=features_dir,
        video_path=video_path,
        repo_root=repo_root,
        extra_dirs=extra_dirs,
        filenames=filenames,
    ):
        if os.path.isfile(cand):
            return cand
    return ""
