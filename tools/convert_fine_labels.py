"""Convert old fine-grained action segmentation annotations to the new schema.

Key rules (as requested by user):
1) Output segments in the new format:
   {
     "action_label": int,   # final_label_fine id (or 0 for null / to-be-relabeled)
     "verb": int,           # verb vocab id, -1 if blank
     "noun": int,           # noun vocab id, -1 if blank
     "start_frame": int,
     "end_frame": int,
     "phase": "normal" | "anomaly" | "recovery",
     "anomaly_type": [0/1 x6],  # ordered as:
         [error_temporal, error_spatial, error_handling, error_wrong_part, error_wrong_tool, error_procedural]
     "entity": "left" | "right" | ...
   }

2) Old error_* labels -> phase="anomaly" + anomaly_type vector; verb/noun blank (-1), action_label=0.
3) Old "recovery" label -> phase="recovery"; verb/noun blank (-1), action_label=0.
4) Old "transfer_tool" label is deprecated/removed in the new schema:
   keep the segment but mark it as to-be-re-labeled: action_label=0, verb/noun=-1, phase="normal".
5) Normal actions:
   map old action name to final action id (exact match, then explicit rename patch).
   verb/noun are inferred from the mapped final action name.
6) The script writes a mapping report JSON for auditing (unmapped actions, deprecated kept segments, etc.).

Examples:
python convert_fine_labels.py \
  --input 20250408_1119_color_left_clipped.json \
  --old_label_fine old_label_fine.txt \
  --final_label_fine final_fine_label.txt \
  --out_dir converted

python convert_fine_labels.py \
  --input path/to/old_json_folder \
  --old_label_fine old_label_fine.txt \
  --final_label_fine final_fine_label.txt \
  --out_dir converted
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


# -----------------------
# User-specified anomaly order
# -----------------------
ANOMALY_ORDER = [
    "error_temporal",
    "error_spatial",
    "error_handling",
    "error_wrong_part",
    "error_wrong_tool",
    "error_procedural",
]
ANOMALY_INDEX = {k: i for i, k in enumerate(ANOMALY_ORDER)}


# -----------------------
# I/O helpers
# -----------------------
def read_label_txt(path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Parse label file lines like:
      name 0
    Accepts extra spaces. Keeps first occurrence if duplicates exist.
    """
    id2name: Dict[int, str] = {}
    name2id: Dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        name = " ".join(parts[:-1]).strip()
        try:
            idx = int(parts[-1])
        except ValueError:
            continue
        id2name[idx] = name
        if name not in name2id:
            name2id[name] = idx
    return id2name, name2id


def gather_json_inputs(p: Path) -> List[Path]:
    if p.is_dir():
        return sorted([x for x in p.rglob("*.json") if x.is_file()])
    return [p]


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


# -----------------------
# Rename patch (explicit only; no guessing)
# -----------------------
def build_rename_patch() -> Dict[str, str]:
    """
    Explicit old->new name mappings.
    Only include mappings you are confident about.
    """
    return {
        # refine_* (old) -> adjust/align_* (new)
        "refine_drive_shaft": "adjust_drive_shaft",  # user confirmed duplicates mean same thing
        "refine_bearing_plate": "adjust_bearing_plate",
        "refine_lever": "align_lever",
        "refine_anti_vibration_handle": "adjust_anti_vibration_handle",
        "refine_nut": "align_nut",
        "refine_M6_nut": "align_M6_nut",
        "refine_screw": "align_screw",
        # old long name -> simplified new name (if it exists in final list)
        "attach_screw_to_phillips_screwdriver": "attach_screw",
    }


# -----------------------
# Old JSON parsing helpers
# -----------------------
def build_old_id2name_from_json(old_json: Dict[str, Any]) -> Dict[int, str]:
    """
    Old json usually has:
      "labels": [{"id":0,"name":"xxx"}, ...]
    """
    out: Dict[int, str] = {}
    for item in old_json.get("labels", []):
        try:
            i = int(item.get("id"))
        except Exception:
            continue
        n = item.get("name")
        if isinstance(n, str):
            out[i] = normalize_name(n)
    return out


def is_recovery_label(name: str) -> bool:
    return name == "recovery"


def parse_possible_multi_tags(name: str) -> List[str]:
    """
    Support future combined labels like:
      "error_temporal+error_spatial"
    or separated by commas / spaces.
    """
    tokens = re.split(r"[+,;/\s]+", name)
    return [t for t in tokens if t]


# -----------------------
# Mapping: old action name -> final action id
# -----------------------
def map_old_name_to_final_id(
    old_name: str,
    final_name2id: Dict[str, int],
    rename_patch: Dict[str, str],
) -> Optional[int]:
    old_name = normalize_name(old_name)

    # exact
    if old_name in final_name2id:
        return final_name2id[old_name]

    # patched rename
    if old_name in rename_patch:
        new_name = rename_patch[old_name]
        if new_name in final_name2id:
            return final_name2id[new_name]

    return None


# -----------------------
# Verb/Noun vocab inference (from final labels)
# -----------------------
KNOWN_VERB_PREFIXES = [
    # multi-token verbs first
    "pick_up",
    "hand_tighten",
    "hand_loosen",
    # single-token
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
]


def infer_verb_noun(action_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse action label name into (verb, noun) using KNOWN_VERB_PREFIXES.
    Example:
      "pick_up_drive_shaft" -> ("pick_up", "drive_shaft")
      "tighten_screw" -> ("tighten", "screw")
      "null" -> (None, None)
    """
    action_name = normalize_name(action_name)
    if action_name == "null":
        return None, None

    for v in sorted(KNOWN_VERB_PREFIXES, key=len, reverse=True):
        prefix = v + "_"
        if action_name.startswith(prefix):
            noun = action_name[len(prefix) :]
            return v, noun if noun else None

    # fallback: first token as verb if present
    if "_" in action_name:
        verb, noun = action_name.split("_", 1)
        return verb, noun if noun else None

    return action_name, None


def build_vocabs_from_final(
    final_id2name: Dict[int, str]
) -> Tuple[List[str], List[str], Dict[str, int], Dict[str, int]]:
    verb_set = set()
    noun_set = set()
    for i in sorted(final_id2name.keys()):
        name = final_id2name[i]
        if name == "null":
            continue
        v, n = infer_verb_noun(name)
        if v:
            verb_set.add(v)
        if n:
            noun_set.add(n)

    verbs = sorted(verb_set)
    nouns = sorted(noun_set)
    verb2id = {v: idx for idx, v in enumerate(verbs)}
    noun2id = {n: idx for idx, n in enumerate(nouns)}
    return verbs, nouns, verb2id, noun2id


# -----------------------
# Conversion core
# -----------------------
def convert_one_file(
    in_path: Path,
    out_dir: Path,
    final_id2name: Dict[int, str],
    final_name2id: Dict[str, int],
) -> Tuple[Path, Path]:
    old_json = json.loads(in_path.read_text(encoding="utf-8"))
    old_id2name = build_old_id2name_from_json(old_json)

    rename_patch = build_rename_patch()
    verbs, nouns, verb2id, noun2id = build_vocabs_from_final(final_id2name)

    report = {
        "file": in_path.name,
        "deprecated_segments": [],  # segments like transfer_tool kept as null for re-labeling
        "unmapped_action_names": {},  # name -> count (normal actions that couldn't map)
        "missing_old_label_id": {},  # old action_label id not found in old_json["labels"]
        "stats": {
            "num_segments_in": len(old_json.get("segments", [])),
            "num_segments_out": 0,
            "num_anomaly": 0,
            "num_recovery": 0,
            "num_deprecated": 0,
            "num_unmapped_normal": 0,
        },
    }

    segments_out = []

    for seg in old_json.get("segments", []):
        old_action_id = seg.get("action_label", None)
        start = int(seg.get("start_frame"))
        end = int(seg.get("end_frame"))
        entity = seg.get("entity")

        old_name = old_id2name.get(old_action_id, None)
        if old_name is None:
            # Unknown id in segments
            key = str(old_action_id)
            report["missing_old_label_id"][key] = (
                report["missing_old_label_id"].get(key, 0) + 1
            )
            old_name = "null"

        old_name = normalize_name(old_name)
        tokens = parse_possible_multi_tags(old_name)

        # ---------- anomaly ----------
        if any(t in ANOMALY_INDEX for t in tokens):
            phase = "anomaly"
            vec = [0, 0, 0, 0, 0, 0]
            for t in tokens:
                if t in ANOMALY_INDEX:
                    vec[ANOMALY_INDEX[t]] = 1

            segments_out.append(
                {
                    "action_label": 0,  # null
                    "verb": -1,  # left blank
                    "noun": -1,  # left blank
                    "start_frame": start,
                    "end_frame": end,
                    "phase": phase,
                    "anomaly_type": vec,
                    "entity": entity,
                }
            )
            report["stats"]["num_anomaly"] += 1
            continue

        # ---------- recovery ----------
        if any(t == "recovery" for t in tokens) or is_recovery_label(old_name):
            segments_out.append(
                {
                    "action_label": 0,  # null
                    "verb": -1,  # left blank
                    "noun": -1,  # left blank
                    "start_frame": start,
                    "end_frame": end,
                    "phase": "recovery",
                    "anomaly_type": [0, 0, 0, 0, 0, 0],
                    "entity": entity,
                }
            )
            report["stats"]["num_recovery"] += 1
            continue

        # ---------- deprecated: transfer_tool ----------
        if old_name == "transfer_tool":
            segments_out.append(
                {
                    "action_label": 0,  # keep as null, to be relabeled by you
                    "verb": -1,
                    "noun": -1,
                    "start_frame": start,
                    "end_frame": end,
                    "phase": "normal",
                    "anomaly_type": [0, 0, 0, 0, 0, 0],
                    "entity": entity,
                }
            )
            report["stats"]["num_deprecated"] += 1
            report["deprecated_segments"].append(
                {
                    "old_name": old_name,
                    "start_frame": start,
                    "end_frame": end,
                    "entity": entity,
                }
            )
            continue

        # ---------- normal action ----------
        final_id = map_old_name_to_final_id(old_name, final_name2id, rename_patch)
        if final_id is None:
            # Do not guess; mark as null and record for auditing
            report["unmapped_action_names"][old_name] = (
                report["unmapped_action_names"].get(old_name, 0) + 1
            )
            report["stats"]["num_unmapped_normal"] += 1
            final_id = 0

        # infer verb/noun from the final action name if possible (more consistent)
        parse_name = final_id2name.get(final_id, old_name)
        v_str, n_str = infer_verb_noun(parse_name)

        verb_id = verb2id.get(v_str, -1) if v_str else -1
        noun_id = noun2id.get(n_str, -1) if n_str else -1

        segments_out.append(
            {
                "action_label": int(final_id),
                "verb": int(verb_id),
                "noun": int(noun_id),
                "start_frame": start,
                "end_frame": end,
                "phase": "normal",
                "anomaly_type": [0, 0, 0, 0, 0, 0],
                "entity": entity,
            }
        )

    report["stats"]["num_segments_out"] = len(segments_out)

    # ---------- build output JSON with header vocabs ----------
    out = {
        "video_id": old_json.get("video_id", in_path.stem),
        "anomaly_types": [{"id": i, "name": n} for i, n in enumerate(ANOMALY_ORDER)],
        "verbs": [{"id": i, "name": v} for i, v in enumerate(verbs)],
        "nouns": [{"id": i, "name": n} for i, n in enumerate(nouns)],
        "action_labels": [
            {"id": i, "name": final_id2name[i]} for i in sorted(final_id2name.keys())
        ],
        "segments": segments_out,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = out_dir / f"{in_path.stem}_final_fine.json"
    out_report_path = out_dir / f"{in_path.stem}_mapping_report.json"

    out_json_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return out_json_path, out_report_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="Old annotation json file OR a directory containing jsons",
    )
    ap.add_argument(
        "--old_label_fine",
        type=str,
        required=True,
        help="old_label_fine.txt path (optional for reference)",
    )
    ap.add_argument(
        "--final_label_fine", type=str, required=True, help="final_fine_label.txt path"
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    args = ap.parse_args()

    in_paths = gather_json_inputs(Path(args.input))
    out_dir = Path(args.out_dir)

    # labels: final is required (for action id mapping)
    final_id2name, final_name2id = read_label_txt(Path(args.final_label_fine))

    if 0 not in final_id2name or final_id2name.get(0) != "null":
        print(
            "[WARN] final_label_fine does not contain 'null 0' (or it is not id=0). Proceeding anyway."
        )

    # old_label_fine is not required by conversion logic (we use old json labels),
    # but we keep it in args for your pipeline consistency.
    _ = args.old_label_fine

    for p in in_paths:
        out_json, out_report = convert_one_file(
            p, out_dir, final_id2name, final_name2id
        )
        print(f"[OK] {p.name} -> {out_json}")
        print(f"     report -> {out_report}")


if __name__ == "__main__":
    main()
