# Canonical annotation schema + adapters for ActivityNet-like segments and per-frame TXT.
from typing import Dict, Any, List, Optional, Tuple

Canonical = Dict[str, Any]


def to_canonical_template(video_meta: Dict[str, Any]) -> Canonical:
    return {
        "version": "v1",
        "video": {
            "id": video_meta.get("id", "unknown"),
            "name": video_meta.get("name", "unknown.mp4"),
            "fps": float(video_meta.get("fps", 30.0)),
            "num_frames": int(video_meta.get("num_frames", 0)),
            "duration_sec": float(video_meta.get("duration_sec", 0.0)),
        },
        "categories": [],
        "annotations": [],
        "extras": {},
    }


class AdapterBase:
    name: str = "Base"

    def detect(self, payload: Dict[str, Any]) -> bool:
        return False

    def import_to_canonical(
        self, payload: Dict[str, Any], meta: Dict[str, Any]
    ) -> Canonical:
        raise NotImplementedError

    def export_from_canonical(
        self, canonical: Canonical, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError


class ActivityNetAdapter(AdapterBase):
    name = "ActivityNet"

    def detect(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        anns = payload.get("annotations")
        if not isinstance(anns, list) or not anns:
            return False
        sample = anns[0]
        return isinstance(sample, dict) and "segments" in sample

    def import_to_canonical(
        self, payload: Dict[str, Any], meta: Dict[str, Any]
    ) -> Canonical:
        fps = float(meta.get("fps", 30.0))
        can = to_canonical_template(meta)
        cats: Dict[str, int] = {}
        cid = 0
        for idx, ann in enumerate(payload.get("annotations", [])):
            label = str(ann.get("label", "unknown"))
            if label not in cats:
                cid += 1
                cats[label] = cid
                can["categories"].append({"id": cid, "name": label})
            for seg in ann.get("segments", []):
                s_sec, e_sec = float(seg[0]), float(seg[1])
                s_fr, e_fr = int(round(s_sec * fps)), int(round(e_sec * fps))
                can["annotations"].append(
                    {
                        "id": f"ann_{idx}_{s_fr}",
                        "category_id": cats[label],
                        "start": {"value": s_fr, "unit": "frame"},
                        "end": {"value": e_fr, "unit": "frame"},
                        "attributes": {"source": "activitynet"},
                    }
                )
        return can

    def export_from_canonical(
        self, canonical: Canonical, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        fps = float(canonical["video"]["fps"] or 30.0)
        id2name = {c["id"]: c["name"] for c in canonical.get("categories", [])}
        grouped: Dict[str, List[Tuple[float, float]]] = {}
        for ann in canonical.get("annotations", []):
            name = id2name.get(ann["category_id"], "unknown")
            s = ann["start"]["value"] / fps
            e = ann["end"]["value"] / fps
            grouped.setdefault(name, []).append([s, e])
        return {
            "annotations": [{"label": k, "segments": v} for k, v in grouped.items()]
        }


class FrameTXTAdapter(AdapterBase):
    """Payload is {"frame_labels": ["pour","pour","stir", ...]}"""

    name = "FrameTXT"

    def detect(self, payload: Dict[str, Any]) -> bool:
        return isinstance(payload, dict) and isinstance(
            payload.get("frame_labels"), list
        )

    def import_to_canonical(
        self, payload: Dict[str, Any], meta: Dict[str, Any]
    ) -> Canonical:
        labels: List[str] = payload.get("frame_labels", [])
        fps = float(meta.get("fps", 30.0))
        can = to_canonical_template(
            {**meta, "num_frames": len(labels), "duration_sec": len(labels) / fps}
        )
        cats: Dict[str, int] = {}
        cid = 0

        def cid_of(name: str) -> int:
            nonlocal cid
            if name not in cats:
                cid += 1
                cats[name] = cid
                can["categories"].append({"id": cid, "name": name})
            return cats[name]

        if labels:
            cur = labels[0]
            start = 0
            for i, lb in enumerate(labels[1:], start=1):
                if lb != cur:
                    can["annotations"].append(
                        {
                            "id": f"ann_{len(can['annotations'])}",
                            "category_id": cid_of(cur),
                            "start": {"value": start, "unit": "frame"},
                            "end": {"value": i - 1, "unit": "frame"},
                            "attributes": {"source": "frame_txt"},
                        }
                    )
                    cur, start = lb, i
            can["annotations"].append(
                {
                    "id": f"ann_{len(can['annotations'])}",
                    "category_id": cid_of(cur),
                    "start": {"value": start, "unit": "frame"},
                    "end": {"value": len(labels) - 1, "unit": "frame"},
                    "attributes": {"source": "frame_txt"},
                }
            )
        can["extras"]["frame_labels"] = labels
        return can

    def export_from_canonical(
        self, canonical: Canonical, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        num_frames = int(canonical["video"].get("num_frames") or 0)
        id2name = {c["id"]: c["name"] for c in canonical.get("categories", [])}
        frame_labels = ["bg"] * max(1, num_frames)
        for ann in canonical.get("annotations", []):
            name = id2name.get(ann["category_id"], "unknown")
            s = int(ann["start"]["value"])
            e = int(ann["end"]["value"])
            for f in range(max(0, s), min(num_frames - 1, e) + 1):
                frame_labels[f] = name
        return {"frame_labels": frame_labels}


class FACTAdapter(AdapterBase):
    """Adapter for FACT output: {"segments": [...], "classes": [...]}"""

    name = "FACT"

    def detect(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        if not isinstance(payload.get("segments"), list):
            return False
        if not isinstance(payload.get("classes"), list):
            return False
        segs = payload.get("segments") or []
        if not segs:
            return True
        sample = segs[0]
        return (
            isinstance(sample, dict)
            and "start_frame" in sample
            and "end_frame" in sample
        )

    def import_to_canonical(
        self, payload: Dict[str, Any], meta: Dict[str, Any]
    ) -> Canonical:
        """
        payload: FACT json like:
        {
            "classes": ["c0","c1",...],
            "segments": [
                {"start_frame":0, "end_frame":149, "class_id":9, "class_name":"c9"},
                ...
            ]
        }
        """
        can = to_canonical_template(meta)

        classes = payload.get("classes", [])
        segments = payload.get("segments", [])

        can["categories"] = []
        for cid, cname in enumerate(classes):
            can["categories"].append(
                {
                    "id": cid,
                    "name": str(cname),
                }
            )

        id2name = {c["id"]: c["name"] for c in can["categories"]}

        anns = []
        for i, seg in enumerate(segments):
            try:
                s = int(seg["start_frame"])
                e = int(seg["end_frame"])
                cid = int(seg.get("class_id", 0))
            except Exception:
                continue

            if cid not in id2name:
                continue

            anns.append(
                {
                    "id": f"ann_{i}",
                    "category_id": cid,
                    "start": {"value": s, "unit": "frame"},
                    "end": {"value": e, "unit": "frame"},
                    "attributes": {"source": "fact"},
                }
            )

        can["annotations"] = anns
        can["extras"]["raw_fact"] = payload
        return can

    def export_from_canonical(
        self, canonical: Canonical, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        cats = sorted(canonical.get("categories", []), key=lambda c: c["id"])
        classes = [c["name"] for c in cats]
        id2idx = {c["id"]: idx for idx, c in enumerate(cats)}

        segments = []
        for ann in canonical.get("annotations", []):
            cid = int(ann["category_id"])
            idx = id2idx.get(cid, cid)
            s = int(ann["start"]["value"])
            e = int(ann["end"]["value"])
            segments.append(
                {
                    "start_frame": s,
                    "end_frame": e,
                    "class_id": idx,
                    "class_name": (
                        classes[idx] if 0 <= idx < len(classes) else f"c{idx}"
                    ),
                }
            )

        return {
            "classes": classes,
            "segments": segments,
        }


class OurV1Adapter(AdapterBase):
    name = "OurV1"

    def detect(self, payload):
        return isinstance(payload, dict) and "your_special_key" in payload

    def import_to_canonical(self, payload, meta):
        can = to_canonical_template(meta)
        return can

    def export_from_canonical(self, canonical, options=None):
        return {"...": "..."}


class NativeAdapter(AdapterBase):
    """Adapter for the tool's own JSON (labels/segments[, view_start/view_end])."""

    name = "Native"

    def detect(self, payload: Dict[str, Any]) -> bool:
        return (
            isinstance(payload, dict) and "segments" in payload and "labels" in payload
        )

    def import_to_canonical(
        self, payload: Dict[str, Any], meta: Dict[str, Any]
    ) -> Canonical:
        can = to_canonical_template(meta)
        view_start = int(payload.get("view_start", 0) or 0)
        view_end = payload.get("view_end", None)
        if view_end is not None:
            try:
                view_end = int(view_end)
            except Exception:
                view_end = None

        cats = []
        for item in payload.get("labels", []):
            try:
                lid = int(item.get("id"))
            except Exception:
                continue
            name = item.get("name", f"Label_{lid}")
            cats.append({"id": lid, "name": name})
        can["categories"] = sorted(cats, key=lambda x: x["id"])

        for seg in payload.get("segments", []):
            try:
                lid = int(seg.get("action_label"))
                s = int(seg.get("start_frame", 0)) + view_start
                e = int(seg.get("end_frame", s)) + view_start
            except Exception:
                continue
            attrs = {}
            ent = seg.get("entity")
            if ent not in (None, ""):
                attrs["entity"] = ent
            if view_end is not None:
                if e < view_start or s > view_end:
                    continue
                e = min(e, view_end)
            ann = {
                "id": f"ann_{lid}_{s}",
                "category_id": lid,
                "start": {"value": s, "unit": "frame"},
                "end": {"value": e, "unit": "frame"},
            }
            if attrs:
                ann["attributes"] = attrs
            can["annotations"].append(ann)
        return can

    def export_from_canonical(
        self, canonical: Canonical, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        vid = canonical.get("video", {}) or {}
        view_start = int(vid.get("view_start", 0) or 0)
        view_end = vid.get("view_end", None)
        if view_end is not None:
            try:
                view_end = int(view_end)
            except Exception:
                view_end = None

        labels = []
        for c in canonical.get("categories", []):
            labels.append({"id": int(c.get("id", 0)), "name": c.get("name", "")})

        segments = []
        for ann in canonical.get("annotations", []):
            try:
                cid = int(ann.get("category_id"))
                s_abs = int(ann.get("start", {}).get("value"))
                e_abs = int(ann.get("end", {}).get("value"))
            except Exception:
                continue
            attrs = ann.get("attributes", {}) or {}
            ent = attrs.get("entity")
            s = max(0, s_abs - view_start)
            e = max(0, e_abs - view_start)
            if view_end is not None:
                if s_abs < view_start or e_abs > view_end:
                    continue
            seg = {"action_label": cid, "start_frame": s, "end_frame": e}
            if ent not in (None, ""):
                seg["entity"] = ent
            segments.append(seg)

        return {
            "video_id": vid.get("id", ""),
            "view_start": view_start,
            "view_end": view_end,
            "labels": labels,
            "segments": segments,
        }


ADAPTERS = {
    "Native": NativeAdapter(),
    "ActivityNet": ActivityNetAdapter(),
    "FrameTXT": FrameTXTAdapter(),
    "FACT": FACTAdapter(),
    "OurV1": OurV1Adapter(),
}
