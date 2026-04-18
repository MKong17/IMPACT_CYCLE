# File Formats

This document preserves the file-format notes that used to live in the old README.

## Scope

For the current paper path:

- scene graph output
- VQA output
- cycle / evaluation artifacts

For retained legacy workflows:

- Action Segmentation JSON
- label maps
- interaction sidecars
- PSR / ASR / ASD files
- HOI JSON
- validation logs
- operation logs

## Current Paper Outputs

The most complete current examples for scene graph and VQA payloads are already documented in [IMPACT_SG_quick_reference.md](IMPACT_SG_quick_reference.md).

The main files used by the paper-facing path are:

- scene graph JSON from `tools/build_scene_graph.py`
- VQA JSON from `tools/generate_vqa.py`
- cycle verification JSON from `run_cycle_verify.py`
- evaluation summaries under the selected output directory from `run_vidor_gt_frame_eval.py`

## Retained Legacy Formats

### Action Segmentation Native JSON

Written by the old Action Segmentation export flow.

```json
{
  "video_id": "video_001",
  "view": "Top",
  "meta_data": {
    "fps": 30.0,
    "resolution": {"width": 1920, "height": 1080},
    "num_frames": 1901,
    "view_start": 100,
    "view_end": 2000
  },
  "view_start": 100,
  "view_end": 2000,
  "anomaly_types": [{"id": 0, "name": "error_temporal"}],
  "verbs": [{"id": 0, "name": "pick"}],
  "nouns": [{"id": 0, "name": "gear"}],
  "action_labels": [{"id": 3, "name": "pick_gear"}],
  "segments": [
    {
      "action_label": 3,
      "verb": 0,
      "noun": 0,
      "start_frame": 0,
      "end_frame": 120,
      "phase": "anomaly",
      "anomaly_type": [1],
      "entity": "LeftHand"
    }
  ]
}
```

Notes:

- `start_frame` and `end_frame` are relative to `view_start`.
- `entity` appears on Fine mode segments.
- `phase` and `anomaly_type` are used by Fine phase/anomaly workflows.
- Older JSON without `meta_data` is still loadable.

### Label Map TXT

```text
label_name label_id
```

Example:

```text
pick 3
place 4
```

### Interaction Sidecar

- Saved as `<annotations>_extra.json`
- Contains only `Interaction` spans

### Supported Legacy Import / Export Adapters

The historical annotation import/export actions supported:

- `Native`
- `ActivityNet`
- `FrameTXT`
- `FACT`
- `OurV1`

### PSR / ASR / ASD Component Library

JSON / YAML form:

```json
{
  "components": [
    {"id": 0, "name": "base"},
    {"id": 1, "name": "front_chassis"},
    {"id": 2, "name": "front_chassis_pin"}
  ]
}
```

TXT form:

```text
0, base
1, front_chassis
2, front_chassis_pin
```

Notes:

- If no component file is loaded, components were historically auto-extracted from coarse label nouns.
- Component order defines the state-vector index.

### PSR / ASR / ASD Rules File

```json
{
  "rules": [
    {
      "label": "Install front chassis",
      "components": [{"component": "front_chassis"}],
      "state": 1
    },
    {
      "label": "Error front chassis",
      "components": [{"component": "front_chassis"}],
      "state": -1
    }
  ]
}
```

Notes:

- Rule trigger is the segment end frame.
- If a label contains `error`, the default state is `-1` when not explicitly set.

### Derived ASR Output JSON

```json
{
  "task": "ASR",
  "version": "1.0",
  "video_id": "video_001",
  "fps": 30.0,
  "view_start": 100,
  "view_end": 2000,
  "frame_count": 2400,
  "meta_data": {
    "fps": 30.0,
    "resolution": {"width": 1920, "height": 1080},
    "num_frames": 1901,
    "view_start": 100,
    "view_end": 2000,
    "video_num_frames": 2400,
    "workflow": "assemble",
    "initial_state": 0,
    "initial_state_label": "Not installed",
    "model_type": "CG15-125BL"
  },
  "initial_state": 0,
  "initial_state_vector": [0],
  "components": [{"id": 0, "name": "base"}],
  "state_sequence": [{"frame": 0, "state": [0]}, {"frame": 240, "state": [1]}],
  "state_changes": [{"frame": 240, "component_id": 0, "state": 1}]
}
```

Notes:

- Frame indices are relative to `view_start`.
- `state_sequence` is the authoritative timeline state.
- `state_changes` is a sparse debugging / compatibility view.

### HOI JSON

```json
{
  "version": "HOI-1.0-ActionSeg",
  "video_id": "video_001",
  "video_path": "path/to/video.mp4",
  "fps": 30,
  "frame_size": [1920, 1080],
  "frame_count": 1400,
  "bbox_mode": "xyxy",
  "bbox_normalized": false,
  "object_library": {
    "12": {"label": "screwdriver_1", "category": "screwdriver", "class_id": 4}
  },
  "verb_library": {"0": "pick", "1": "place"},
  "anomaly_rules": {
    "Normal": {"allow_missing_bbox": false, "allow_missing_verb": false}
  },
  "tracks": {
    "T_LHAND": {
      "category": "left_hand",
      "object_id": null,
      "boxes": [{"frame": 10, "bbox": [1, 2, 3, 4]}]
    }
  },
  "hoi_events": {
    "left_hand": [
      {
        "event_id": "L_001",
        "start_frame": 10,
        "contact_onset_frame": 20,
        "end_frame": 40,
        "verb": "pick",
        "interaction": {"tool": "screwdriver_1", "target": "housing_1"},
        "links": {"subject_track_id": "T_LHAND", "tool_track_id": "T_OBJ_12"},
        "anomaly_label": "Normal"
      }
    ],
    "right_hand": []
  }
}
```

Notes:

- Hand tracks store all available hand boxes.
- Object tracks store keyframe boxes.
- Missing verb / bbox can be allowed through anomaly rules.

## Validation Logs

Historical validation outputs:

- Action Segmentation: `<annotations>.validation.log.txt`
- HOI: `<annotations>.validation.json`

## Operation Logs

Enable operation logging with:

```bash
python app.py --oplog
```

Historical log files:

- Action Segmentation: `<annotations>.ops.log.csv`
- HOI: `<annotations>.ops.log.csv`
- HOI validation: `<annotations>.validation.ops.log.csv`

Important behavior:

- logging depends on the logging toggles
- annotation save/export is independent from log writing
- if log writing fails, annotation export can still succeed

## Related References

- [IMPACT_SG_quick_reference.md](IMPACT_SG_quick_reference.md)
- [archive/legacy_annotation_manual.md](archive/legacy_annotation_manual.md)
