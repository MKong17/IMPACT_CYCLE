# Legacy Annotation Manual

This document archives the non-paper operator notes that used to live in the old monolithic README.

## Status

These workflows are not part of the main IMPACT-CYCLE paper reproduction path. They are retained for historical context, internal use, and collaborators who still need the older tooling:

- Action Segmentation
- HandOI / HOI Detection
- Assembly State (`PSR` / `ASR` / `ASD`)
- Transcript workspace

For the current paper GUI, use [../gui_operator_guide.md](../gui_operator_guide.md).

## Action Segmentation

### Top-Level Controls

Historical top-bar controls included:

- task switching
- playback and frame jumping
- grouped `Choose action...` file / model / review utilities
- `Mode` selection for coarse vs fine annotation
- playback speed
- settings
- `EAST Refine`
- `ASOT Pre-label`
- validation toggle
- interaction-mode selection

### Session Startup

`Choose action... -> Open Session...` was the lowest-friction startup path. It could:

- choose the target video
- auto-discover a nearby label TXT
- auto-discover a nearby annotation JSON
- show the current EAST setup
- jump directly into `Configure EAST...`

### EAST Setup

`Choose action... -> EAST Setup...` exposed the main runtime configuration:

- checkpoint
- config
- label text backend
- shared adapter selection
- masked refresh context

Important retained notes:

- the dialog stored state in `.cvhci_local/east_ui_state.json`
- checkpoint and config had to come from the same EAST experiment setup
- runtime artifacts such as `online_adapter.pt` and `model_delta.pt` were not the base checkpoint

Example historical paths and the old long-form explanation are preserved in `README_old.md`.

### ASOT Semantic Remap

`ASOT Pre-label` was treated as a proposal generator, not direct EAST supervision.

Retained behavior:

- remap sidecars such as `asot_label_remap.json` could map cluster IDs to semantic labels
- the UI offered `ASOT: Build Label Remap...`
- untouched ASOT output was still only a proposal
- only human-confirmed corrections were allowed into EAST online learning

### Batch Pre-Labeling

`Choose action... -> Batch Pre-label...` historically:

- reused the current EAST checkpoint / config
- reused the current label bank
- optionally reused the selected shared adapter
- ran offline proposal generation over a video folder

### Runtime Tools

Historical runtime actions included:

- inspect runtime assets
- export runtime report
- export shared adapter
- select or clear shared adapter
- consolidate multiple runtimes into one shared adapter

### Online-Learning Workflow

The essential design remains:

1. Generate a proposal from `ASOT` or `EAST`.
2. Let the annotator confirm or correct it.
3. Convert only the final confirmed state into structured supervision.
4. Rebuild lightweight runtime assets such as `model_delta.pt`, `online_adapter.pt`, and `label_text_bank.pt`.
5. Preserve locked / confirmed regions while refreshing unlocked ones.

For the current method-level explanation, use [../interactive_action_segmentation_method_guide_20260320.md](../interactive_action_segmentation_method_guide_20260320.md).

### Runtime Assets

Historical `east_runtime/` directories could contain:

- `segments.json`
- `boundary.npy`
- `label_scores.npy`
- `seg_embeds.npy`
- `transition.npy`
- `prototype.npy`
- `record_buffer.pkl`
- `model_delta.pt`
- `online_adapter.pt`
- `label_text_bank.pt`
- `meta.json`

### Multi-View and Timeline Editing

Retained UI concepts:

- multiple synchronized views
- active-view editing
- combined or per-label timeline layouts
- gap navigation
- entity-specific labeling in Fine mode
- read-only masked regions during sync edit / validation

### Validation and Review

Historical Action Segmentation validation behavior:

- validation mode required an editor name
- label changes were logged with frame ranges, view name, and editor
- save wrote `<annotations>.validation.log.txt`
- review logs could be imported and accepted / rejected item by item

### Interaction Modes

Two historical interaction modes were documented:

- `Manual Segmentation` for quick global `Interaction` labeling
- `Assisted Review` for model-guided correction using queued uncertain points

### Transcript Workspace

The old GUI also exposed a transcript workspace for subtitle-like speech cues:

- attach external audio
- apply audio offsets
- generate or import transcript JSON
- convert transcript spans into action intervals

## HandOI / HOI Detection

The HOI task historically included:

- frame-level playback and jumping
- import of instrument lists, target lists, verb lists, and YOLO boxes
- YOLO + MediaPipe detection
- editable hand / object boxes
- left-hand and right-hand event timelines
- anomaly rules
- validation overlays and validation logs

The primary saved format was the historical `HOI-1.0-ActionSeg` JSON documented in [../file_formats.md](../file_formats.md).

## Assembly State (`PSR` / `ASR` / `ASD`)

Historical assembly-state notes retained from the old README:

- shared Action timeline with a state-oriented right panel
- `Unobservable` as a normal timeline label
- component-library loading
- rules-file loading and export
- state-json loading for continuation
- split / merge / reset / invert segment tools
- rule trigger at segment end frame
- export to derived ASR JSON

## Shortcuts and Troubleshooting

The old README also kept large default-shortcut tables and troubleshooting notes. Those details were intentionally not promoted back into the main README because they are:

- operator-specific
- partly legacy
- not needed for paper reproduction

If those workflows become actively used again, restore them in a dedicated task-specific doc rather than the main project README.

## Source

This archive was distilled from `README_old.md`, which remains the fullest historical reference for the pre-paper GUI/operator manual.
