# IMPACT VQA Video Understanding Suite
A desktop video understanding and annotation suite built with PyQt5 and OpenCV.

Current build focus:
- Video-first workflow (commercial-style playback controls)
- Task-optimized workspace for:
  - Video Scene Graph
  - Single-turn VQA
  - Multi-turn VQA
  - Video Captioning

This README is an operator manual. It is written for first-time users and documents the currently exposed workflows, controls, and file formats.

## Current task model (March 2026)
When you run `python app.py`, the main window now has one shared video player and four task panels:

1. `Video Scene Graph`
- Build scene graph on selected frame from the loaded video.

2. `Single-turn VQA`
- Generate and inspect single-turn QA from current scene graph.

3. `Multi-turn VQA`
- Generate chain-based QA turns from current scene graph.

4. `Video Captioning`
- Generate caption text for a frame interval with style selection.

## Table of contents
1. Quickstart
2. Installation
3. Launch and CLI options
4. Global UI layout and behaviors
5. Video Scene Graph workspace
6. Single-turn VQA workspace
7. Multi-turn VQA workspace
8. Video Captioning workspace
9. IMPACT-SG CLI workflow
10. File formats
11. Operation logs (oplog)
12. Keyboard shortcuts
13. Troubleshooting
14. Legacy sections

---

## Quickstart

### Shared video controls (fastest path)
1. Run `python app.py`.
2. Click the folder icon to open a video.
3. Use icon transport controls (`open`, `prev`, `play/pause`, `next`, `stop`) and the seek slider to navigate frames.
4. Set a frame for graph/VQA tasks using current frame or frame input.

### Video Scene Graph (fastest path)
1. Run `python app.py`.
2. Select **Video Scene Graph** from Task dropdown.
3. Set frame and click **Build Scene Graph For Frame**.
4. Save with **Save Scene Graph JSON**.

### Single-turn VQA (fastest path)
1. Build scene graph first.
2. Switch to **Single-turn VQA**.
3. Click **Generate Single-turn VQA**.
4. Select a question to inspect details and save JSON.

### Multi-turn VQA (fastest path)
1. Build scene graph first.
2. Switch to **Multi-turn VQA**.
3. Click **Generate Multi-turn VQA**.
4. Inspect chain turns and save JSON.

### Video Captioning (fastest path)
1. Optionally build scene graph first for richer captions.
2. Switch to **Video Captioning**.
3. Set start/end frame and style.
4. Click **Generate Caption** and save TXT.

### IMPACT-SG CLI (fastest path)
1. Build graph:
  - `python tools/build_scene_graph.py --image_id IMG_001 --image_path <frame_image_path> --out out_graph.json`
2. Generate VQA:
  - `python tools/generate_vqa.py --scene_graph out_graph.json --out out_vqa.json`
3. Evaluate graph (optional):
  - `python tools/evaluate_scene_graph.py --pred pred_graph.json --gt gt_graph.json --out graph_metrics.json`
4. Evaluate VQA (optional):
  - `python tools/evaluate_vqa.py --pred pred_vqa.json --gt gt_vqa.json --out vqa_metrics.json`

---

## Installation

### Requirements (one command)
```
pip install -r requirements.txt
```

### Optional external tools
- **ffmpeg/ffprobe**: used by the ASR pipeline and audio extraction.
- **GPU drivers**: optional for YOLO inference on GPU.

---

## Launch and CLI options

### Normal launch
```
python app.py
```

### Enable operation logging
```
python app.py --oplog
```
This enables `*.ops.log.csv` output when you save annotations. See the Operation Logs section for details.

---

## Global UI layout and behaviors

### Workspace switching
- One shared video player is always visible on the left.
- The right workspace changes with the selected task.
- This avoids forcing all tasks into timeline/verb annotation patterns.

### Resizable layout
The main window uses splitters. You can drag the splitter handles to resize:
- Left: shared video player and transport controls
- Right: active task workspace

### Video viewer controls (applies to both tasks)
- **Ctrl + mouse wheel**: zoom around cursor (0.2x to 8x).
- **Left-drag**: pan the video when zoomed.
- **Center** button (top-right on video): re-center while keeping zoom.
- **Double-click**: reset zoom to 1x and center the view.

### Settings (Action Segmentation)
- Open Settings from the **gear button (⚙)** near Speed, or press **Ctrl+,**.
- **Timeline Editing & Snapping** includes:
  - Playhead snap radius
  - Empty-space snap radius
  - Edge search radius
  - Segment soft snap radius
  - Phase soft snap radius
  - Multi-view hover preview enable/disable
  - Hover preview alignment (**Absolute frame** or **Relative to active view**)
- **Keyboard Controls** provides per-action rebinding tabs (Action/Review/Assisted/HOI/PSR).
- Conflicting bindings in the same scope are highlighted live; you can still force-save after confirmation.
- **Logging** provides two independent global toggles (shared by Action/HOI):
  - `ops_csv_enabled`: writes `*.ops.log.csv` and `*.validation.ops.log.csv`
  - `validation_summary_enabled`: writes `*.validation.log.txt` and `*.validation.json` (validation sessions)
  - `validation_comment_prompt_enabled`: controls whether validation edits prompt an optional comment dialog
- Logging toggles are auto-saved and auto-loaded:
  - Primary: `~/.cvhci_video_annotation_suite/logging_policy.json`
  - Backup: `~/.cvhci_video_annotation_suite/logging_policy.backup.json`
  - Backward-compatible: legacy single-key logging configs are still accepted.
- Shortcut bindings are auto-saved and auto-loaded:
  - Primary: `~/.cvhci_video_annotation_suite/shortcuts.json`
  - Backup: `~/.cvhci_video_annotation_suite/shortcuts.backup.json`
  - If primary is missing/corrupted, backup is used; if both fail, built-in defaults are used.
  - Optional override: set `CVHCI_SETTINGS_DIR` to change the settings directory.
- There is no manual shortcut import/export step in the UI.

---

## Video Scene Graph workspace

Uses `ui/video_task_studio.py` with frame-centric graph construction from the loaded video.

Highlights:
- Dual tables for `nodes` and `edges`
- Node-edge linked highlighting (select node -> related edges highlight; select edge -> src/dst nodes highlight)
- JSON preview and direct export

---

## Single-turn VQA workspace

Uses current scene graph to create one-turn QA items and allows per-item detail inspection.

---

## Multi-turn VQA workspace

Uses current scene graph to create multi-turn chains with turn-by-turn inspection.

---

## Video Captioning workspace

Generates style-controlled caption text over selected frame interval.

Highlights:
- Single-segment caption generation (`Concise`, `Detailed`, `Technical`)
- Batch segment list management
- Batch caption generation
- Batch export to `JSONL` and `TXT`

---

## IMPACT-SG CLI workflow

- Build graph:
  - `python tools/build_scene_graph.py --image_id IMG_001 --image_path <frame_image_path> --out out_graph.json`
- Generate VQA:
  - `python tools/generate_vqa.py --scene_graph out_graph.json --out out_vqa.json`
- Evaluate scene graph:
  - `python tools/evaluate_scene_graph.py --pred pred_graph.json --gt gt_graph.json --out graph_metrics.json`
- Evaluate VQA:
  - `python tools/evaluate_vqa.py --pred pred_vqa.json --gt gt_vqa.json --out vqa_metrics.json`

---

## Legacy sections

The detailed sections below were written for earlier task names and remain useful for low-level controls and file formats.
If you see `Action Segmentation`, `HandOI / HOI Detection`, or `Assembly State (PSR/ASR/ASD)`, treat them as legacy reference material that is not part of the current default four-task UI.

---

## Action Segmentation

### Top control bar
- **Task**: switch to Action Segmentation or other tasks.
- **Playback**: rewind (-10), **play/pause (single toggle button)**, stop (resets to crop start), fast-forward (+10).
- **Jump to frame**: enter a frame index and click Go (or press Enter).
- **Choose action...**: grouped file I/O and utilities.
  - **Session**: open/import/export the current working session.
  - **Labels**: import/export label maps.
  - **Model**: ASOT remap, batch pre-labeling, EAST setup, and EAST runtime/shared-adapter tools.
  - **Review**: validation log import and review queue access.
  - **Transcript**: transcript workspace and external audio utilities.
- **Mode**: Coarse or Fine.
- **Speed**: playback rate (0.25x to 2x). Audio may remain at 1x if the device does not support rate change.
- **Settings**: open snapping/shortcut settings.
- **EAST Refine**: run EAST refinement on the current video.
- **ASOT Pre-label**: run ASOT pre-labeling on the current video.
- **Magnifier**: enables select-to-zoom mode in the active view.
- **Validation**: toggle Validation mode on/off.
- **Interaction**: choose Manual Segmentation or Assisted Review modes.

### Open Session
`Choose action... -> Open Session...` is the lowest-friction way to start a new Action Segmentation session.

The dialog is designed to reduce setup overhead without introducing a large project-management workflow. It:

- asks for the target video,
- can autofill a nearby label TXT via the current label-source resolver,
- can autofill a nearby annotation JSON using the video filename as a hint,
- shows the current EAST setup summary,
- lets you jump into **Configure EAST...** without leaving the session flow.

When you confirm the dialog:

- the video is loaded through the normal video-loading path,
- you still choose Start/End frames for the primary view,
- the selected label map is imported if requested,
- the selected JSON is imported if requested.

This keeps the main workflow short while still reusing the same loading logic as the individual menu entries.

### EAST model files
`EAST Refine` needs two files:

- **EAST checkpoint**: a model weight file such as `.pth` or `.pt`
- **EAST config**: a model config file such as `.py`, `.yaml`, or `.yml`

The tool will ask for both files if they are not already configured.

### EAST Setup
`Choose action... -> EAST Setup...` is the recommended place to configure the current EAST workflow without opening a large research-style settings table.

It intentionally exposes only the parameters that affect the current production workflow:

- **Checkpoint**
  - local EAST model checkpoint used by `EAST Refine` and `Batch Pre-label...`
- **Config**
  - local EAST config matched to the checkpoint
- **Label text backend**
  - `Auto`
  - `MobileCLIP / CLIP`
  - `SentenceTransformer`
  - `Lexical fallback`
- **Shared adapter**
  - select or clear the currently active shared adapter bundle
- **Masked refresh context (frames)**
  - controls how much context the local post-update refresh keeps around unlocked windows

Notes:
- The dialog stores these UI-local choices in `.cvhci_local/east_ui_state.json`.
- If `Checkpoint` or `Config` is left blank, `EAST Refine` will still prompt when needed.
- `Reset to defaults` reloads local environment defaults from `feature_defaults.json` / environment variables.

Current default examples in `feature_defaults.json` are:

- `EAST_CFG`: `external/EAST-main/configs/adatad/assembly101/e2e_assembly101_videomaev2_g_768x1_160_adapter.py`
- `EAST_CKPT`: `external/EAST-main/pretrained/e2e_actionformer_videomaev2_g_768x1_160_adapter3_fps6_sde1_p0.5/split1/gpu2_id0/checkpoint/epoch_12.pth`

Recommended local layout:

```text
external/
  EAST-main/
    configs/
      adatad/
        assembly101/
          e2e_assembly101_videomaev2_g_768x1_160_adapter.py
    pretrained/
      vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth
      e2e_actionformer_videomaev2_g_768x1_160_adapter3_fps6_sde1_p0.5/
        split1/
          gpu2_id0/
            checkpoint/
              epoch_12.pth
```

In this layout:

- Select `external/EAST-main/configs/adatad/assembly101/e2e_assembly101_videomaev2_g_768x1_160_adapter.py` as the EAST config.
- Select `external/EAST-main/pretrained/e2e_actionformer_videomaev2_g_768x1_160_adapter3_fps6_sde1_p0.5/split1/gpu2_id0/checkpoint/epoch_12.pth` as the main EAST checkpoint.
- The backbone pretrain `vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth` is a dependency referenced by the config. It should live under `external/EAST-main/pretrained/`, but it is **not** the checkpoint you choose in the UI.

Important:

- These are **example/default paths**, not guaranteed files in this repository.
- If `external/EAST-main` is missing locally, you must provide your own matching `cfg + ckpt`.
- The checkpoint and config should come from the **same EAST experiment setup**.
- For this tool, prefer an **Assembly101 EAST segmentation checkpoint** paired with its matching VideoMAEv2 adapter config.
- `external/` is git-ignored in this repository, so local EAST weights, checkpoints, logs, and backbone pretrains stored there will **not** be uploaded by default.

Do **not** select these files as the main EAST checkpoint:

- `online_adapter.pt`
- `model_delta.pt`
- `prototype.npy`
- `transition.npy`
- `boundary.npy`

Those files are runtime or online-adaptation artifacts, not the base EAST model weights.

### ASOT semantic remap
`ASOT Pre-label` is treated as a proposal generator, not as a direct EAST training target. To make ASOT proposals align better with the current semantic label space, the tool now supports an optional **ASOT cluster-to-label remap**.

Default behavior:
- ASOT first resolves the semantic label bank from:
  1. the currently active Action labels in the UI,
  2. the imported label-map source file if it still matches the current UI labels,
  3. a nearby label TXT discovered from the current video/features path,
  4. a manual TXT selection dialog as fallback.
- When a remap JSON exists, ASOT applies it automatically during export.

Supported remap sidecar names:
- `asot_label_remap.json`
- `label_remap.json`

Discovery order:
- explicit `--label_remap_json`
- `ASOT_LABEL_REMAP_JSON`
- next to the semantic label bank TXT
- next to / above the current `*_features` directory

ASOT now writes both:
- `{pred_prefix}_cluster_per_frame.npy`
  - raw/smoothed cluster IDs for calibration
- `{pred_prefix}_per_frame.npy`
  - semantic IDs after remap, when a remap is active

The exported JSON also records:
- whether a label remap was applied
- which remap file was used
- how many cluster mappings were applied

To build a remap interactively:
- `Choose action... -> ASOT: Build Label Remap...`

This UI action is a lightweight wrapper around `tools/build_asot_label_remap.py`:
- the UI keeps the simple single-root case
- the CLI still supports multiple dataset roots and multiple feature-search roots for larger calibration jobs

The builder expects a dataset root with:
- `groundTruth/`
- `videos/`

It compares GT labels against raw ASOT cluster predictions and writes a majority-vote `asot_label_remap.json`.

Best-practice interpretation:
- untouched ASOT output is still only a proposal
- once remapped, it becomes a cleaner semantic proposal for human review
- only later **human-confirmed** edits are allowed into EAST online learning

### Batch pre-labeling
To precompute EAST proposals at dataset scale without opening one video at a time:

- `Choose action... -> Batch Pre-label...`

This workflow:
- asks for an input video folder
- asks for an output folder
- reuses the current EAST checkpoint/config
- reuses the current label bank
- reuses the currently selected shared adapter, if one is active
- extracts per-video features
- runs EAST inference for every video
- writes per-video outputs under the chosen output directory

This is intentionally separate from the single-video interactive loop:
- **batch EAST** is for offline proposal generation
- **interactive EAST** is for human correction, online updates, and masked refresh

The main UI now routes batch pre-labeling through EAST by default. Legacy FACT batch tooling remains in the codebase only for compatibility and is no longer exposed as a first-class Action menu entry.

### EAST runtime tools
These **Choose action...** entries work on the current video's `east_runtime` assets and do not require the base EAST checkpoint:

- **EAST: Inspect Runtime Assets**
  Opens the runtime/debug dialog for the current video.
  - **Summary** tab: compact operational view for day-to-day use, including learning-state stats and the current Assisted Review query-policy snapshot.
  - **Raw JSON** tab: full payload for debugging, experiment notes, or bug reports.
- **EAST: Export Runtime Report...**
  Saves a JSON snapshot of the current runtime state for debugging or experiment tracking.
  - Includes current runtime metadata, online-learning stats, the current query-policy snapshot, and the currently selected shared adapter (if any).
- **EAST: Export Shared Adapter...**
  Exports the current video's runtime assets as a reusable shared adapter bundle.
- **EAST: Select Shared Adapter...**
  Selects a shared adapter bundle to use during later `EAST Refine` runs.
  - The selection is restored across app restarts.
  - Missing/deleted bundle paths are ignored automatically; the app will not pass an invalid path into `EAST Refine`.
- **EAST: Clear Shared Adapter**
  Clears the currently selected shared adapter bundle.
- **EAST: Consolidate Shared Adapter...**
  Scans a folder tree for multiple `east_runtime` directories and merges them into one shared adapter bundle for multi-video reuse.
  - After consolidation, the newly written bundle is selected automatically for later `EAST Refine` runs in the current session.

### EAST online-learning workflow
This is the current intended workflow for correction-driven improvement inside the Action Segmentation module.

For a collaborator-facing method summary, see:
- `docs/interactive_action_segmentation_method_guide_20260320.md`

1. Generate a starting annotation:
   - **ASOT Pre-label** for a quick initial proposal, or
   - **EAST Refine** for the current EAST-based refinement path.
2. Edit the result manually:
   - split
   - move a boundary
   - remove a boundary
   - delete a segment
   - replace a label
   - remove a label
3. Confirm the final result for that review session.
4. The tool compares the **final confirmed result** against the current baseline and writes a structured correction buffer.
5. Background jobs then rebuild:
   - `model_delta.pt`
   - `online_adapter.pt`
   - `label_text_bank.pt`
6. After the online update finishes, the tool may run a quiet **masked post-update refresh**:
   - directly confirmed label spans stay locked
   - unlocked regions are refreshed with the updated EAST runtime
   - refresh is computed on **local unlocked windows with context**, rather than rebuilding the entire video and only applying part of the result
   - confirmed boundary edits also split later refresh windows, so local boundary fixes do not keep forcing a full-span refresh
   - only the unlocked part of the current baseline is updated, so future correction diffs are still human-only
7. The runtime is also marked dirty, so a later manual **EAST Refine** run rebuilds runtime assets using the latest confirmed corrections.
8. Segments that were directly relabeled and confirmed are exported as **locked segments** so later refinement preserves those accepted spans and focuses on the remaining unlocked regions.
9. If a video started from **ASOT Pre-label** and later accumulates confirmed human supervision but still has no EAST runtime yet, the tool can automatically bootstrap an EAST feature/runtime directory in the background and start EAST online learning from those confirmed records.
   - This bootstrap step only initializes EAST features/runtime and online-learning assets.
   - It does not immediately run a full visible EAST refinement pass over the current timeline.

Important behavior:
- The system does **not** train on every intermediate drag or temporary click.
- Only **human-confirmed supervision** is allowed into the EAST online-learning path.
  - `corrected` spans: the annotator changed the boundary and/or label.
  - `accepted` spans: the annotator explicitly reviewed the proposal and accepted it as correct.
  - `finalized` spans: the video was explicitly finalized and the current labels/boundaries are treated as full GT-strength supervision.
  - untouched auto-proposals are never used as EAST supervision, even if they came from ASOT or EAST.
- This is intentional: transient editing noise is ignored, while the accepted end result is preserved.
- Boundary edits still remain active even when a full segment is not locked:
  - explicit human boundaries are written to the correction buffer
  - later EAST refinement protects those boundary frames
  - the automatic masked refresh only rewrites areas that are still considered unlocked
- Supervision strength is intentionally layered:
  - `accepted`: weaker positive supervision
  - `corrected`: stronger positive supervision plus hard negatives
  - `finalized`: strongest supervision, also used by the second-timescale offline shared-adapter training path

### Confirmed correction semantics
The current correction buffer is versioned and stores structured end-state differences rather than raw UI events.

Supported confirmed action types include:
- `label_assign`
- `label_accept`
- `label_replace`
- `label_remove`
- `label_finalize`
- `boundary_add`
- `boundary_accept`
- `boundary_move`
- `boundary_remove`
- `boundary_finalize`
- `transition_add`
- `transition_remove`
- `segment_lock`

These are compiled from the final annotation state, not from each intermediate editing step.

For label corrections, records may also include:
- `hard_negative_labels`
  - the replaced old label
  - overlapping model top-k candidates that were not accepted by the human
  - persistent confusion-memory negatives accumulated from earlier confirmed corrections of the accepted label

This is used by the online label learner as explicit correction-driven negative supervision.

### EAST runtime assets (`east_runtime/`)
The current video's EAST runtime directory may contain:

- `segments.json`
  Current runtime segment payload used by EAST caching/rebuild logic.
- `boundary.npy`
  Boundary score curve used for snapping and runtime rebuilding.
- `label_scores.npy`
  Segment-level class scores used for top-k suggestions.
- `seg_embeds.npy`
  Segment embeddings used for label reweighting and prototype logic.
- `transition.npy`
  Runtime transition matrix derived from current segmentation output.
- `prototype.npy`
  Runtime prototype table written by EAST runtime generation.
- `record_buffer.pkl`
  Confirmed correction buffer compiled from final human edits.
- `model_delta.pt`
  Lightweight online-learning payload containing label prototypes and transition deltas.
- `online_adapter.pt`
  Lightweight online adapter used to recalibrate features and boundary scores.
- `label_text_bank.pt`
  Label-side text prior table used to initialize and stabilize segment-label alignment.
- `meta.json`
  Runtime metadata, cache identity, dirty flag, and online-update bookkeeping.

### What `model_delta.pt` and `online_adapter.pt` do
- `model_delta.pt`
  - stores label prototype updates
  - stores transition priors/deltas
  - stores persistent label-confusion memory (`confusion_deltas`)
  - is rebuilt from confirmed corrections
- `online_adapter.pt`
  - is a lightweight learned correction module
  - calibrates boundary probability and feature representation
  - keeps a learned class table (`text_table`) for label reranking
  - keeps a lightweight learned fusion gate so label reranking does not rely on one fixed global mixture
  - is trained in the background from confirmed corrections
  - uses persistent confusion memory to expand hard-negative ranking supervision
- `label_text_bank.pt`
  - stores a label-side text prior table in feature space
  - first tries a CLIP/MobileCLIP text encoder through the optional `mobileclip` runner environment
  - then falls back to `sentence-transformers` in the current environment
  - otherwise falls back to a deterministic lexical text prior so the workflow still runs without extra dependencies

Optional text-bank environment knobs:
- `EAST_TEXT_BANK_BACKEND`
  - `auto` (default)
  - `mobileclip`
  - `sentence_transformers`
  - `hashed_lexical`
- `EAST_TEXT_BANK_CLIP_MODEL`
  - local HuggingFace model id or cached MobileCLIP / CLIP model name for the `mobileclip` runner
- `EAST_TEXT_BANK_SENTENCE_MODEL`
  - sentence-transformer fallback model in the current environment
- `EAST_TEXT_BANK_ALLOW_REMOTE=1`
  - allows the external CLIP/MobileCLIP loader to fetch from HuggingFace if the model is not cached locally
  - leave this unset for offline or controlled environments

The current label-side alignment is therefore driven by four sources:
- base EAST label scores
- label text prior (`label_text_bank.pt`)
- learned online class table (`online_adapter.pt`)
- confirmed visual prototypes (`model_delta.pt`)

During runtime reranking, the tool no longer assumes one fixed fusion ratio for all segments. A lightweight learned fusion gate can reweight these sources per segment using:
- base confidence / margin / entropy
- segment length
- whether text / class-table / prototype priors are currently available

Neither file is the base EAST model checkpoint.

### Shared adapter bundles
A shared adapter bundle is a reusable multi-video asset built from one or more runtimes.

It may contain:
- merged `online_adapter`
- merged `model_delta`
- label bank information
- bundle-level metadata
- member runtime summary

If some member runtimes were explicitly finalized, consolidation now does more than weighted averaging:
- finalized runtimes are used as the second timescale of training
- the consolidator can train a shared `online_adapter` offline from finalized GT-strength supervision
- this offline shared adapter can also carry the learned fusion gate
- the merged bundle still keeps the lighter `model_delta` / text-bank style assets for reuse on future videos

Typical usage:
1. Correct one or more videos.
2. Let the background online-learning jobs finish.
3. If a video is fully verified, use **Choose action... -> EAST: Finalize Current Video** before exporting/consolidating.
4. Export each runtime as a shared adapter, or directly consolidate a folder tree.
5. Select the resulting shared adapter.
6. Run **EAST Refine** on a new video so it starts with prior corrected knowledge.

### Shared adapter persistence
- The currently selected shared adapter is stored locally in:
  - `.cvhci_local/east_ui_state.json`
- This folder is intentionally git-ignored.
- Shortcut and logging settings still use `~/.cvhci_video_annotation_suite/...`; EAST shared-adapter UI state is separate because it is project-local runtime state.

### EAST Runtime Debug panel
The debug panel is intentionally split into two layers:

- **Summary**
  - operational status only
  - runtime, learning state, selected shared adapter, confirmed action counts
- **Raw JSON**
  - complete machine-oriented payload
  - useful for debugging or experiment snapshots

The default UI aims to stay quiet and readable; raw details are available only when explicitly requested.

### Video area (multi-view)
- **Views + button**: add a synchronized view (max 5).
- **View name dropdown**: select a preset name or choose **Custom...** to type a unique name.
- **Close view (X)**: closes the view (prompts if unsaved changes exist).
- **Click a view**: makes it the active view. Only the active view is edited.
- **Active view highlight**: a blue border indicates the active view.
- **Ctrl+click views**: toggle multi-view sync-edit selection (works in both normal annotation and Validation).
- **Exit multi-select quickly**: single-click any view (without Ctrl) to return to single-view editing on that view.
- **Play behavior**: when Play starts, all views are aligned to the active view's current frame and then run in sync; when paused, each view can be scrubbed independently.

### Timeline area
- **Single timeline** toggle:
  - On: combined row(s) with exclusive labels.
  - Off: one row per label (or per entity+label in Fine mode).
- **View start slider**: pans the visible window.
- **View span slider**: zooms the visible window.
- **Gap summary + < >**: shows unlabeled gaps and lets you jump between them.
- **Timeline hover**: previews hovered frames; in multi-view validation it can preview selected views together (configurable in Settings).
- **Multi-view sync edit mask**: non-overlapping regions are shown as gray diagonal hatch (including combined rows) and are read-only.

### Label panel (left)
- **Search**: filters labels and highlights matching segments on the timeline.
- **New label**: name field + id + color, then Add.
- **Remove Selected**: deletes the selected label and its segments.
- **Rename**: double-click a label to rename.
- **Color**: preset or custom color key for display.

### Entity panel (Fine mode only)
- **Add entity**: name + id + Add.
- **Remove Selected**: deletes entity.
- **Rename**: double-click a list item.
- **Checkboxes**:
  - In Fine mode (normal): selects which entities a label applies to.
  - In Fine + Validation: selects which entities are visible on video overlays.

### Loading a video
1. Choose **Choose action... -> Load Video...**.
2. Enter **Start frame** and **End frame**.
3. The crop range defines the valid timeline span.
4. If a video has audio, it is attached automatically (if possible).

### Adding a synchronized view
1. Click **+** in the Views toolbar.
2. Select the additional video.
3. Enter **Start** and **End** frames.
4. The span must match the primary view; otherwise import is blocked.
5. The new view receives a cloned copy of the active view annotations.

### Coarse mode workflow
1. Add labels in the Label panel.
2. Choose **Single timeline** or per-label rows.
3. Create a segment:
   - Combined row: drag on empty space to create a new unlabeled segment.
   - Per-label row: drag directly on the label row.
4. Assign or change labels:
   - Combined row: double-click a segment to select it, then click a label on the left.
5. Edit segments:
   - Drag edges to resize.
   - Drag inside to move.
   - Right-click to delete.

### Fine mode workflow
Fine mode supports entity-specific labeling.

1. Create entities in the Entity panel.
2. Select a label in the Label panel.
3. Check the entities that label applies to (applicability mapping).
4. Annotate:
   - **Single timeline ON**: combined layout per entity, and phase rows are shown in the combined stack.
   - **Single timeline OFF**: rows appear per entity+label according to applicability mapping; phase rows remain as per-entity rows at the end.
5. Edits work the same as in Coarse mode (create, resize, move, delete).

Fine verb/noun decomposition behavior:
- Fine-mode `verb` / `noun` vocab is rebuilt from the current action labels.
- If you refine the verb interpretation of multi-token labels (for example `hand_spin_drive_shaft` should parse as `hand_spin + drive_shaft`), the noun list is rebuilt automatically from the updated decomposition.
- This refresh only updates the `verb+noun` interpretation and exported Fine vocab; it does **not** move or relabel already confirmed timeline segments.

### Assembly state annotation mode
PSR/ASR/ASD uses the shared Action timeline, but the right panel is organized around assembly-state editing.

- Select **Assembly State (PSR/ASR/ASD)** in the Task dropdown to enter this mode.
- The label list automatically includes **Unobservable**.
- Use **Unobservable** for any span where the assembly state is not visible or cannot be reliably inferred.
- This label is a normal timeline label (it replaces the state label for that span).
- Load a component library and rule file:
  - **Choose action... -> Assembly State: Load Components...**
  - **Choose action... -> Assembly State: Load Rules...**
- To continue from an existing Assembly State Recognition result, use:
  - **Choose action... -> Assembly State: Load State JSON...**
- The PSR side panel is grouped into:
  - **Current Segment**: summary, component table, scope buttons, split, merge, reset, invert.
  - **Rules**: load, edit, apply, and export rules.
  - **Advanced**: learn rules from edits and batch-convert datasets.
- Fast editing controls in **Current Segment**:
  - **This Segment**: edit only the selected segment.
  - **From Here**: edit the selected segment and all following segments.
  - **Split**: split the selected segment at the current playhead frame.
  - **Merge**: merge adjacent identical state segments.
  - **Reset**: restore the selected segment to the rule-derived state.
  - **Invert**: toggle Installed <-> Not installed for the selected scope.
  - **🔁**: restore selected segment to rule-derived state.
  - **⇄**: Installed <-> Not installed for selected scope.
- Model selection is loaded from `configs/psr_models.json`.
- If two or fewer enabled models are configured, the PSR panel shows compact buttons.
- If more than two enabled models are configured, or the current file references an unknown model, the selector switches to a dropdown automatically.
- Default PSR shortcuts:
  - `Ctrl+K` split at playhead
  - `Ctrl+Shift+S` set scope to segment
  - `Ctrl+Shift+F` set scope to from-here
  - `Ctrl+Backspace` reset selected segment
  - `Ctrl+I` invert selected segment
  - `Ctrl+M` merge identical neighbors
- Components are fixed to the HAS catalog (16 parts + M6_nut); loading a component
  file is treated as catalog verification.
- Rule trigger is the **segment end frame** (step completion).
- If a label contains `error`, the component is marked as **-1 (incorrect)**.
- Labels that do not match any component are ignored.
- Export uses the ASR JSON format (see File formats).
- The right-side panel shows the current assembly state for each component at the active frame.

### Saving annotations
- Save prompts (e.g., when closing/switching with unsaved changes) use the built-in **Save JSON** flow.
- Save behavior:
  - **Single view loaded**: choose one JSON file path.
  - **Multiple views loaded**: choose one output root folder; the app creates one subfolder per view and writes one JSON per view.
- File names default to each view's source video base name.
- Frames are saved relative to each view's `view_start`.
- Validation log is saved as `<anchor>.validation.log.txt` (single summary log).
- If a write error occurs, recovery JSONs are written to `_recovery_YYYYMMDD_HHMMSS` under the target folder.
- The terminal prints `[SAVE] ...` lines with final saved paths.

### Importing annotations
- Use one of:
  - **Choose action... -> Import JSON (fill empty views)...**
  - **Choose action... -> Import JSON (current view)...**
- Auto-detects supported formats (Native, ActivityNet, FrameTXT, FACT, OurV1).
- If the JSON contains `entity` fields, the app switches to Fine mode automatically.
- In multi-view setups:
  - **fill empty views** updates the active view plus any view that currently has no annotations.
  - **current view** updates only the active view.
- If annotations extend beyond the current crop range, the app warns and clamps them.

### Import/Export label map (TXT)
- Format: `LabelName LabelId` (one entry per line).
- Import replaces the current label list.
- Export writes labels in ascending id order.
- The tool now ships with built-in default label templates for both **Coarse** and **Fine** modes.
  - On a fresh start, **Coarse** mode is seeded with the built-in coarse template.
  - If you switch between **Coarse** and **Fine** before loading or creating annotations, the tool can automatically swap between the built-in mode-specific templates.
  - This automatic swap is intentionally conservative: it only happens while the current labels still match one of the built-in templates and there are no real annotations in the timeline stores.
  - If you import your own label map, rename labels, or start annotating, the app will keep your current labels and will not overwrite them on mode switch.

---

## Validation and review (Action Segmentation)

### Enter/exit Validation
- Toggle **Validation** on the top bar.
- You must enter the editor name.
- Toggle again to exit validation.

### Logging behavior
- Every label change is recorded with frame ranges, view name, and editor.
- On save, a review log is written:
  - `<annotations>.validation.log.txt`
- The email content for the log is copied to the clipboard automatically.

### Review import
- Use **Choose action... -> Import Review Log...**.
- The Review panel appears with the change list.
- Click **Accept Change** or **Reject Change** to decide each change.
- **Left/Right arrows** move through review items.

---

## Interaction modes (Action Segmentation)

### Manual Segmentation (Interaction label)
Purpose: quick global segmentation for the special `Interaction` label.

- Choose **Interaction -> Manual Segmentation (toggle)**.
- Click inside the video to place a boundary; the span between boundaries is filled automatically.
- The Interaction spans are saved in a sidecar file when possible:
  - `<annotations>_extra.json`

### Assisted Review
Purpose: correct model-predicted boundaries/labels from EAST, ASOT, or legacy FACT outputs.

- Requires automatic segmentation output to be available.
- Choose **Interaction -> Assisted Review (toggle)**.
- Use the on-screen prompts and keyboard shortcuts to confirm or skip points.
- The review queue now uses an explicit query policy instead of pure timeline order.
  - Boundary points are prioritized by a weighted combination of boundary uncertainty, local boundary energy, and learned label-confusion evidence between the left/right segments.
  - Label points are prioritized by a weighted combination of label uncertainty, disagreement between the current label and the top candidate, and learned confusion memory from earlier confirmed corrections.
  - Points are ordered by priority buckets first and then by timeline position, so the queue stays readable without reducing everything to strict chronological order.
- The current query-policy snapshot is also exposed in **Choose action... -> EAST: Inspect Runtime Assets**.
  - The debug panel shows the active sort mode, review thresholds, queue size, label/boundary point counts, average and maximum query scores, and the current priority-bucket histogram.

Assisted Review shortcuts:
- **S** or **Down**: confirm boundary.
- **Left/Right**: nudge boundary (1 frame).
- **N / P**: next/previous uncertain point.
- **X**: skip point.
- **Backspace / Delete**: merge boundary (apply left label over right).

---

## Gap checking (Action Segmentation)

- Gaps are unlabeled frame ranges inside the crop.
- The gap summary shows counts per view and per entity in Fine mode.
- **< / >** buttons jump to the previous/next gap in the active view.
- Gap warnings appear on save and on close.

---

## Audio and Transcript Workspace

### Audio
- **Transcript Audio: Attach External Audio...**: attach a separate audio file for transcript generation.
- **Transcript Audio: Set Audio Offset (ms)...**: adjust transcript audio sync.
- Playback speed attempts to keep A/V synchronized. If audio rate change is unsupported, audio stays at 1x.

### Transcript Workspace
- This workspace is separate from **PSR/ASR/ASD**. Here, `transcript` means speech subtitles; in PSR, `ASR` still means **Assembly State Recognition**.
- Open it with **Choose action... -> Transcript: Open Workspace**.
- Use **Transcript: Quick Generate / Import...** for the old one-step path, or use the panel buttons for a cleaner workflow:
  - **Generate / Import**: generate subtitles with `ASR_CMD`, or load transcript JSON when `ASR_CMD` is not configured.
  - **Apply to Label**: convert transcript spans into action intervals on the active view.
  - **Clear**: remove transcript cues from the timeline.
- The right panel shows:
  - a compact summary (`cues | language | span | max gap`)
  - a current-cue card that follows the playhead
  - **Prev / Current / Next** cue navigation
  - the cue list for direct time jumps

### Transcript Mock Data
- A ready-to-import mock transcript is included at [test_data/transcript_workspace_mock.json](./test_data/transcript_workspace_mock.json).
- It contains 12 English assembly cues across about 28 seconds, with short pauses between steps so you can test:
  - current-cue tracking
  - gap display between cues
  - cue navigation
  - transcript-to-label conversion
- Recommended test flow:
  1. Load any video clip that is at least 30 seconds long.
  2. Open **Transcript Workspace**.
  3. Use **Generate / Import**.
  4. If `ASR_CMD` is not set, choose `test_data/transcript_workspace_mock.json`.
  5. Scrub the playhead and verify the current-cue card and cue list update together.
  6. Use **Apply to Label** to test transcript-assisted segmentation on the active view.

---

## HandOI / HOI Detection

### Top bar controls
- **Task**: switch tasks.
- **Playback**: rewind/play/pause/stop/fast-forward.
- **Jump**: go to a specific frame.
- **Start**: align external frame 0 to this video frame (offset for imports).
- **End**: optional maximum frame; imported boxes outside this range are ignored.
- **Files**: all HOI I/O actions.
- **Detect**: runs YOLO + MediaPipe on the current frame.
- **Auto swap L/R**: swaps MediaPipe Left/Right labels for mirrored video (new detections only).
- **Edit boxes**: enables interactive bbox editing.
- **Validation**: toggle HOI validation mode.

### Files menu
- Load Video...
- Import Instrument List...
- Import Target List...
- Load Class Map (data.yaml)...
- Import YOLO Boxes...
- Load YOLO Model...
- Detect Current Frame
- Load Hands XML...
- Import Verb List...
- Load HOI Annotations...
- Save HOI Annotations...
- Export Hands XML...

### Entity Library (Instrument / Target lists)
- Import instrument and target lists from TXT files.
- Format: `category,count` per line, comma-separated.
  - Example:
    - `screwdriver,3`
    - This creates `screwdriver_1`, `screwdriver_2`, `screwdriver_3` with unique ids.
- Entities are shown as `[id] name` in the Instrument/Target combos.
- The combos are searchable.

### Object list (current frame)
- The right panel shows all bboxes on the current frame.
- Selecting an object assigns it to the currently selected hand:
  - First selection sets Instrument.
  - Second selection sets Target.
- Right-click an object to:
  - **Change ID / Propagate...** (optional IoU-based propagation)
  - **Delete Box**

### Verb panel
- Add, rename, delete verbs.
- Import verb list (txt): one verb per line; ids are auto-assigned.
- Select a color for a verb; the timeline updates to match.

### Hand selection
- Use **Hands: L / R** to choose which hand you are editing.
- Only one hand can be active at a time.
- **Swap** swaps left/right hand boxes on the current frame only.

### HandOI timeline
- Two rows: Left hand and Right hand.
- Create a segment by dragging in a row.
- Resize by dragging segment edges.
- Move by dragging the segment body.
- Adjust onset by dragging the onset marker inside the segment.
- Right-click a segment to delete it.
- Segments cannot overlap within the same row.
- The left gutter shows the selected event summary:
  - `L/R` + check mark + Verb/Tool/Target.
- View Start/Span sliders control the visible timeline window.

### Anomaly labels
- Single-select list with **Normal** as default.
- Clicking anywhere on a label toggles it.
- **Rules** opens a dialog where each anomaly can allow missing verb or missing bbox.
- Missing fields are checked on save and when using the Incomplete indicator.

### Edit boxes (bbox editing)
When **Edit boxes** is enabled:
- **Drag** a box to move it.
- **Drag handles** (8 points) to resize.
- **Ctrl + left-drag**: draw a new box.
- **Right-click** a box: delete.
- **Double-click** a box: rename by id or class name.
  - If you enter a new numeric id, you will be prompted for its label.

### Detect (YOLO + MediaPipe)
- Requires: data.yaml + Instrument/Target lists + YOLO model.
- If boxes already exist on the frame, you are asked to **Append** or **Replace** (optional remember).
- MediaPipe hands are added every time you detect.

### Validation mode (HOI)
- Toggle **Validation** on the top bar.
- Enter editor name when prompted.
- Validation overlays highlight relations by drawing lines between hand, instrument, and target boxes.
- A validation summary is written on save when `validation_summary_enabled` is enabled:
  - `<annotations>.validation.json`
- Validation ops log is written when `ops_csv_enabled` is enabled:
  - `<annotations>.validation.ops.log.csv`

### Incomplete indicator (HOI)
- Bottom right of the timeline: **Incomplete: N** with < and > buttons.
- Jumps to the onset frame of each incomplete entry.
- Uses anomaly rules to decide whether missing verb/bbox is allowed.

---

## File formats

### Action Segmentation native JSON
Written by **Export JSON...** when format is **Native**.
```
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
  "action_labels": [
    {"id": 3, "name": "pick_gear"}
  ],
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
- `entity` appears in Fine mode segments.
- `phase` and `anomaly_type` are used in Fine phase/anomaly workflows.
- Older JSON without `meta_data` is still loadable; new saves always include `meta_data`.

### Label map TXT
```
label_name label_id
```
Example:
```
pick 3
place 4
```

### Interaction sidecar
- Saved as `<annotations>_extra.json`.
- Contains only the `Interaction` spans.

### Adapter formats
The **Import/Export annotations (JSON)** actions support these formats:
- Native
- ActivityNet
- FrameTXT
- FACT
- OurV1

### PSR/ASR/ASD component library (JSON/YAML)
```
{
  "components": [
    {"id": 0, "name": "base"},
    {"id": 1, "name": "front_chassis"},
    {"id": 2, "name": "front_chassis_pin"}
  ]
}
```
Notes:
- If this file is not loaded, components are auto-extracted from coarse label nouns.
- Component order defines the state vector index.

### PSR/ASR/ASD component library (TXT)
```
0, base
1, front_chassis
2, front_chassis_pin
```
Notes:
- Comma is optional; whitespace separation is also accepted.

### PSR/ASR/ASD rules file (JSON/YAML)
```
{
  "rules": [
    {
      "label": "Install front chassis",
      "components": [{"component": "front_chassis"}],
      "state": 1
    },
    {
      "label": "Install sub-assembly",
      "components": [{"component": "front_chassis"}, {"component": "front_chassis_pin"}],
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
- Trigger is the **segment end frame**.
- If a label contains `error`, the state defaults to `-1` when not specified.
- Labels that do not match any component are ignored.

### ASR output JSON (derived)
```
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
  "state_sequence": [
    {"frame": 0, "state": [0]},
    {"frame": 240, "state": [1]}
  ],
  "state_changes": [
    {"frame": 240, "component_id": 0, "state": 1}
  ]
}
```
Notes:
- Frame indices are **relative to `view_start`**.
- `video_id` is a canonical video stem shared across views (view suffixes like `front/top/...` are stripped).
- `meta_data.workflow` is derived from initial state (`Installed -> disassemble`, `Not installed -> assemble`).
- `meta_data.model_type` comes from the PSR model registry in `configs/psr_models.json`.
- Component order is the array order in `components` (no separate `component_order` field).
- `state_sequence` is the authoritative timeline state.
- `state_changes` is an optional sparse event list (kept for debugging/step derivation and backward compatibility).
- State vector length is always equal to `len(components)`.

### HOI JSON (HOI-1.0-ActionSeg)
```
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
  "anomaly_rules": {"Normal": {"allow_missing_bbox": false, "allow_missing_verb": false}},
  "tracks": {
    "T_LHAND": {"category": "left_hand", "object_id": null, "boxes": [{"frame": 10, "bbox": [x1,y1,x2,y2]}]},
    "T_RHAND": {"category": "right_hand", "object_id": null, "boxes": [...]},
    "T_OBJ_12": {"category": "screwdriver_1", "object_id": 12, "class_id": 4, "boxes": [...]}
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
- Object tracks store boxes at keyframes (start/onset/end and other event keyframes).
- Missing verb/bbox can be allowed per anomaly rule.

### Validation logs
- Action Segmentation: `<annotations>.validation.log.txt`
- HOI: `<annotations>.validation.json`

---

## Operation logs (oplog)

Enable with:
```
python app.py --oplog
```

Files created on save/export (depending on logging toggles):
- `ops_csv_enabled`:
  - Action Segmentation: `<annotations>.ops.log.csv`
  - HOI: `<annotations>.ops.log.csv`
  - HOI Validation (if validation is active): `<annotations>.validation.ops.log.csv`
- `validation_summary_enabled`:
  - Action Segmentation: `<annotations>.validation.log.txt`
  - HOI Validation: `<annotations>.validation.json`
- Annotation save/export is independent from logging: if log writing fails, annotation files still succeed.

---

## Keyboard shortcuts

All shortcuts below are **default bindings**. They can be changed in **Settings (⚙) -> Keyboard Controls**.

### Action Segmentation (defaults)
- Space: Play/Pause
- A / D: Step -1 / +1 frame
- Shift+A / Shift+D: Step -10 / +10 frames
- J / K / L: Seek -1 second / Pause / +1 second
- Home / End: Jump to crop start / crop end
- Ctrl+Z / Ctrl+Y: Undo / Redo
- Ctrl+Shift+U: Adjust uncertainty margin
- Ctrl+, : Open Settings

### Action Review / Assisted Review (defaults)
- Review: Left / Right (previous / next review item)
- Assisted: Left / Right nudge boundary
- Assisted: S or Down confirm boundary
- Assisted: N / P next / previous point
- Assisted: X skip point
- Assisted: Backspace / Delete merge boundary

### HandOI / HOI Detection (defaults)
- Left / Right: step -1 / +1 frame
- Up / Down: seek -1 / +1 second
- Space / K: Play-Pause / Pause
- Ctrl+Shift+D: Detect current frame
- Ctrl+Z / Ctrl+Y: Undo / Redo

### PSR/ASR/ASD (defaults)
- Ctrl+Z / Ctrl+Y: Undo / Redo

---

## Troubleshooting

- **Audio speed warnings**: some devices do not support audio rate changes. The app keeps audio at 1x and continues video playback at the requested speed.
- **No detections**: make sure `data.yaml`, Instrument/Target lists, and a YOLO model are loaded before Detect.
- **Boxes outside frame**: check Start/End alignment and verify YOLO labels match the video frame numbering.
- **Only one hand detected**: check MediaPipe settings or try toggling Auto swap L/R if the video is mirrored.
- **Annotations not visible after load**: confirm crop start/end and view span; the timeline uses cropped frame indices.

---

## Task placeholders
The Task dropdown includes placeholders for:
- Single-turn VQA
- Multi-turn VQA
- Video Captioning
These are not implemented in the current build.
