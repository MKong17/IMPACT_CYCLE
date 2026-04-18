# GUI Operator Guide

This document keeps the operator-facing GUI notes out of the paper `README.md` while preserving the parts that are still useful for demos, qualitative inspection, and human-in-the-loop runs.

## Scope

This guide covers the current four-task video GUI:

- `Video Scene Graph`
- `Single-turn VQA`
- `Multi-turn VQA`
- `Video Captioning`

For the older Action Segmentation / HOI / PSR stack, see [archive/legacy_annotation_manual.md](archive/legacy_annotation_manual.md).

## Launch

Start the desktop app with:

```bash
python app.py
```

Enable operation logging with:

```bash
python app.py --oplog
```

## Current Task Model

When the app starts, the main window shows:

- one shared video player on the left
- one task-specific workspace on the right

The shared player lets you load a video once and then switch between the four task panels without changing the playback context.

## Shared Controls

### Playback and navigation

- Open a video with the folder button.
- Use the transport controls for `open`, `prev`, `play/pause`, `next`, and `stop`.
- Use the seek slider to scrub through frames.
- Use the frame input / current-frame actions when a task needs a specific frame index.

### Video viewer interactions

- `Ctrl + mouse wheel`: zoom around the cursor.
- Left-drag: pan while zoomed.
- `Center`: re-center without resetting zoom.
- Double-click: reset zoom and center.

### Layout

- The main window uses splitters.
- The left side is the shared player.
- The right side is the active task workspace.

## Task Workflows

### Video Scene Graph

Fastest path:

1. Load a video.
2. Switch to `Video Scene Graph`.
3. Set the frame you want to inspect.
4. Run the scene-graph build action for that frame.
5. Inspect `nodes` and `edges`.
6. Save the resulting graph JSON.

Current workspace behavior:

- frame-centric graph construction from the loaded video
- separate node and edge tables
- linked highlighting between related nodes and edges
- JSON preview and direct export

### Single-turn VQA

Fastest path:

1. Build or load a scene graph first.
2. Switch to `Single-turn VQA`.
3. Generate single-turn questions.
4. Inspect a selected question and its details.
5. Save the VQA JSON.

### Multi-turn VQA

Fastest path:

1. Build or load a scene graph first.
2. Switch to `Multi-turn VQA`.
3. Generate multi-turn chains.
4. Inspect the chain turn by turn.
5. Save the VQA JSON.

### Video Captioning

Fastest path:

1. Optionally build a scene graph first.
2. Switch to `Video Captioning`.
3. Set the start and end frame.
4. Choose a style.
5. Generate the caption.
6. Save the caption output.

Current captioning features:

- single-segment caption generation
- styles such as `Concise`, `Detailed`, and `Technical`
- batch segment list management
- batch export to `JSONL` and `TXT`

## Saving and Logging

The current paper-facing GUI can save task outputs directly from each panel. When `--oplog` is enabled, the app also writes operation-log data alongside saved outputs.

For retained notes on validation logs, operation logs, and historical output files, see [file_formats.md](file_formats.md).

## Related References

- [../README.md](../README.md)
- [IMPACT_SG_quick_reference.md](IMPACT_SG_quick_reference.md)
- [archive/legacy_annotation_manual.md](archive/legacy_annotation_manual.md)
