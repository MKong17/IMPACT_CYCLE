# PSR / ASR / ASD Code Map

This file is a maintenance index for the PSR/ASR/ASD page logic.
It is intended to make debugging and future refactor safer.

## Main Files

- `ui/action_window.py`: task lifecycle, timeline synchronization, state edit logic, import/export.
- `ui/psr_window.py`: right-side panel UI (component table + state comboboxes + buttons).
- `core/psr_state.py`: label/rule to component-state event derivation and state sequence/runs.
- `ui/timeline.py`: combined timeline interaction (drag/split/delete/select).

## Functional Zones (ActionWindow)

Use `rg "_psr_" ui/action_window.py` to jump quickly.

- **Mode entry/exit + panel wiring**
  - `set_psr_panel`
  - `enter_psr_mode` / `exit_psr_mode`
  - `_apply_psr_controls`
  - `_apply_psr_action_dropdown`
  - `_apply_psr_asr_panel`
  - `_apply_psr_left_panel`
  - `_apply_psr_timeline`

- **Timeline binding and state data cache**
  - `_psr_bind_timeline_changed`
  - `_psr_bind_timeline_toggle`
  - `_psr_recompute_cache`
  - `_psr_build_state_runs`
  - `_psr_refresh_state_timeline`

- **Selection and right panel synchronization**
  - `_psr_set_selected_segment`
  - `_psr_sync_selected_segment`
  - `_psr_restore_selected_segment_on_timeline`
  - `_psr_panel_frame`
  - `_psr_update_component_panel`

- **Editing and auto-carry logic**
  - `_psr_on_state_changed`
  - `_psr_on_state_timeline_changed`
  - `_psr_invert_selected_segment`
  - `_psr_on_state_segment_delete`
  - `_psr_on_state_segment_split`

- **Manual event model + undo/redo**
  - `_psr_set_manual_event`
  - `_psr_merge_events`
  - `_psr_snapshot` / `_psr_restore_snapshot`
  - `_psr_push_undo` / `_psr_undo` / `_psr_redo`

- **Import/export/batch**
  - `_psr_parse_action_json`
  - `_psr_build_export_payload`
  - `_export_psr_asr_asd`
  - `_psr_batch_convert`

## Interaction Entry Points

- Timeline segment selection -> `ActionWindow._on_timeline_segment_selected`
- Right panel combobox change -> `PSRWindow._emit_state_changed` -> `ActionWindow._psr_on_state_changed`
- Ctrl + left click split on timeline -> `CombinedTimelineRow` split handler -> `ActionWindow._psr_on_state_segment_split`

## Refactor Rule (Safety)

When refactoring PSR code, keep these invariants:

1. Selected segment must survive timeline rebuild.
2. Right panel edits target selected segment (if selected), not current playback frame.
3. Dragging timeline boundaries must not be interrupted by panel refresh.
4. Auto-carry only applies when next segment has no explicit manual override.
