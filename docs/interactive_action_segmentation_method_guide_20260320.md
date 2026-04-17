# Interactive Action Segmentation Method Guide

Updated: 2026-03-20

This document summarizes the method-level changes added to the Action Segmentation workflow, why they were added, and where they live in the codebase.

It is intended for collaborators who need to understand the current interactive training paradigm without reverse-engineering the full UI code.

## 1. High-Level Goal

The current workflow is no longer just:

- model predicts
- annotator fixes
- export labels

It is now:

1. generate a proposal with `ASOT` or `EAST`
2. let the annotator confirm or correct the proposal
3. convert only the final confirmed difference into structured supervision
4. update a lightweight EAST corrective layer online
5. preserve confirmed regions and refresh only unlocked regions
6. accumulate video-level knowledge into a shared adapter
7. use finalized videos as the stronger second timescale of training

The core design principle is:

- proposals are cheap
- confirmed supervision is valuable
- untouched auto output is never treated as GT

## 2. Proposal Sources and Training Eligibility

### Proposal sources

- `ASOT Pre-label`
  - cold-start semantic proposal
  - cluster/semantic remap may be applied
- `EAST Refine`
  - main proposal and refinement model

### Training eligibility

Training is no longer gated purely by proposal source.

Instead it is gated by **human confirmation status**:

- `accepted`
  - the annotator explicitly reviewed a proposal and accepted it
- `corrected`
  - the annotator changed the label and/or boundary
- `finalized`
  - the video is explicitly marked as finalized and can act as GT-strength supervision

Untouched auto proposals do not enter EAST training.

Main code:

- [ui/action_window.py](../ui/action_window.py)
- [core/east_online_update.py](../core/east_online_update.py)

## 3. Correction Buffer v2

The tool does not train from raw UI event logs.

It compares:

- the current confirmed annotation state
- against the current baseline

and writes structured correction records.

Supported record types include:

- `label_accept`
- `label_assign`
- `label_replace`
- `label_remove`
- `label_finalize`
- `boundary_accept`
- `boundary_add`
- `boundary_move`
- `boundary_remove`
- `boundary_finalize`
- `transition_add`
- `transition_remove`
- `segment_lock`

Main code:

- [ui/action_window.py](../ui/action_window.py)
- [core/action_corrections.py](../core/action_corrections.py)

## 4. Layered Supervision Strength

Supervision is now intentionally hierarchical.

### Current policy

- `accepted`
  - weaker positive supervision
- `corrected`
  - stronger positive supervision
  - may generate hard negatives and ranking supervision
- `finalized`
  - strongest supervision
  - used by the offline second-timescale consolidation path

### Quality-aware weighting

In addition to the supervision kind, each correction record now gets a mild quality factor based on:

- span coverage
- edit magnitude
- hard-negative richness
- finalized-video context

The goal is not to drastically change optimization, but to reflect that:

- a large corrected span is usually more informative than a tiny accepted span
- a label replacement carries more supervision than a passive acceptance
- finalized videos should contribute more strongly during consolidation

Main code:

- [core/east_online_update.py](../core/east_online_update.py)

Key helpers:

- `_confirmed_kind(...)`
- `_record_quality_factor(...)`
- `_supervision_weight(...)`

## 5. Accepted vs Corrected vs Finalized Objectives

We now separate the role of each supervision type more explicitly.

### Accepted

`accepted` records are used mainly for:

- positive label alignment
- positive boundary calibration

They do **not** carry strong ranking / negative-label penalties by default.

### Corrected

`corrected` records are used for:

- positive label alignment
- hard negatives
- pairwise ranking loss
- boundary add/remove/move supervision
- confusion-memory updates

### Finalized

`finalized` records behave like the strongest version of corrected supervision and are also used in the offline shared-adapter training path.

Main code:

- [core/east_online_adapter_train.py](../core/east_online_adapter_train.py)

Key point:

`_segment_rows_from_records(...)` now avoids generating ranking/negative supervision from plain `accepted` label records.

## 6. Label-Segment Alignment

Label assignment is no longer treated as a single classifier output.

The current reranking stack combines:

- base EAST label scores
- text prior (`label_text_bank.pt`)
- learned class table (`online_adapter.pt`)
- confirmed visual prototypes (`model_delta.pt`)
- transition prior

Main code:

- [tools/east/east_infer_adapter.py](../tools/east/east_infer_adapter.py)
- [core/label_text_bank.py](../core/label_text_bank.py)
- [core/east_online_update.py](../core/east_online_update.py)
- [core/east_online_adapter_train.py](../core/east_online_adapter_train.py)

## 7. Fusion Gate

Previously label fusion used fixed global weights.

It now supports a lightweight learned gate that predicts how much to trust each source for the current segment.

### Gate inputs

The gate uses compact per-segment features:

- base confidence
- base margin
- base entropy
- normalized segment length
- whether text prior is available
- whether class-table prior is available
- whether prototype prior is available

### Gate outputs

It predicts per-segment mixture weights over:

- `base`
- `text_prior`
- `class_table`
- `prototype`

This lets the system adaptively shift trust:

- some segments are better handled by the base model
- some benefit more from text prior
- some benefit more from prototypes or online class adaptation

Main code:

- [core/east_label_fusion.py](../core/east_label_fusion.py)
- [core/east_online_adapter_train.py](../core/east_online_adapter_train.py)
- [tools/east/east_infer_adapter.py](../tools/east/east_infer_adapter.py)

## 8. Hard Negatives and Confusion Memory

The system now uses two levels of label negatives.

### Local hard negatives

Derived from the current correction:

- replaced old label
- overlapping top-k model candidates that were not accepted

### Persistent confusion memory

Accumulated across corrections:

- for each positive label, the system remembers which mistaken labels repeatedly appeared as confusions

This supports better ranking-based learning over time.

Main code:

- [ui/action_window.py](../ui/action_window.py)
- [core/east_online_update.py](../core/east_online_update.py)
- [core/east_online_adapter_train.py](../core/east_online_adapter_train.py)

## 9. Query Policy

Assisted Review is no longer just threshold-based uncertainty filtering.

The queue now uses an explicit query score.

### Boundary query terms

- uncertainty
- label confusion
- local boundary energy
- same-label merge bonus
- training utility

### Label query terms

- uncertainty
- disagreement with current label
- confusion memory
- training utility

### Training utility term

The recent update adds a small expected-training-utility component.

The idea is:

- if reviewing this point is likely to improve the model more, it should be prioritized higher

Current low-risk approximation uses:

- label scarcity
- hard-negative potential
- confusion strength

This is intentionally conservative: it improves queue quality without introducing a heavy learned acquisition policy.

Main code:

- [ui/action_window.py](../ui/action_window.py)

## 10. Locked Segments and Masked Refresh

After confirmed supervision, the system does not blindly rerun full refinement and overwrite the whole video.

Instead:

- directly confirmed relabeled segments become locks
- confirmed human boundaries become protected anchors
- post-update refresh only rewrites unlocked windows with context

This keeps the annotator from repeatedly fighting the model on already-fixed spans.

Main code:

- [ui/action_window.py](../ui/action_window.py)
- [ui/action_workers.py](../ui/action_workers.py)
- [tools/east/east_infer_adapter.py](../tools/east/east_infer_adapter.py)

## 11. ASOT to EAST Bootstrap

Videos can start from ASOT.

If a video only has ASOT proposals but accumulates confirmed supervision, the tool can bootstrap an EAST runtime in the background:

- initialize EAST features/runtime
- write confirmed supervision into EAST runtime assets
- start online EAST learning

Important:

- untouched ASOT output still does not become EAST supervision
- only confirmed human supervision crosses the bridge

Main code:

- [ui/action_window.py](../ui/action_window.py)
- [ui/action_workers.py](../ui/action_workers.py)

## 12. Finalized Video and Second-Timescale Training

This is the main second-timescale extension.

### Before

Shared adapter consolidation mostly merged runtime assets by weighted averaging.

### Now

If member runtimes are finalized:

- consolidation can train a shared `online_adapter` offline from finalized supervision
- this produces a stronger shared adapter than simple averaging alone
- the resulting shared adapter can also carry a trained fusion gate

This is the intended second timescale:

- online timescale
  - accepted/corrected interactive updates inside a video
- offline timescale
  - finalized videos drive shared-adapter training across videos

Main code:

- [core/east_shared_assets.py](../core/east_shared_assets.py)
- [core/east_online_adapter_train.py](../core/east_online_adapter_train.py)

Relevant UI entry:

- `Choose action... -> EAST: Finalize Current Video`

## 13. Files You Should Know

### Core method files

- [ui/action_window.py](../ui/action_window.py)
- [ui/action_workers.py](../ui/action_workers.py)
- [core/east_online_update.py](../core/east_online_update.py)
- [core/east_online_adapter_train.py](../core/east_online_adapter_train.py)
- [core/east_label_fusion.py](../core/east_label_fusion.py)
- [core/east_shared_assets.py](../core/east_shared_assets.py)
- [tools/east/east_infer_adapter.py](../tools/east/east_infer_adapter.py)

### Runtime assets

Inside each video's `east_runtime/`:

- `segments.json`
- `boundary.npy`
- `label_scores.npy`
- `seg_embeds.npy`
- `transition.npy`
- `prototype.npy`
- `record_buffer.pkl`
- `finalized_record_buffer.pkl`
- `model_delta.pt`
- `online_adapter.pt`
- `label_text_bank.pt`
- `meta.json`

## 14. Practical Interpretation

For collaborators, the simplest way to think about the current method is:

- `ASOT` and `EAST` generate proposals
- humans convert proposals into supervision
- supervision is not binary; it has levels
- label alignment is multimodal and adaptive
- confirmed regions are preserved
- finalized videos provide the stronger second timescale

That is the current intended training paradigm.
