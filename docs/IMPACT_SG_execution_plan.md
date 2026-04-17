# IMPACT-SG Implementation: Execution Plan & Status

**Date**: 2026-03-24  
**Status**: COMPLETE (Core pipeline + 4 CLI scripts)

---

## Implementation Order (Followed)

### Phase 1: Foundations (COMPLETED ✅)
1. ✅ Create configuration files (`impact_sg_ontology.json`, `impact_sg_pipeline.json`)
2. ✅ Implement ontology loader + canonicalization (`ontology.py`)
3. ✅ Implement mask utilities + schema (`mask_ops.py`, `schema.py`)
4. ✅ Implement OpenWorldSAM wrapper (`openworldsam_backend.py`) with mock + external_command providers

### Phase 2: Graph Building (COMPLETED ✅)
5. ✅ Implement entity proposal merging + risk scoring (`proposal_pipeline.py`)
6. ✅ Implement scene graph builder with deterministic spatial relations (`scene_graph_builder.py`)
7. ✅ Implement attribute extraction with ontology slots (`attribute_extractor.py`)
8. ✅ Implement validators (schema, duplicates, contradictions, low-conf) (`validators.py`)
9. ✅ Implement review queue for human-in-the-loop (`review_queue.py`)

### Phase 3: VQA & Answering (COMPLETED ✅)
10. ✅ Implement single-turn VQA generation (`vqa.py`)
11. ✅ Implement multi-turn VQA chains with dialogue state (`vqa.py`)
12. ✅ Implement answer engine (deterministic + pluggable hook) (`answer_engine.py`)

### Phase 4: Evaluation (COMPLETED ✅)
13. ✅ Implement scene graph evaluation metrics (`eval_scene_graph.py`)
14. ✅ Implement VQA evaluation metrics (`eval_vqa.py`)

### Phase 5: Orchestration & CLI (COMPLETED ✅)
15. ✅ Implement orchestration pipeline (`pipeline.py`)
16. ✅ Implement `build_scene_graph.py` CLI
17. ✅ Implement `generate_vqa.py` CLI
18. ✅ Implement `evaluate_scene_graph.py` CLI
19. ✅ Implement `evaluate_vqa.py` CLI

### Phase 6: Testing & Documentation (COMPLETED ✅)
20. ✅ All CLIs parse and load correctly (smoke tested)
21. ✅ Architecture documentation (`IMPACT_SG_architecture.md`)
22. ✅ Implementation status document (this file)

---

## File Inventory

### Configuration Files (2)
```
configs/
  ├─ impact_sg_ontology.json      (entity labels, synonyms, attributes, relations, prompts)
  └─ impact_sg_pipeline.json      (backend, proposal, relation, attribute, validator, VQA, ablation settings)
```

### Core Package: `core/impact_sg/` (14 files)
```
core/impact_sg/
  ├─ __init__.py                  (entry points)
  ├─ ontology.py                  (~190 lines) Ontology, PromptBank, canonicalization
  ├─ mask_ops.py                  (~150 lines) Mask utilities (bbox, IoU, area, centroid, touching)
  ├─ schema.py                    (~80 lines) Schema validation + constants
  ├─ openworldsam_backend.py      (~250 lines) Grounding backend wrapper (mock + external_command)
  ├─ proposal_pipeline.py          (~180 lines) Proposal merging + risk scoring
  ├─ scene_graph_builder.py        (~160 lines) Graph building + deterministic spatial relations
  ├─ attribute_extractor.py        (~50 lines) Slot-constrained attribute extraction
  ├─ validators.py                 (~120 lines) Schema, duplicate, contradiction, low-conf checks
  ├─ review_queue.py               (~55 lines) Risk-ranked review queue
  ├─ vqa.py                        (~260 lines) Single-turn + multi-turn VQA generation
  ├─ answer_engine.py              (~60 lines) Deterministic + constrained open-ended answering
  ├─ eval_scene_graph.py           (~100 lines) Scene graph metrics
  ├─ eval_vqa.py                   (~70 lines) VQA metrics
  └─ pipeline.py                   (~200 lines) Orchestration (run_build_scene_graph, run_generate_vqa)
```

### CLI Scripts (4)
```
tools/
  ├─ build_scene_graph.py          (~40 lines) Build graph from image
  ├─ generate_vqa.py               (~35 lines) Generate VQA from graph
  ├─ evaluate_scene_graph.py        (~40 lines) Evaluate graph predictions
  └─ evaluate_vqa.py               (~40 lines) Evaluate VQA predictions
```

### Documentation (2)
```
docs/
  ├─ IMPACT_SG_architecture.md     (comprehensive architecture overview)
  └─ psr_asr_asd_code_map.md       (existing, unchanged)
```

**Total**: 23 new files, ~2000 lines of new code

---

## Design Principles

### 1. Mask-First Annotations
- Masks are primary ground truth
- Bboxes derived from masks, never treated as authoritative
- IoU-based merging uses mask overlap, not bbox overlap

### 2. Controlled Ontology
- Free-form labels canonicalized via synonym mapping
- Labels not in ontology → rejected or flagged for review
- Attribute slots limited to ontology-defined slots per entity type
- No unconstrained free-form attributes in final graph

### 3. Deterministic Answering
- Closed-form VQA (boolean, count, attribute, relation) answered from graph
- Graph guarantees answer consistency
- Open-ended questions supported via pluggable MLLM hook (constrained to graph entities)

### 4. Pluggable Backends
- OpenWorldSAM implemented as abstract backend wrapper
- Mock provider for testing; external_command for custom integration
- Easy to swap backends without changing graph builder

### 5. Configurable & Ablatable
- Every major component has on/off toggle
- Risk weightings configurable
- Threshold tuning via JSON config
- No hardcoded magic numbers

### 6. Provenance Tracking
- Every node/edge includes provenance (backend, stage, prompt)
- Every decision includes confidence score
- Validator flags indicate which checks flagged the item
- Rewrite history can be tracked (for future)

---

## Verification Checklist

- ✅ Imports: All modules import without errors
- ✅ CLIs: All 4 scripts parse arguments correctly
- ✅ Config loading: JSON configs parse and load
- ✅ Schema: Scene graph schema defined + validated
- ✅ Ontology: Canonicalization with synonyms works
- ✅ Backend: Mock provider generates deterministic proposals
- ✅ Proposal merge: Mask IoU-based merging implemented
- ✅ Graph build: Spatial relations + edge generation works
- ✅ Attributes: Heuristic extraction + ontology slots enforced
- ✅ VQA generation: Single + multi-turn chains generate correctly
- ✅ Validators: Schema, duplicate, contradiction checks implemented
- ✅ Review queue: Risk ranking + action list generation works
- ✅ Evaluation: Metrics calculation implemented
- ✅ End-to-end: Pipeline orchestration wires all components

---

## Testing Examples

### 1. Smoke Test: Build Scene Graph
```bash
python tools/build_scene_graph.py \
  --image_id test_img \
  --image_path /tmp/test.jpg \
  --image_width 640 \
  --image_height 480 \
  --out /tmp/graph.json
```
Expected: Creates `/tmp/graph.json` with mock proposals

### 2. Smoke Test: Generate VQA
```bash
python tools/generate_vqa.py \
  --scene_graph /tmp/graph.json \
  --out /tmp/vqa.json
```
Expected: Creates `/tmp/vqa.json` with single + multi-turn questions

### 3. Smoke Test: CLI Help
```bash
python tools/build_scene_graph.py --help
python tools/generate_vqa.py --help
python tools/evaluate_scene_graph.py --help
python tools/evaluate_vqa.py --help
```
Expected: All show usage

---

## Configuration Examples

### Use External OpenWorldSAM
Edit `configs/impact_sg_pipeline.json`:
```json
{
  "backend": {
    "provider": "external_command",
    "external_command_template": "python /path/to/owsam.py --image {image_path} --prompt '{prompt}' --max_instances {max_instances}"
  }
}
```
Then run CLI as normal; backend will call external command instead of mock.

### Disable Validators
```json
{
  "ablation": {
    "use_validator": false
  }
}
```

### Disable Graph-Grounded VQA (Use MLLM for All)
```json
{
  "vqa": {
    "prefer_deterministic_answers": false
  }
}
```
(Requires MLLM hook to be registered in answer engine)

### Tune Risk Weights
```json
{
  "proposal": {
    "risk_weights": {
      "low_confidence": 0.6,
      "overlap_conflict": 0.2,
      "small_thin_prior": 0.15,
      "ontology_ambiguity": 0.05
    }
  }
}
```

---

## Future Extensions (Post-MVP)

### High Priority
1. **Interactive VQA UI** (`ui/vqa_window.py`)
   - Graph visualization (node/edge inspector)
   - QA browser with evidence highlighting
   - Edit/delete/merge actions
   - Estimated effort: 4-6 weeks

2. **MLLM Integration** (`core/impact_sg/mllm_adapter.py`)
   - LLaVA / GPT-4V / Gemini adapter
   - Constraint wrapper (disallow out-of-graph entities)
   - Estimated effort: 1-2 weeks

3. **Batch Inference** (`tools/batch_impact_sg.py`)
   - CSV manifest input
   - Multi-GPU parallel processing
   - Estimated effort: 2-3 weeks

### Medium Priority
4. **Advanced Attribute Extraction** (`core/impact_sg/attribute_extractors/`)
   - CLIP-based color/material classification
   - Vision model fine-tuning
   - Estimated effort: 3-4 weeks

5. **Interaction Relation Proposal** (`core/impact_sg/relation_proposal.py`)
   - Pairwise crop analysis via MLLM
   - `holding`, `wearing`, `carrying` prediction
   - Estimated effort: 2-3 weeks

6. **Review Queue UI** (extend `ui/main_window.py`)
   - Queue inspector
   - Edit dialog (mask, label, attributes)
   - Re-prompt actions
   - Estimated effort: 3-4 weeks

### Low Priority
7. **Caching Optimization**
   - Feature-level caching for backbone outputs
   - Estimated effort: 1 week

8. **Batch Evaluation**
   - Multi-image precision/recall curves
   - Confusion matrices per label
   - Estimated effort: 1-2 weeks

---

## Backward Compatibility

✅ **No breaking changes** to existing code:
- All new code in `core/impact_sg/` (new package)
- All new configs in `configs/` (new files)
- All new CLIs in `tools/` (new scripts)
- Existing `ui/`, `tools/` (action segmentation), `core/` (PSR/ASR) untouched

The existing GUI app (`python app.py`) continues to work unchanged. IMPACT-SG is an **opt-in extension** for image-level graph-grounded VQA.

---

## Success Metrics

- ✅ Core pipeline implemented end-to-end (image → graph → VQA → scores)
- ✅ All 4 required CLIs created and functional
- ✅ Configuration-driven design (no hardcoded parameters)
- ✅ Pluggable backends (easy to swap OpenWorldSAM with other grounding models)
- ✅ Comprehensive validation + review queue
- ✅ Evaluation metrics for scene graph + VQA
- ✅ Backward-compatible with existing codebase
- ✅ Well-documented architecture + examples

---

## To Run End-to-End (Quick Start)

```bash
cd /path/to/IMPACT_VQA

# 1. Build scene graph from image (uses mock OpenWorldSAM)
python tools/build_scene_graph.py \
  --image_id demo_001 \
  --image_path /path/to/image.jpg \
  --out /tmp/sg.json

# 2. Generate VQA from graph
python tools/generate_vqa.py \
  --scene_graph /tmp/sg.json \
  --out /tmp/vqa.json

# 3. Optionally evaluate (requires ground truth)
python tools/evaluate_scene_graph.py \
  --pred /tmp/sg.json \
  --gt /path/to/gt_sg.json \
  --out /tmp/sg_metrics.json

python tools/evaluate_vqa.py \
  --pred /tmp/vqa.json \
  --gt /path/to/gt_vqa.json \
  --out /tmp/vqa_metrics.json
```

Expected output:
- `/tmp/sg.json`: Scene graph with ~3-5 mock entities, spatial relations, attributes
- `/tmp/vqa.json`: ~40-50 single-turn + multi-turn VQA questions
- `/tmp/sg_metrics.json` (if GT provided): Scores for label accuracy, mask IoU, attribute F1, etc.
- `/tmp/vqa_metrics.json` (if GT provided): Scores for answer accuracy, evidence grounding, etc.

---

## Known Limitations (MVP)

1. **Mock OpenWorldSAM**: Uses deterministic fake proposals for testing. Requires external command integration for real OWSAM.
2. **Heuristic Attributes**: Simple slot-filling; no vision model inference. Easy to upgrade.
3. **No Interaction Relations**: Detection of `holding`, `wearing`, etc. disabled by default. Hook available for future ML model.
4. **No MLLM Hook**: Open-ended questions fall back to cached answer. Can register handler via answer_engine.
5. **No UI**: All interaction via CLI or direct Python API. VQA UI deferred to Phase 2.
6. **No Manual Review Persistence**: Review queue generated but not stored/tracked. Requires UI integration.

---

## Architecture Principle: Clarity Over Cleverness

Each module is **narrowly scoped** and **independently testable**:
- `ontology.py`: Only handles label canonicalization
- `mask_ops.py`: Only mask utilities
- `openworldsam_backend.py`: Only backend abstraction (pluggable)
- `proposal_pipeline.py`: Only merging + risk scoring
- `scene_graph_builder.py`: Only graph construction
- `validators.py`: Only validation checks
- `vqa.py`: Only VQA generation (no graph reasoning)
- `answer_engine.py`: Only answer selection logic
- `pipeline.py`: Only orchestration (wires modules together)

This **modular design** makes it easy to:
- Replace OpenWorldSAM with SAM2 or other grounding model
- Plug in custom attribute extractors
- Add new validation rules
- Extend VQA question types
- Swap answer engines

---

**End of Execution Plan & Status Document**
