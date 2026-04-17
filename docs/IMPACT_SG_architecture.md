# IMPACT-SG Architecture: Current vs. Target Mapping

**Date**: 2026-03-24  
**Scope**: Graph-centric annotation pipeline with OpenWorldSAM grounding backend

---

## 1. Current Architecture Summary

### Existing System (Before IMPACT-SG)
The repository is a **GUI-centric video annotation suite** for:
- **Action Segmentation** (ASR): temporal action label annotation on video timeline
- **HandOI/HOI Detection**: hand-object and human-object interaction annotation
- **Assembly State (PSR/ASR/ASD)**: component-state-based assembly process annotation

**Key characteristics**:
- PyQt5 desktop GUI as primary interface
- Adapter pattern for import/export (canonical intermediate representation for segments)
- Model backends (EAST, ASOT, FACT) for temporal sequence modeling
- Feature extraction pipelines (EAST backbone, ResNet50, DINOv2)
- Validation/review queue for temporal segments (not yet for spatial graphs)
- Operation logging for audit trail

**NOT currently present**:
- Spatial entity detection or grounding
- Scene graph representation
- Image-level visual grounding (OpenWorldSAM or SAM-based)
- Multi-entity spatial/interaction relations
- Graph-grounded VQA
- Attribute prediction on detected objects
- Canonical entity ontology for object labels

---

## 2. Target IMPACT-SG Architecture

### High-Level Pipeline
```
Image → OpenWorldSAM (grounding backend)
      → Entity Proposals (category + sentence prompts)
      → Merge / Risk Scoring
      → Scene Graph Builder (spatial relations)
      → Attribute Extraction (ontology-constrained)
      → Validators (schema, contradictions, low-conf)
      → Review Queue (human-in-the-loop)
      ↓
Scene Graph (canonical representation)
      ↓
VQA Generator (single-turn + multi-turn chains)
      ↓
Answer Engine (graph answering + MLLM hook)
      ↓
Evaluation (scene graph metrics + VQA metrics)
```

### Core Modules (Now Implemented)

| Component | File | Purpose |
|-----------|------|---------|
| **Ontology** | `core/impact_sg/ontology.py` | Load controlled entity/relation vocabulary; canonicalize free-form labels via synonyms |
| **Mask Ops** | `core/impact_sg/mask_ops.py` | Mask-first utilities (bbox derivation, IoU, area, centroid) |
| **OpenWorldSAM Backend** | `core/impact_sg/openworldsam_backend.py` | Pluggable grounding wrapper (category discovery + sentence refinement); mock + external_command providers |
| **Proposal Pipeline** | `core/impact_sg/proposal_pipeline.py` | Merge duplicate proposals by mask IoU + canonical label; compute risk scores |
| **Scene Graph Builder** | `core/impact_sg/scene_graph_builder.py` | Build canonical JSON with nodes (entities) and edges (spatial/interaction relations) |
| **Schema** | `core/impact_sg/schema.py` | Schema validation and required field constants |
| **Attribute Extractor** | `core/impact_sg/attribute_extractor.py` | Heuristic extraction of ontology-defined slots per node |
| **Validators** | `core/impact_sg/validators.py` | Check schema, duplicates, contradictions, low confidence, missing mandatory attributes, answer-evidence mismatch |
| **Review Queue** | `core/impact_sg/review_queue.py` | Rank risky/flagged nodes/edges/QA items for human review |
| **VQA Generator** | `core/impact_sg/vqa.py` | Single-turn (existence, count, attribute, spatial, comparison) + multi-turn QA chains |
| **Answer Engine** | `core/impact_sg/answer_engine.py` | Deterministic graph answering + pluggable open-ended handler |
| **Eval Scene Graph** | `core/impact_sg/eval_scene_graph.py` | Metrics: label acc, mask/bbox IoU, attribute F1, relation F1, graph edit distance |
| **Eval VQA** | `core/impact_sg/eval_vqa.py` | Metrics: answer accuracy, evidence grounding, chain consistency, rewrite rate, mismatch rate |
| **Pipeline Orchestration** | `core/impact_sg/pipeline.py` | High-level orchestration (graph building + VQA generation) |

### Configuration Layer

| File | Purpose |
|------|---------|
| `configs/impact_sg_ontology.json` | Entity labels, synonyms, attribute slots, relation vocabulary, question types, prompt templates |
| `configs/impact_sg_pipeline.json` | Backend provider, proposal merging, risk weights, relation settings, attribute defaults, validator thresholds, VQA limits, ablation flags |

### CLI Entry Points (Executable Scripts)

| Script | Purpose | Usage |
|--------|---------|-------|
| `tools/build_scene_graph.py` | Build scene graph from image | `python tools/build_scene_graph.py --image_id IMG_001 --image_path image.jpg --out graph.json` |
| `tools/generate_vqa.py` | Generate VQA from scene graph | `python tools/generate_vqa.py --scene_graph graph.json --out vqa.json` |
| `tools/evaluate_scene_graph.py` | Compare pred vs. GT scene graph | `python tools/evaluate_scene_graph.py --pred pred.json --gt gt.json --out metrics.json` |
| `tools/evaluate_vqa.py` | Compare pred vs. GT VQA answers | `python tools/evaluate_vqa.py --pred pred.json --gt gt.json --out metrics.json` |

---

## 3. Module-to-Target Mapping

### A. Ontology + Prompt Bank ✅
- **Target**: Config-driven ontology with canonical labels, synonyms, attribute slots, relation vocabulary, question types
- **Implemented**:
  - `ontology.py`: `Ontology` class + `PromptBank` generation
  - `impact_sg_ontology.json`: Controlled vocabulary
  - Canonicalization logic with fallback to free-form rejection

### B. OpenWorldSAM Backend ✅
- **Target**: Category-level + sentence-level prompting; configurable max instances; two-stage refinement toggle; caching
- **Implemented**:
  - `openworldsam_backend.py`: `OpenWorldSAMBackend` class
  - Providers: `mock` (test), `external_command` (custom integration)
  - Caching for image features + prompt results
  - Output: mask, bbox (from mask), score, prompt_used, stage, metadata

### C. Entity Proposal & Merging ✅
- **Target**: Merge duplicates by mask IoU + canonical label; risk score from confidence, overlap, size, ontology ambiguity
- **Implemented**:
  - `proposal_pipeline.py`: `build_entity_proposals()` function
  - Risk weighting per configurable thresholds
  - Provenance tracking

### D. Scene Graph Builder ✅
- **Target**: JSON schema with nodes (entity_id, canonical_label, mask, bbox, score, attributes, provenance, risk, verified) and edges (spatial + interaction relations)
- **Implemented**:
  - `scene_graph_builder.py`: `build_scene_graph()` function
  - Deterministic spatial relations: `left_of`, `right_of`, `above`, `below`, `overlap`, `inside`, `surrounding`, `intersect`, `touching`
  - Pluggable interaction relation hook (disabled by default)
  - `schema.py`: JSON schema constants and validation

### E. Attribute Extraction ✅
- **Target**: Ontology-slot-constrained attributes with confidence and provenance
- **Implemented**:
  - `attribute_extractor.py`: Heuristic slot filling
  - Per-slot confidence tracking
  - Pluggable hook for custom extractors (future: MLLM-based)

### F. Validator Layer ✅
- **Target**: Schema violations, duplicates, contradictions, low confidence, missing mandatory attributes, answer-evidence mismatch
- **Implemented**:
  - `validators.py`: `validate_scene_graph()`, `validate_vqa_evidence()`
  - Flags per node/edge, plus global graph-level flags

### G. Review Queue ✅
- **Target**: Risk-ranked items (nodes/edges/VQA pairs) with prioritized actions
- **Implemented**:
  - `review_queue.py`: `build_review_queue()` function
  - Supports: accept, edit_label, edit_mask, re_prompt_by_category, re_prompt_by_referring_expression, mark_invalid, add_missing_node/edge

### H. VQA Generation ✅
- **Target**: Single-turn (existence, count, attribute, spatial, interaction, comparison, referring-expression); multi-turn chains with dialogue state
- **Implemented**:
  - `vqa.py`: `generate_single_turn_vqa()`, `generate_multi_turn_vqa()`
  - Multi-turn chains with `chain_id`, `turn`, `dialogue_state` (focus_node_id, focus_edge_id, etc.)
  - Evidence linkage (node_ids, edge_ids)

### I. Answer Engine ✅
- **Target**: Deterministic graph answering for closed-form; constrained open-ended hook
- **Implemented**:
  - `answer_engine.py`: `AnswerEngine` class
  - Serializes subgraph + evidence for MLLM hook
  - Fallback to cached answer

### J. Quality Evaluation ✅
- **Target**: Scene graph metrics (label accuracy, mask/bbox IoU, attribute F1, relation F1, graph edit distance) + VQA metrics (answer accuracy, evidence grounding, chain consistency, rewrite rate, mismatch)
- **Implemented**:
  - `eval_scene_graph.py`: Per-node label match, mask IoU, attribute extraction F1, relation F1, edit distance
  - `eval_vqa.py`: Answer accuracy, grounding accuracy, chain consistency, rewrite detection, mismatch detection

---

## 4. Missing Modules (By Design / Future Work)

### Currently Not Implemented (Out of Scope for Initial Release)

1. **Interactive VQA UI** (`ui/vqa_window.py`)
   - Graph viewer with node/edge visualization
   - Interactive QA interface with evidence highlighting
   - Deferred to next phase (GUI integration)

2. **Advanced Interaction Relation Proposal** (`core/impact_sg/relation_proposal.py`)
   - MLLM-based pairwise crop analysis for `holding`, `wearing`, `carrying`, `sitting_on`, etc.
   - Integration point already provided in `scene_graph_builder.py` via `interaction_relation_hook` parameter
   - Default: disabled (`enable_interaction_relations: false` in `impact_sg_pipeline.json`)

3. **MLLM Integration** (`core/impact_sg/mllm_adapter.py`)
   - Actual open-ended question answering hook
   - Answer Engine already supports pluggable handler via `open_ended_hook` parameter
   - Default: None (uses cached answer fallback)

4. **Batch Inference Pipeline** (`tools/batch_impact_sg.py`)
   - Multi-image processing with CSV manifests
   - Could reuse CLI scripts via subprocess or direct import
   - Deferred to workflow optimization phase

5. **Advanced Mask Editing / Re-prompting UI**
   - Review queue actions (edit_mask, re_prompt_by_category, re_prompt_by_referring_expression)
   - Would require UI integration
   - Deferred to GUI phase

### Intentionally Excluded

- **Direct INSTANCE/PANOPTIC segmentation**: Use OpenWorldSAM for grounding only
- **Unconstrained free-form labels**: All labels must canonicalize to ontology or be rejected
- **Unsupervised relation discovery**: Relations limited to controlled vocabulary

---

## 5. Ablation & Configuration

All major components are **configurable and ablatable** via `configs/impact_sg_pipeline.json`:

```json
{
  "backend": { "provider": "mock|external_command", "max_instances_per_prompt": 20, "enable_two_stage_refinement": false, ... },
  "proposal": { "merge_mask_iou_threshold": 0.75, "risk_weights": {...}, ... },
  "validators": { "node_low_conf_threshold": 0.4, "mandatory_attributes_required": true, ... },
  "vqa": { "single_turn_max_questions": 64, "multi_turn_max_chains": 24, "prefer_deterministic_answers": true },
  "ablation": {
    "backend_mode": "openworldsam_mock",
    "use_sentence_refinement": false,
    "use_validator": true,
    "use_graph_answering": true
  }
}
```

**Example ablations**:
- `use_validator=false`: Skip all validation checks
- `use_sentence_refine=true`: Enable two-stage OpenWorldSAM refinement
- `use_graph_answering=false`: Fall back to MLLM for all VQA (if hook provided)
- `backend_mode` variants: mock (test), external_command (custom OWSAM)

---

## 6. Data Flow Example

### End-to-End Scene Graph + VQA Generation

```bash
# 1. Build scene graph from image
python tools/build_scene_graph.py \
  --image_id demo_001 \
  --image_path /path/to/image.jpg \
  --ontology configs/impact_sg_ontology.json \
  --pipeline_cfg configs/impact_sg_pipeline.json \
  --image_width 1280 \
  --image_height 720 \
  --out /output/scene_graph.json

# 2. Generate VQA + review queue from scene graph
python tools/generate_vqa.py \
  --scene_graph /output/scene_graph.json \
  --pipeline_cfg configs/impact_sg_pipeline.json \
  --out /output/vqa.json

# 3. Evaluate scene graph against ground truth
python tools/evaluate_scene_graph.py \
  --pred /output/scene_graph.json \
  --gt /path/to/gt_scene_graph.json \
  --iou_threshold 0.5 \
  --out /output/sg_metrics.json

# 4. Evaluate VQA answers against ground truth
python tools/evaluate_vqa.py \
  --pred /output/vqa.json \
  --gt /path/to/gt_vqa.json \
  --out /output/vqa_metrics.json
```

### Scene Graph JSON Structure

```json
{
  "image_id": "demo_001",
  "nodes": [
    {
      "entity_id": "ent_abc123def",
      "canonical_label": "person",
      "prompt_used": "person",
      "mask": {"pixels": [[x, y], ...]},
      "bbox": [x, y, w, h],
      "score": 0.87,
      "attributes": [
        {"slot": "color", "value": "purple", "confidence": 0.45, "provenance": "heuristic_slot_extractor", "verified": false},
        {"slot": "state", "value": "visible", "confidence": 0.9, "provenance": "heuristic_slot_extractor", "verified": false}
      ],
      "provenance": [{"backend": "OpenWorldSAM", "stage": "category_discovery", "prompt": "person", "image_path": "..."}],
      "risk": 0.15,
      "verified": false,
      "validator_flags": []
    },
    ...
  ],
  "edges": [
    {
      "edge_id": "edge_xyz789abc",
      "src_id": "ent_abc123def",
      "relation": "left_of",
      "dst_id": "ent_def456ghi",
      "score": 1.0,
      "evidence": {"type": "deterministic_spatial", "src_bbox": [...], "dst_bbox": [...]},
      "validator_flags": [],
      "risk": 0.0,
      "verified": false
    },
    ...
  ],
  "validator_flags": [...],
  "metadata": {
    "image_path": "...",
    "graph_snapshot_id": "graph_...",
    "ablation": {...},
    "backend_provider": "mock"
  }
}
```

---

## 7. Key Design Constraints Satisfied

✅ **Constraint 1**: Do NOT use OpenWorldSAM as the final scene-graph generator.
- OWSAM is plugged in as `openworldsam_backend.py` for entity proposal only; graph build is separate

✅ **Constraint 2**: Use OpenWorldSAM only for open-vocabulary entity proposal, multi-instance discovery, and referring-expression refinement.
- `discover_entities_by_category()` and `refine_with_sentence_prompts()` methods support these use cases

✅ **Constraint 3**: Use mask as the primary annotation source of truth. Derive bbox from mask.
- `mask_ops.py::bbox_from_mask_rle()` derives bbox from mask; bbox never treated as primary truth

✅ **Constraint 4**: Use a controlled ontology with canonical labels + synonyms.
- `impact_sg_ontology.json` + `Ontology.canonicalize_label()` enforce this

✅ **Constraint 5**: VQA must be graph-grounded; answer deterministically from graph when possible.
- `answer_engine.py` prefers deterministic graph answering

✅ **Constraint 6**: Every generated annotation must have provenance, confidence, and validator flags.
- All nodes/edges include `provenance`, `score`, `validator_flags`, `verified` fields

✅ **Constraint 7**: Everything must be configurable and ablatable.
- All major components controlled via `impact_sg_pipeline.json` + `ablation` section

---

## Summary

The IMPACT-SG implementation provides a **complete graph-centric annotation pipeline** anchored by:
- **Controlled ontology** (labels, synonyms, attributes, relations)
- **Pluggable OpenWorldSAM backend** for entity discovery (grounding only, not reasoning)
- **Mask-first merging and risk scoring** for robust entity proposals
- **Deterministic scene graph building** with spatial relations
- **Ontology-constrained attribute extraction**
- **Graph-grounded VQA generation** (single + multi-turn)
- **Comprehensive validation** (schema, duplicates, contradictions, low confidence)
- **Human-in-the-loop review queue** (prioritized by risk)
- **Quantitative evaluation** (scene graph metrics + VQA metrics)

All components are **configurable**, **ablatable**, and **pluggable** for future extensions (MLLM, custom relation models, advanced UI).
