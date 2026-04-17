# IMPACT-SG: Implementation Summary & Quick Reference

**Date**: 2026-03-24  
**Scope**: Graph-centric VQA pipeline using OpenWorldSAM as grounding backend

---

## ✅ What's Implemented (MVP Complete)

### Core Pipeline (1 Function Orchestrates Everything)
```python
from core.impact_sg.pipeline import run_build_scene_graph, run_generate_vqa

# 1. Build scene graph from image
graph = run_build_scene_graph(
    image_id="demo_001",
    image_path="/path/to/image.jpg",
    ontology_path="configs/impact_sg_ontology.json",
    pipeline_cfg_path="configs/impact_sg_pipeline.json",
    image_size=(1280, 720),
    enable_sentence_refine=False,
)
# Output: {"image_id", "nodes": [...], "edges": [...], "validator_flags": [...], "metadata": {...}}

# 2. Generate VQA from graph
vqa = run_generate_vqa(graph, pipeline_cfg_path="configs/impact_sg_pipeline.json")
# Output: {"single_turn": [...], "multi_turn": [...], "all": [...], "review_queue": [...]}
```

### 4 CLI Entrypoints (Ready to Use)
```bash
# 1. Build scene graph
python tools/build_scene_graph.py \
  --image_id img_001 \
  --image_path image.jpg \
  --out graph.json

# 2. Generate VQA
python tools/generate_vqa.py \
  --scene_graph graph.json \
  --out vqa.json

# 3. Evaluate scene graph (requires GT)
python tools/evaluate_scene_graph.py \
  --pred graph.json \
  --gt gt_graph.json \
  --out sg_metrics.json

# 4. Evaluate VQA (requires GT)
python tools/evaluate_vqa.py \
  --pred vqa.json \
  --gt gt_vqa.json \
  --out vqa_metrics.json
```

### 14 Core Modules (Fully Integrated)
| Module | Purpose | Lines |
|--------|---------|-------|
| `ontology.py` | Label canonicalization + prompt templates | ~190 |
| `mask_ops.py` | Mask utilities (IoU, bbox, area, touching) | ~150 |
| `schema.py` | Scene graph schema + validation | ~80 |
| `openworldsam_backend.py` | Pluggable backend (mock + external_command) | ~250 |
| `proposal_pipeline.py` | Entity merging + risk scoring | ~180 |
| `scene_graph_builder.py` | Graph construction + spatial relations | ~160 |
| `attribute_extractor.py` | Heuristic slot-constrained attributes | ~50 |
| `validators.py` | Schema/duplicate/contradiction checks | ~120 |
| `review_queue.py` | Risk-ranked human review queue | ~55 |
| `vqa.py` | Single/multi-turn VQA generation | ~260 |
| `answer_engine.py` | Deterministic + constrained open-ended answering | ~60 |
| `eval_scene_graph.py` | Graph metrics (label acc, mask IoU, F1) | ~100 |
| `eval_vqa.py` | VQA metrics (answer acc, grounding, consistency) | ~70 |
| `pipeline.py` | Orchestration (wires all components) | ~200 |

### 2 Configuration Files (JSON, Human-Editable)
```json
// configs/impact_sg_ontology.json
{
  "canonical_entities": [
    {"label": "person", "synonyms": [...], "attribute_slots": [...], "mandatory_attributes": [...]},
    ...
  ],
  "relation_vocabulary": {...},
  "question_types": [...],
  "prompt_templates": [...]
}

// configs/impact_sg_pipeline.json
{
  "backend": {...},           // OpenWorldSAM config
  "proposal": {...},          // Merging + risk settings
  "relations": {...},         // Spatial/interaction toggles
  "attributes": {...},        // Slot defaults
  "validators": {...},        // Validation thresholds
  "vqa": {...},              // VQA generation limits
  "ablation": {              // ON/OFF toggles for all major components
    "backend_mode": "openworldsam_mock",
    "use_sentence_refinement": false,
    "use_validator": true,
    "use_graph_answering": true
  }
}
```

---

## 🎯 Key Design Decisions

| Constraint | Implementation |
|-----------|-----------------|
| Mask-first | `mask_ops.py::bbox_from_mask_rle()` derives bbox from mask, never treat as primary |
| Controlled ontology | `ontology.py::canonicalize_label()` uses synonym mapping; rejects free-form labels |
| Pluggable backend | `openworldsam_backend.py` with mock + external_command providers; easy to swap |
| Deterministic answering | `answer_engine.py` prefers graph answering; MLLM hook for open-ended (constrained) |
| Configurable | Every component has JSON config knob + ablation toggle |
| Provenance | Every node/edge includes `provenance`, `score`, `validator_flags`, `verified` |

---

## 📊 Graph & VQA Output Format

### Scene Graph JSON
```json
{
  "image_id": "demo_001",
  "nodes": [
    {
      "entity_id": "ent_abc123",
      "canonical_label": "person",        // Canonicalized via ontology
      "prompt_used": "person",            // Prompt that generated this
      "mask": {"pixels": [[x, y], ...]},  // Mask-first annotation
      "bbox": [x, y, w, h],               // Derived from mask
      "score": 0.87,                      // Confidence
      "attributes": [                     // Ontology-constrained slots only
        {"slot": "color", "value": "purple", "confidence": 0.45, "provenance": "...", "verified": false}
      ],
      "provenance": [                     // Audit trail
        {"backend": "OpenWorldSAM", "stage": "category_discovery", "prompt": "person", "image_path": "..."}
      ],
      "risk": 0.15,                       // Risk score (low-conf, overlap, size, ambiguity)
      "verified": false,                  // Human verified?
      "validator_flags": []               // Schema/duplicate/contradiction flags
    }
  ],
  "edges": [
    {
      "edge_id": "edge_xyz789",
      "src_id": "ent_abc123",
      "relation": "left_of",              // From controlled vocabulary
      "dst_id": "ent_def456",
      "score": 1.0,                       // Deterministic spatial = 1.0
      "evidence": {"type": "deterministic_spatial", "src_bbox": [...], "dst_bbox": [...]},
      "validator_flags": [],
      "risk": 0.0,
      "verified": false
    }
  ],
  "validator_flags": [...],               // Graph-level validation issues
  "metadata": {
    "image_path": "...",
    "graph_snapshot_id": "graph_...",
    "ablation": {...},                   // Which components were enabled?
    "backend_provider": "mock"           // Which backend generated this?
  }
}
```

### VQA Output JSON
```json
{
  "graph_snapshot_id": "graph_...",
  "single_turn": [              // Existence, counting, attributes, spatial, comparison
    {
      "qid": "sq_exist_...",
      "question": "Is there a person in the image?",
      "answer": "yes",
      "answer_type": "boolean",
      "evidence_node_ids": ["ent_abc123"],      // Grounded to graph
      "evidence_edge_ids": [],
      "graph_snapshot_id": "graph_...",
      "risk": 0.1,                              // Risk of missing/wrong
      "verified": false,
      "validator_flags": []
    }
  ],
  "multi_turn": [               // Chains with dialogue state
    {
      "qid": "mq_t1_...",
      "chain_id": "chain_...",
      "turn": 1,
      "question": "Which object are we talking about?",
      "answer": "person (ent_abc123)",
      "answer_type": "reference",
      "evidence_node_ids": ["ent_abc123"],
      "evidence_edge_ids": [],
      "dialogue_state": {
        "focus_node_id": "ent_abc123",
        "focus_edge_id": null,
        "comparison_group": null
      },
      ...
    }
  ],
  "all": [...],                 // Flat list of all Q&A items
  "review_queue": [             // Risk-ranked items for human review
    {
      "item_type": "node|edge|vqa",
      "item_id": "...",
      "priority": 0.8,          // Risk score (0-1, higher = more urgent)
      "reasons": ["low_confidence_node", "missing_mandatory_attributes"],
      "actions": ["accept", "edit_label", "mark_invalid", ...]
    }
  ]
}
```

---

## 🚀 Running End-to-End

### Quick Test (Mock Backend)
```bash
cd /path/to/IMPACT_VQA

# 1. Build graph (will use mock OpenWorldSAM)
python tools/build_scene_graph.py \
  --image_id test_001 \
  --image_path /tmp/test.jpg \
  --image_width 640 \
  --image_height 480 \
  --out /tmp/graph.json

# 2. Generate VQA
python tools/generate_vqa.py \
  --scene_graph /tmp/graph.json \
  --out /tmp/vqa.json

# 3. Check outputs
cat /tmp/graph.json | python -m json.tool | head -50
cat /tmp/vqa.json | python -m json.tool | head -50
```

### With Real OpenWorldSAM (Future)
```bash
# Edit configs/impact_sg_pipeline.json:
{
  "backend": {
    "provider": "external_command",
    "external_command_template": "python /path/to/owsam.py --image {image_path} --prompt '{prompt}'"
  }
}

# Then run CLI as normal (will call external command instead of mock)
python tools/build_scene_graph.py --image_id img_001 --image_path image.jpg --out graph.json
```

---

## 🔧 Configuration Recipes

### Disable Validators (Skip All Checks)
```json
// configs/impact_sg_pipeline.json
{
  "ablation": {
    "use_validator": false
  }
}
```

### Enable Two-Stage Sentence Refinement
```json
{
  "backend": {
    "enable_two_stage_refinement": true
  },
  "ablation": {
    "use_sentence_refinement": true
  }
}
```

### Tune Risk Weights
```json
{
  "proposal": {
    "risk_weights": {
      "low_confidence": 0.6,          // Heavy weight on confidence
      "overlap_conflict": 0.1,        // Less weight on overlaps
      "small_thin_prior": 0.2,
      "ontology_ambiguity": 0.1
    }
  }
}
```

### Use Different Attribute Extractor (Future)
```json
// (Requires Phase 2 implementation)
{
  "attributes": {
    "extractor": "clip",              // or "dino", "finetuned", etc.
    "model_path": "/path/to/model.pth"
  }
}
```

---

## 📋 Verification Checklist

- ✅ All imports work (no syntax errors)
- ✅ All 4 CLIs parse arguments correctly
- ✅ Mock backend generates deterministic proposals
- ✅ Scene graph schema validates
- ✅ VQA chains link correctly to nodes/edges
- ✅ Validators detect low-confidence nodes
- ✅ Review queue ranks by risk
- ✅ End-to-end pipeline runs without errors
- ✅ Output JSON is valid and well-formed

---

## 🎓 For Next Developer (What's Next?)

### Immediate (Week 1-2)
1. Review `docs/IMPACT_SG_architecture.md` (understand design)
2. Run smoke tests (mock OpenWorldSAM)
3. Study ontology format + canonicalization logic
4. Pick a Tier 1 task from `docs/IMPACT_SG_missing_modules.md`

### High Priority
1. **Interactive VQA UI** (ui/vqa_window.py) — Replace PlaceholderPane
2. **MLLM Integration** (core/impact_sg/mllm_adapter.py) — LLaVA first
3. **Batch Inference** (tools/batch_impact_sg.py) — Multi-image processing

### Medium Priority
4. **Advanced Attributes** (core/impact_sg/attribute_extractors/) — CLIP-based color/material
5. **Interaction Relations** (core/impact_sg/relation_proposal.py) — holding/wearing/carrying
6. **Review Queue UI** (ui/review_window.py) — Human review actions

### Deferred
7. Caching optimization
8. Production deployment (Docker + REST API)
9. Benchmark datasets

---

## 📚 Documentation Files (Start Here)

| File | Purpose |
|------|---------|
| `docs/IMPACT_SG_architecture.md` | Comprehensive system design + module mapping |
| `docs/IMPACT_SG_execution_plan.md` | What was implemented, how, and testing guidance |
| `docs/IMPACT_SG_missing_modules.md` | Post-MVP features + implementation roadmap |
| `README.md` (existing) | General repo overview |
| `configs/impact_sg_ontology.json` | Entity labels, attributes, relations (editable) |
| `configs/impact_sg_pipeline.json` | All tunable parameters + ablation flags (editable) |

---

## 🔌 Extension Hooks (Plug Your Code Here)

| Component | How to Extend |
|-----------|---------------|
| **Backend** | Implement external_command OWSAM endpoint; set provider="external_command" |
| **Attributes** | Replace heuristic extractor in `attribute_extractor.py`; register new model class |
| **MLLM** | Implement `MLLMAdapter` class; register via `answer_engine.open_ended_hook` |
| **Relations** | Implement interaction predictor; register via `interaction_relation_hook` |
| **Validators** | Add new validation rules in `validators.py` |
| **VQA Types** | Add new question generators in `vqa.py` |

---

## ⚠️ Known Limitations (MVP)

1. **Mock OpenWorldSAM** — Deterministic test proposals; needs real OWSAM integration
2. **Heuristic Attributes** — No vision model; upgrade to CLIP in Phase 2
3. **No Interaction Relations** — Disabled by default; requires Phase 2 implementation
4. **No MLLM Hook** — Open-ended questions cached; needs Phase 2 integration
5. **No UI** — All via CLI; GUI deferred to Phase 2
6. **No Persistent Review** — Review queue generated but not saved; needs UI integration

---

## 🎯 Success Criteria Met

✅ Graph-grounded annotation pipeline  
✅ OpenWorldSAM as grounding backend (pluggable)  
✅ Mask-first annotations  
✅ Controlled ontology with canonicalization  
✅ Scene graph JSON schema  
✅ Deterministic spatial relations  
✅ Attribute extraction (heuristic, upgradable)  
✅ Comprehensive validators  
✅ Review queue for human feedback  
✅ VQA generation (single + multi-turn)  
✅ Graph-grounded answering  
✅ Evaluation metrics (scene graph + VQA)  
✅ 4 CLI entry points  
✅ Fully configurable & ablatable  
✅ Production-ready code structure  

---

**Status: READY FOR PHASE 2 (UI + MLLM Integration)**
