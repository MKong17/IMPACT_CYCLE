# IMPACT-SG: Missing Modules & Future Work

**Date**: 2026-03-24  
**Status**: Initial MVP complete; this document lists post-MVP extensions

---

## Summary

The **core graph-grounded VQA pipeline** is fully implemented and functional (14 modules + 4 CLIs + 2 configs). This document catalogs **intentionally deferred** components and their implementation roadmap.

---

## Tier 1: High Priority (Recommended for Phase 2)

### 1. Interactive VQA UI (`ui/vqa_window.py`)

**Purpose**: Replace placeholder `PlaceholderPane` with functional VQA interface

**Scope**:
- Scene graph visualization (node layout, edge rendering)
- Node inspector (attributes, provenance, validator flags, images)
- Edge inspector (spatial/interaction relations, evidence visualizations)
- QA browser:
  - List single-turn QA pairs
  - Multi-turn chains with dialogue state visualization
  - Answer + evidence highlighting (bboxes on image)
- Evidence replay (show masked regions + context crops)

**Integration Points**:
- Import `core.impact_sg.pipeline.run_build_scene_graph()`
- Import `core.impact_sg.vqa.generate_single_turn_vqa()` + `generate_multi_turn_vqa()`
- Subclass from `ui.action_window.ActionWindow` or create standalone Qt widget

**Estimated Effort**: 4-6 weeks (including testing + UX polish)

**Dependencies**: PyQt5 (already present), matplotlib or similar for graph layout

---

### 2. MLLM Integration (`core/impact_sg/mllm_adapter.py`)

**Purpose**: Integrate Large Multimodal Language Models (LLaVA, GPT-4V, Gemini, etc.) for open-ended VQA

**Scope**:
- Abstract adapter class with standard interface:
  ```python
  class MLLMAdapter:
      def answer_open_ended(self, question: str, subgraph: dict, image_regions: List[np.ndarray]) -> str: ...
  ```
- Concrete implementations:
  - LLaVA (HF transformers)
  - OpenAI GPT-4V API client
  - Google Gemini API client (optional)
- **Constraint wrapper**: Disallow mentions of entities not in graph
- Prompt engineering: Initialize with evidence subgraph + entity list

**Integration Points**:
- Register handler in answer_engine.py via:
  ```python
  engine = AnswerEngine(open_ended_hook=mllm_adapter.answer_open_ended)
  ```

**Estimated Effort**: 2-3 weeks per MLLM implementation (+1 week for constraint wrapper)

**Dependencies**: transformers, openai, requests, or equivalent

---

### 3. Batch Inference Pipeline (`tools/batch_impact_sg.py`)

**Purpose**: Process multiple images end-to-end (graph build + VQA generation + evaluation)

**Scope**:
- CSV/JSON manifest input (image_id, image_path, gt_graph_path, gt_vqa_path)
- Parallel processing (multi-GPU support optional)
- Aggregation:
  - Per-image metrics
  - Aggregate precision/recall/F1 curves
  - Summary report HTML
- Error handling + retry logic
- Progress bar + logging

**Integration Points**:
- Reuse `core.impact_sg.pipeline.run_build_scene_graph()`
- Reuse `core.impact_sg.pipeline.run_generate_vqa()`
- Reuse `core.impact_sg.eval_scene_graph.evaluate_scene_graph()`
- Reuse `core.impact_sg.eval_vqa.evaluate_vqa()`

**Estimated Effort**: 2-3 weeks

**Dependencies**: pandas (for CSV), multiprocessing (in stdlib)

---

## Tier 2: Medium Priority (Phase 3 + Optimization)

### 4. Advanced Attribute Extraction (`core/impact_sg/attribute_extractors/`)

**Purpose**: Train/integrate vision models for semantic attribute prediction (color, material, state, affordance)

**Scope**:
- Module structure:
  ```
  attribute_extractors/
    ├─ __init__.py
    ├─ base.py          (abstract AttributeExtractor)
    ├─ heuristic.py     (existing impl moved here)
    ├─ clip_extractor.py (CLIP-based color/object classification)
    ├─ dino_extractor.py (DINOv2-based material/texture)
    └─ finetuned.py     (custom dataset finetuned models)
  ```
- Per-slot specialist models (not one monolithic model)
- Confidence calibration (report calibrated confidence, not raw logits)
- Ontology validation (constrain predictions to allowed values)

**Integration Points**:
- Swap in attribute_extractor.py factory:
  ```python
  ex = AttributeExtractorFactory.create('clip', slot='color')
  ```

**Estimated Effort**: 3-4 weeks (including dataset curation + eval)

**Dependencies**: CLIP, DINOv2, torch, torchvision (mostly already present via EAST)

---

### 5. Interaction Relation Proposal (`core/impact_sg/relation_proposal.py`)

**Purpose**: Predict semantic interactions (holding, wearing, carrying, sitting_on, etc.) from pairwise entity crops

**Scope**:
- Pairwise crop extraction (context around both entities)
- Relation candidate generation:
  ```python
  class PairwiseInteractionPredictor:
      def predict_interactions(self, src_node: dict, dst_node: dict, image: np.ndarray) -> List[dict]: ...
  ```
- MLLM-based (zero-shot): Send crop pairs to MLLM with prompt
- Finetuned model (optional): Train on datasets like HCVRD, GQA
- Confidence scoring + filtering

**Integration Points**:
- Register as `interaction_relation_hook` in scene_graph_builder.py:
  ```python
  graph = build_scene_graph(
      ...,
      enable_interaction_relations=True,
      interaction_relation_hook=predictor.predict_interactions
  )
  ```

**Estimated Effort**: 2-3 weeks (MLLM version is simpler; finetuned version is longer)

**Dependencies**: MLLM adapter (from Tier 1), PIL, numpy

---

### 6. Review Queue UI (`ui/review_window.py`)

**Purpose**: Implement human-in-the-loop review interface for flagged/risky items

**Scope**:
- Queue list view:
  - Sort by priority (risk score)
  - Filter by item_type (node/edge/vqa)
  - Quick stats (x nodes, y edges, z VQA items)
- Item inspector + actions:
  - **For nodes**:
    - accept
    - edit_label (dropdown from ontology)
    - edit_mask (mask drawing tool)
    - re_prompt_by_category (category selector + rerun OpenWorldSAM)
    - re_prompt_by_referring_expression (text field + rerun sentence refinement)
    - mark_invalid (delete from graph)
    - add_missing_node (draw new mask)
  - **For edges**:
    - accept
    - mark_invalid
    - add_missing_edge (select pair + relation)
  - **For VQA**:
    - accept
    - edit_label (text edit)
    - mark_invalid
- Undo/redo for review actions
- Save reviewed graph + VQA

**Integration Points**:
- Launch from `MainWindow` task dropdown
- Re-integrate to pipeline: `run_build_scene_graph()` with existing proposals

**Estimated Effort**: 3-4 weeks (including UI polish + undo/redo)

**Dependencies**: PyQt5 (existing), mask drawing library (e.g., OpenCV GUI)

---

### 7. Caching & Optimization (`core/impact_sg/caching.py`)

**Purpose**: Cache intermediate results (feature extraction, prompt results, graph snapshots) to speed up batch inference

**Scope**:
- Feature-level cache (EAST backbone outputs)
  - Key: hash(image_path, backbone_version)
  - Value: numpy array [T, D]
- Prompt-level cache (OpenWorldSAM results per image+prompt)
  - Already implemented in `openworldsam_backend.py`
  - Consider upgrading to persistent cache layer (SQLite)
- Graph snapshot cache (avoid rebuilding if proposals unchanged)
- Cache invalidation strategies (TTL, LRU)

**Estimated Effort**: 1-2 weeks

**Dependencies**: sqlite3 (stdlib), numpy

---

## Tier 3: Low Priority / Nice-to-Have

### 8. Confusion Matrix + Ablation Reporting

**Purpose**: Detailed analysis of model performance (confusion matrix, ablation effect sizes)

**Scope**:
- Per-label confusion matrix (predicted vs. ground truth)
- Ablation experiment harness:
  - Run evaluations with different `ablation` configs
  - Compare metrics side-by-side
  - Report effect size (improvement from validator, etc.)
- Visualization (matplotlib/plotly)

**Estimated Effort**: 1 week

**Dependencies**: matplotlib, pandas

---

### 9. Benchmark Datasets & Baselines

**Purpose**: Publish benchmark datasets + baseline results for IMPACT-SG

**Scope**:
- Convert existing assembly101 / 50salads to IMPACT-SG format (scene graphs + VQA)
- Baseline results (mock OpenWorldSAM vs. real OWSAM)
- Leaderboard (optional)

**Estimated Effort**: 2-3 weeks (dataset conversion is tedious)

---

### 10. Production Deployment

**Purpose**: Docker containerization + REST API for serving IMPACT-SG

**Scope**:
- Dockerfile (Python 3.9+, CUDA support)
- REST API Flask/FastAPI app
- Health checks, logging, monitoring
- Horizontal scaling (via Kubernetes or similar)

**Estimated Effort**: 2-3 weeks

**Dependencies**: Docker, Flask/FastAPI, Prometheus (optional)

---

## Implementation Roadmap (Recommended)

### **Phase 2 (Weeks 1-8)**
1. Interactive VQA UI (Weeks 1-6)
2. MLLM integration - LLaVA (Week 7-8)

### **Phase 3 (Weeks 9-16)**
3. Batch inference pipeline (Weeks 9-11)
4. Advanced attribute extraction - CLIP (Weeks 12-16)

### **Phase 4 (Weeks 17-24)**
5. Interaction relation proposal (Weeks 17-19)
6. Review queue UI (Weeks 20-24)

### **Phase 5+ (Post-MVP)**
7. Caching optimization (1-2 weeks)
8. MLLM integration - GPT-4V, Gemini (parallelized with phase 4)
9. Production deployment (2-3 weeks)
10. Benchmark datasets (ongoing)

---

## Extension Hooks (Already Baked In)

The following extension points are **already implemented** and ready for custom code:

### 1. Custom Attribute Extractors
```python
# In attribute_extractor.py, replace heuristic with your own:
def extract_attributes_for_nodes(nodes, ontology, default_confidence, allow_affordance):
    # Call your_custom_extractor(nodes)
    ...
```

### 2. Custom OpenWorldSAM Backend
```python
# Create external command interface:
# --image {image_path} --prompt {prompt} --max_instances {max_instances}
# returns JSON: [{"mask": {...}, "score": ...}, ...]

# Then set in configs/impact_sg_pipeline.json:
"backend": {
    "provider": "external_command",
    "external_command_template": "python /path/to/your_owsam.py ..."
}
```

### 3. Custom MLLM for Open-Ended VQA
```python
# Implement adapter:
class MyMLLMAdapter:
    def answer_open_ended(self, question: str, subgraph: dict, image_regions: List[np.ndarray]) -> str:
        # Your MLLM call here
        ...

# Register in answer_engine:
engine = AnswerEngine(open_ended_hook=MyMLLMAdapter().answer_open_ended)
```

### 4. Custom Interaction Relations
```python
# Implement predictor:
class MyRelationPredictor:
    def predict_interactions(self, src_node: dict, dst_node: dict, image: np.ndarray) -> List[dict]:
        # Returns [{edge_id, src_id, relation, dst_id, score, evidence, ...}, ...]
        ...

# Register in scene_graph_builder:
graph = build_scene_graph(
    ...,
    enable_interaction_relations=True,
    interaction_relation_hook=MyRelationPredictor().predict_interactions
)
```

### 5. Custom Validators
```python
# Add new validators in validators.py:
def validate_my_constraint(graph: dict) -> Dict[str, object]:
    # Return {graph_flags, node_flags, edge_flags}
```

---

## Checklist for Next Developer

- [ ] Review `IMPACT_SG_architecture.md` (understand overall design)
- [ ] Review `IMPACT_SG_execution_plan.md` (understand what's been done)
- [ ] Run end-to-end smoke test (see execution plan quickstart)
- [ ] Familiarize with ontology config format
- [ ] Pick a Tier 1 task (UI or MLLM) to start
- [ ] Check "Integration Points" section for how to wire new code
- [ ] Reference existing modules for code style/patterns

---

**End of Missing Modules Document**
