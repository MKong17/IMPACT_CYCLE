# IMPACT 4-Task Cycle Verification Roadmap

Updated: 2026-04-04

This document gives the recommended low-risk, high-novelty implementation path for turning the current four tasks:

- Video Scene Graph
- Single-turn VQA
- Multi-turn VQA
- Video Captioning

into one unified cross-task cycle with lightweight human arbitration.

The intended paper direction is:

`Cross-Task Cycle Verification for Human-Light Video Scene Graph Annotation`

The intended system direction is:

- scene graph is the latent structured state
- VQA and caption are not just outputs, but supervisory views over the graph
- models propose, cross-check, and revise
- humans only arbitrate the residual high-conflict cases

---

## 1. Executive Decision

### Final recommendation

The best path is not:

- a pure engineering tool
- a pure end-to-end scene graph generator
- a pure LLM prompt-chain system

The best path is a hybrid:

1. Keep the current mask-first graph pipeline as the deterministic backbone.
2. Add a cross-task verification loop:
   - graph -> single-turn VQA
   - graph -> multi-turn VQA
   - graph -> normalized caption
   - VLM/LLM answers + rewrites -> claim extraction -> graph revision
3. Add a conservative belief-update layer:
   - only high-confidence multi-view agreement updates the graph automatically
   - medium-confidence conflicts enter a ranked queue
   - only the smallest unresolved conflicts are sent to humans
4. Reuse the repository's existing interactive-learning ideas from Action Segmentation:
   - layered supervision
   - correction memory
   - query utility
   - lock confirmed regions
   - online and offline two-timescale adaptation

### Why this is the best trade-off

It is innovative enough because it unifies four tasks into one closed supervision loop.

It is low-risk enough because:

- the graph remains the source of truth
- visual grounding remains externalized to the existing OpenWorldSAM path
- deterministic geometry and validators remain in the loop
- the VLM is used as a verifier and constrained reviser, not as the sole graph generator

This avoids the two biggest risks:

- training a data-hungry end-to-end video scene graph model from scratch
- over-trusting free-form LLM outputs without structural constraints

---

## 2. What To Keep From Existing Markdown Files

After reviewing the current repository documents, the following parts are worth keeping.

### 2.1 Keep from `docs/IMPACT_SG_architecture.md`

Keep these principles as the core system contract:

- OpenWorldSAM is proposal-only, not the final reasoner
- mask-first annotations
- controlled ontology
- graph-grounded VQA
- validators + review queue
- provenance everywhere
- configurable and ablatable design

These are already aligned with a publishable systems-plus-method paper.

### 2.2 Keep from `docs/interactive_action_segmentation_method_guide_20260320.md`

This document is the most underestimated asset in the repository.

Its method ideas should be transferred into the graph/VQA/caption workflow:

- only confirmed supervision becomes learning signal
- accepted / corrected / finalized supervision should have different strength
- correction memory should be persistent
- queue ranking should include expected training utility
- already confirmed regions should be locked and not repeatedly overwritten
- online per-video updates and offline cross-video consolidation should be separated

This is directly relevant to the new four-task loop.

### 2.3 Keep from `README.md`

Keep the operator-facing four-task framing in the UI:

- Video Scene Graph
- Single-turn VQA
- Multi-turn VQA
- Video Captioning

But in the paper, do not describe them as four independent tools.

Describe them as four evidence views over one latent graph.

### 2.4 Do not center the paper around these current parts

Do not make these the main contribution:

- current heuristic attribute extraction
- current heuristic caption text
- current template-only VQA generation
- current threshold-only review queue

These should become baselines or initialization components, not the headline method.

---

## 3. Latest Relevant Developments Through 2026-04-04

The goal here is not to chase every new model.

The goal is to identify which recent developments are mature enough to influence a low-risk design.

### 3.1 Open-vocabulary scene graph generation with VLMs

`From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models` (CVPR 2024) explicitly frames scene graph generation as image-to-graph generation with VLMs and targets novel relation concepts. This is important because it shows that open-vocabulary graph generation is now a real direction, but it is still best treated as an inspiration layer rather than the whole production backbone for this repo.

Design takeaway:

- use VLMs for open-vocabulary relation and attribute verification
- do not replace the whole current graph pipeline with fully generative graph decoding

### 3.2 VQA can verify scene graphs

`Knowledge Informed Sequential Scene Graph Verification Using VQA` (ICCVW 2023) is directly aligned with your idea. It uses VQA as a proxy for visual content analysis to detect scene graph inconsistencies and propose corrections.

Design takeaway:

- your SG <-> VQA mutual verification idea is legitimate and publishable
- but the new system should improve it by localizing evidence, adding captions, adding video memory, and minimizing human intervention

### 3.3 Video scene graph research still suffers from data scarcity and pipeline fragility

`Panoptic Video Scene Graph Generation` (CVPR 2023) established mask-grounded video scene graphs as an important target.

`Learning 4D Panoptic Scene Graph Generation from Rich 2D Visual Scene` (CVPR 2025) explicitly highlights data scarcity, OOV issues, and weaknesses of cascade systems for 4D PSG.

Design takeaway:

- do not attempt a full end-to-end learned video PSG model in this project
- keep the modular pipeline
- focus the novelty on cross-task verification and human-light refinement

### 3.4 Detailed caption evaluation is now graph-centric

`Benchmarking Large Vision-Language Models via Directed Scene Graph for Comprehensive Image Captioning` (CVPR 2025 / CompreCap) evaluates detailed caption quality using object, attribute, and relationship coverage from directed scene graphs.

Design takeaway:

- caption quality should not be evaluated by BLEU/CIDEr style metrics alone
- caption should be treated as a graph-supervisory view
- graph-to-caption and caption-to-graph consistency is methodologically timely

### 3.5 Closed-loop VQA is now a recognized way to evaluate embodied/spatial understanding

`MetaVQA` (CVPR 2025) emphasizes VQA plus closed-loop simulation for spatial reasoning and scene dynamics.

Design takeaway:

- a closed-loop evaluation framing is modern and accepted
- your system should report not only answer accuracy but also loop consistency and correction efficiency

### 3.6 Practical local multimodal models are now strong enough for constrained verification

Official Qwen2.5-VL model documentation states:

- it understands long videos
- it can localize objects with bounding boxes or points
- it can produce stable JSON outputs for coordinates and attributes

This makes it a strong low-risk local verifier.

InternVL's official repository also shows that by late 2024 and 2025, open multimodal models reached strong perception and reasoning performance, including video understanding.

Design takeaway:

- local VLM verification is now practical
- we no longer need to rely entirely on closed APIs
- the safest path is local-first verification with optional API escalation only for hard conflicts

---

## 4. Final System Design

### 4.1 System name

Use a name like:

- IMPACT-Cycle
- IMPACT-4T
- GraphCycle

I recommend:

`IMPACT-Cycle: Cross-Task Cycle Verification for Human-Light Video Scene Graph Annotation`

### 4.2 Core idea

There is exactly one latent object of record:

- `VerifiedSceneGraph`

Everything else is a view over it:

- `SingleTurnVQA`
- `MultiTurnVQA`
- `NormalizedCaption`
- `ConflictQueue`

### 4.3 Main loop

For each sampled frame or tracked mini-clip:

1. build initial graph from grounding + deterministic relations
2. extract atomic claims from graph
3. generate:
   - yes/no and counting probes
   - compositional multi-turn probes
   - structured caption
4. ask local VLM to answer and/or critique
5. map answers and caption back into claims
6. score support/conflict/uncertainty per claim
7. update graph conservatively
8. send only unresolved high-value claims to human arbitration
9. log the final decision into correction memory
10. use memory to bias future prompts, ranking, and revision

### 4.4 Why this is better than a fully learned end-to-end model

Because the difficult parts are separated:

- grounding stays explicit
- graph stays explicit
- revision stays explicit
- human intervention stays explicit
- failure modes stay debuggable

This matters for a real annotation platform and for a defensible SMC paper.

---

## 5. Low-Risk Implementation Path

### Phase 0. Do not rebuild the core

Do not rewrite:

- `core/impact_sg/pipeline.py`
- `core/impact_sg/openworldsam_backend.py`
- `core/impact_sg/scene_graph_builder.py`
- `core/impact_sg/tracking.py`

These remain the proposal and graph backbone.

### Phase 1. Introduce claim-level verification

Add a new abstraction:

- every node, edge, and attribute becomes one or more atomic claims

Examples:

- `exists(person#track_0001)`
- `label(track_0001)=person`
- `attr(track_0001,state)=visible`
- `rel(track_0001,left_of,track_0002)`

Why this first:

- it is easy to implement
- it immediately gives a common currency across SG, VQA, and caption
- it is the minimum needed for cross-task consistency

### Phase 2. Replace heuristic caption with structured caption

Current captioning is template text in the UI. Replace it with:

- graph-to-caption prompt
- caption-to-claim extractor
- coverage and contradiction scoring

Important:

- caption is generated from graph plus optional cropped evidence
- caption is then parsed back into claims
- only claims, not raw free text, can update the graph

### Phase 3. Add local VLM verifier

Use a local verifier first.

Recommended default:

- `Qwen2.5-VL-7B-Instruct`

Reason:

- stable Transformers usage
- local video support
- localization support
- JSON-friendly outputs
- lower integration risk than heavier or more exotic pipelines

Optional stronger local upgrade:

- `InternVL2.5-8B` or `InternVL3` family if hardware allows

### Phase 4. Add optional closed-model escalation

Do not use the API model for everything.

Only use it for:

- top-K unresolved conflicts
- relation disambiguation
- complex attribute ambiguity
- caption conflict arbitration

This keeps cost low and prevents the method from depending on one vendor.

### Phase 5. Transfer correction-memory ideas from Action Segmentation

Implement:

- accepted / corrected / finalized supervision levels
- persistent label confusion memory
- relation confusion memory
- prompt alias memory
- region lock / track lock
- online per-video memory
- offline cross-video consolidation

This is where the existing repo has unusually strong reusable ideas.

---

## 6. Proposed Directory-Level Additions

Recommended new files:

```text
core/impact_sg/
  cycle_types.py
  claim_graph.py
  captioning.py
  consistency.py
  belief_update.py
  arbitration.py
  correction_memory.py
  cycle_pipeline.py
  mllm_adapters/
    __init__.py
    base.py
    qwen25_vl.py
    api_verifier.py
tools/
  run_cycle_refine.py
tests/
  test_claim_graph.py
  test_consistency.py
  test_belief_update.py
  test_correction_memory.py
configs/
  impact_cycle.json
```

Recommended later UI additions:

```text
ui/
  cycle_review_panel.py
  caption_review_panel.py
```

---

## 7. Data Model

### 7.1 Claim object

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Claim:
    claim_id: str
    claim_type: str
    subject_id: str
    predicate: str
    object_id: str = ""
    value: str = ""
    source_graph_snapshot_id: str = ""
    evidence_node_ids: List[str] = field(default_factory=list)
    evidence_edge_ids: List[str] = field(default_factory=list)
    provenance: List[Dict[str, object]] = field(default_factory=list)
    prior_score: float = 0.5
    support_score: float = 0.0
    conflict_score: float = 0.0
    uncertainty_score: float = 1.0
    status: str = "proposed"
```

### 7.2 Cross-task evidence object

```python
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvidenceView:
    view_id: str
    view_type: str
    payload: Dict[str, object]
    derived_claim_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    provenance: List[Dict[str, object]] = field(default_factory=list)
```

### 7.3 Belief state

```python
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BeliefState:
    graph: Dict[str, object]
    claims: Dict[str, Claim] = field(default_factory=dict)
    views: Dict[str, EvidenceView] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
```

---

## 8. Config Extension

Add a new config:

```json
{
  "local_verifier": {
    "provider": "qwen25_vl",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "device": "cuda",
    "max_new_tokens": 256,
    "use_flash_attention": false
  },
  "api_verifier": {
    "enabled": false,
    "provider": "generic_api",
    "model": "",
    "max_calls_per_frame": 4
  },
  "cycle": {
    "enable_single_turn_probes": true,
    "enable_multi_turn_probes": true,
    "enable_caption_probe": true,
    "auto_accept_threshold": 0.85,
    "auto_reject_threshold": 0.80,
    "human_escalation_threshold": 0.45,
    "max_human_queries_per_frame": 3,
    "max_revision_rounds": 2
  },
  "memory": {
    "enable_label_confusion_memory": true,
    "enable_relation_confusion_memory": true,
    "enable_prompt_alias_memory": true,
    "finalized_weight_boost": 1.25
  },
  "caption": {
    "style": "technical",
    "require_relation_mentions": true,
    "max_sentences": 4
  }
}
```

---

## 9. Core Algorithms

### 9.1 Graph to atomic claims

```python
from __future__ import annotations

from typing import Dict, List


def graph_to_claims(graph: Dict[str, object]) -> List[Claim]:
    out: List[Claim] = []
    snapshot_id = str((graph.get("metadata") or {}).get("graph_snapshot_id", graph.get("image_id", "")))

    for node in graph.get("nodes") or []:
        nid = str(node.get("entity_id", ""))
        label = str(node.get("canonical_label", ""))
        out.append(
            Claim(
                claim_id=f"claim_exists_{nid}",
                claim_type="existence",
                subject_id=nid,
                predicate="exists",
                value="true",
                source_graph_snapshot_id=snapshot_id,
                evidence_node_ids=[nid],
                prior_score=float(node.get("score", 0.5)),
                provenance=[{"source": "scene_graph", "field": "node"}],
            )
        )
        out.append(
            Claim(
                claim_id=f"claim_label_{nid}",
                claim_type="label",
                subject_id=nid,
                predicate="label",
                value=label,
                source_graph_snapshot_id=snapshot_id,
                evidence_node_ids=[nid],
                prior_score=float(node.get("score", 0.5)),
                provenance=[{"source": "scene_graph", "field": "canonical_label"}],
            )
        )
        for att in node.get("attributes") or []:
            if not isinstance(att, dict):
                continue
            slot = str(att.get("slot", "")).strip()
            value = str(att.get("value", "")).strip()
            if not slot or not value:
                continue
            out.append(
                Claim(
                    claim_id=f"claim_attr_{nid}_{slot}",
                    claim_type="attribute",
                    subject_id=nid,
                    predicate=slot,
                    value=value,
                    source_graph_snapshot_id=snapshot_id,
                    evidence_node_ids=[nid],
                    prior_score=float(att.get("confidence", 0.35)),
                    provenance=[{"source": "scene_graph", "field": "attribute"}],
                )
            )

    for edge in graph.get("edges") or []:
        eid = str(edge.get("edge_id", ""))
        src = str(edge.get("src_id", ""))
        rel = str(edge.get("relation", ""))
        dst = str(edge.get("dst_id", ""))
        out.append(
            Claim(
                claim_id=f"claim_rel_{eid}",
                claim_type="relation",
                subject_id=src,
                predicate=rel,
                object_id=dst,
                source_graph_snapshot_id=snapshot_id,
                evidence_node_ids=[src, dst],
                evidence_edge_ids=[eid],
                prior_score=float(edge.get("score", 0.5)),
                provenance=[{"source": "scene_graph", "field": "edge"}],
            )
        )

    return out
```

### 9.2 Claim probes from graph

```python
from __future__ import annotations

from typing import Dict, List


def build_single_turn_probes(graph: Dict[str, object]) -> List[Dict[str, object]]:
    probes: List[Dict[str, object]] = []
    for node in graph.get("nodes") or []:
        nid = str(node.get("entity_id", ""))
        label = str(node.get("canonical_label", "object"))
        probes.append(
            {
                "probe_id": f"probe_exist_{nid}",
                "probe_type": "single_turn",
                "question": f"Is there a {label} in this frame? Answer only yes or no.",
                "target_claim_id": f"claim_exists_{nid}",
                "evidence_node_ids": [nid],
            }
        )
        probes.append(
            {
                "probe_id": f"probe_label_{nid}",
                "probe_type": "single_turn",
                "question": f"Is the highlighted object best described as '{label}'? Answer yes or no and briefly explain.",
                "target_claim_id": f"claim_label_{nid}",
                "evidence_node_ids": [nid],
            }
        )
    for edge in graph.get("edges") or []:
        eid = str(edge.get("edge_id", ""))
        src = str(edge.get("src_id", ""))
        dst = str(edge.get("dst_id", ""))
        rel = str(edge.get("relation", ""))
        probes.append(
            {
                "probe_id": f"probe_rel_{eid}",
                "probe_type": "single_turn",
                "question": f"Does object {src} stand in relation '{rel}' to object {dst}? Answer yes or no and explain using visible evidence.",
                "target_claim_id": f"claim_rel_{eid}",
                "evidence_node_ids": [src, dst],
                "evidence_edge_ids": [eid],
            }
        )
    return probes
```

### 9.3 Caption as a supervisory view

```python
from __future__ import annotations

from typing import Dict


def build_caption_prompt(graph: Dict[str, object]) -> str:
    node_lines = []
    for node in graph.get("nodes") or []:
        node_lines.append(
            f"- {node.get('entity_id')}: {node.get('canonical_label')} bbox={node.get('bbox')}"
        )
    edge_lines = []
    for edge in graph.get("edges") or []:
        edge_lines.append(
            f"- {edge.get('src_id')} {edge.get('relation')} {edge.get('dst_id')}"
        )

    return (
        "You are given a scene graph for one frame.\n"
        "Write a technical caption of at most 4 sentences.\n"
        "Requirements:\n"
        "- mention only visible entities supported by the graph\n"
        "- cover important relations\n"
        "- do not invent unseen objects\n"
        "- keep wording canonical when possible\n\n"
        "Nodes:\n"
        + "\n".join(node_lines)
        + "\n\nEdges:\n"
        + "\n".join(edge_lines)
    )
```

### 9.4 Caption to claims

Low-risk strategy:

- do not use open information extraction as the first version
- use constrained extraction against known graph entities, ontology labels, and relation vocabulary

```python
from __future__ import annotations

from typing import Dict, List


def caption_to_claim_votes(
    caption_text: str,
    graph: Dict[str, object],
    ontology,
) -> List[Dict[str, object]]:
    text = str(caption_text or "").strip().lower()
    votes: List[Dict[str, object]] = []

    for node in graph.get("nodes") or []:
        nid = str(node.get("entity_id", ""))
        label = str(node.get("canonical_label", "")).strip().lower()
        if label and label in text:
            votes.append(
                {
                    "claim_id": f"claim_exists_{nid}",
                    "view_type": "caption",
                    "vote": "support",
                    "score": 0.65,
                }
            )
            votes.append(
                {
                    "claim_id": f"claim_label_{nid}",
                    "view_type": "caption",
                    "vote": "support",
                    "score": 0.60,
                }
            )

    for edge in graph.get("edges") or []:
        eid = str(edge.get("edge_id", ""))
        rel = str(edge.get("relation", "")).strip().lower()
        if rel and rel.replace("_", " ") in text:
            votes.append(
                {
                    "claim_id": f"claim_rel_{eid}",
                    "view_type": "caption",
                    "vote": "support",
                    "score": 0.60,
                }
            )

    return votes
```

### 9.5 Conservative consistency scoring

```python
from __future__ import annotations

from typing import Dict, Iterable


def aggregate_claim_scores(
    claims: Dict[str, Claim],
    votes: Iterable[Dict[str, object]],
) -> Dict[str, Claim]:
    for vote in votes:
        cid = str(vote.get("claim_id", ""))
        if cid not in claims:
            continue
        score = float(vote.get("score", 0.0))
        decision = str(vote.get("vote", "")).strip().lower()
        row = claims[cid]
        if decision == "support":
            row.support_score += score
        elif decision == "conflict":
            row.conflict_score += score
        else:
            row.uncertainty_score = max(0.0, row.uncertainty_score - (0.25 * score))

        total = row.support_score + row.conflict_score + 1e-6
        disagreement = abs(row.support_score - row.conflict_score) / total
        row.uncertainty_score = min(row.uncertainty_score, 1.0 - disagreement)
    return claims
```

### 9.6 Belief update

```python
from __future__ import annotations

from typing import Dict


def revise_graph_from_claims(
    graph: Dict[str, object],
    claims: Dict[str, Claim],
    *,
    auto_accept_threshold: float = 0.85,
    auto_reject_threshold: float = 0.80,
) -> Dict[str, object]:
    out = {
        "image_id": graph.get("image_id"),
        "nodes": [dict(x) for x in graph.get("nodes") or []],
        "edges": [dict(x) for x in graph.get("edges") or []],
        "validator_flags": list(graph.get("validator_flags") or []),
        "metadata": dict(graph.get("metadata") or {}),
    }

    node_by_id = {str(n.get("entity_id")): n for n in out["nodes"]}
    edge_by_id = {str(e.get("edge_id")): e for e in out["edges"]}

    for claim in claims.values():
        denom = claim.support_score + claim.conflict_score + 1e-6
        support_ratio = claim.support_score / denom
        conflict_ratio = claim.conflict_score / denom

        if claim.claim_type == "label" and support_ratio >= auto_accept_threshold:
            node = node_by_id.get(claim.subject_id)
            if node is not None and claim.value:
                node["canonical_label"] = claim.value
                node.setdefault("provenance", []).append(
                    {"source": "cycle_refine", "mode": "auto_accept_label", "claim_id": claim.claim_id}
                )

        if claim.claim_type == "relation" and conflict_ratio >= auto_reject_threshold:
            edge = edge_by_id.get(claim.evidence_edge_ids[0] if claim.evidence_edge_ids else "")
            if edge is not None:
                edge.setdefault("validator_flags", []).append("cycle_relation_conflict")
                edge["risk"] = max(float(edge.get("risk", 0.0)), 0.8)

    return out
```

### 9.7 Lightweight human arbitration policy

Transfer the Action Segmentation idea here:

- not just uncertainty
- also correction utility

```python
from __future__ import annotations

from typing import Dict, List


def build_human_arbitration_queue(
    claims: Dict[str, Claim],
    *,
    max_items: int = 3,
) -> List[Dict[str, object]]:
    queue: List[Dict[str, object]] = []
    for claim in claims.values():
        uncertainty = float(claim.uncertainty_score)
        conflict = float(claim.conflict_score)
        scarcity_bonus = 0.15 if claim.claim_type in {"relation", "attribute"} else 0.05
        utility = (0.5 * uncertainty) + (0.35 * conflict) + scarcity_bonus
        if utility < 0.45:
            continue
        queue.append(
            {
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type,
                "priority": utility,
                "question": build_minimal_human_question(claim),
                "subject_id": claim.subject_id,
                "object_id": claim.object_id,
            }
        )
    queue.sort(key=lambda x: float(x["priority"]), reverse=True)
    return queue[: max(1, int(max_items))]


def build_minimal_human_question(claim: Claim) -> str:
    if claim.claim_type == "label":
        return f"Is object {claim.subject_id} correctly labeled as '{claim.value}'?"
    if claim.claim_type == "relation":
        return f"Does relation '{claim.predicate}' hold between {claim.subject_id} and {claim.object_id}?"
    if claim.claim_type == "attribute":
        return f"Is {claim.subject_id} '{claim.predicate}={claim.value}'?"
    return f"Please verify claim {claim.claim_id}."
```

---

## 10. Local Verifier Adapter

### 10.1 Base interface

```python
from __future__ import annotations

from typing import Dict, List, Protocol


class VisionVerifier(Protocol):
    def answer_probe(
        self,
        *,
        image_path: str,
        question: str,
        regions: List[Dict[str, object]],
    ) -> Dict[str, object]:
        ...

    def generate_caption(
        self,
        *,
        image_path: str,
        prompt: str,
        regions: List[Dict[str, object]],
    ) -> Dict[str, object]:
        ...
```

### 10.2 Recommended default local implementation: Qwen2.5-VL

```python
from __future__ import annotations

from typing import Dict, List

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen25VLVerifier:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _run(self, messages: List[Dict[str, object]], max_new_tokens: int = 256) -> str:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[messages[0]["content"][0]["image"]],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return str(output).strip()

    def answer_probe(self, *, image_path: str, question: str, regions: List[Dict[str, object]]) -> Dict[str, object]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file:///{image_path.replace('\\\\', '/')}"},
                    {"type": "text", "text": question + "\\nReturn JSON: {\\\"answer\\\": \\\"yes|no|uncertain\\\", \\\"reason\\\": \\\"...\\\", \\\"score\\\": 0.0}"},
                ],
            }
        ]
        text = self._run(messages)
        return {"raw_text": text}

    def generate_caption(self, *, image_path: str, prompt: str, regions: List[Dict[str, object]]) -> Dict[str, object]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file:///{image_path.replace('\\\\', '/')}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._run(messages)
        return {"caption": text}
```

Note:

- for the first version, do not depend on visual localization output from the VLM
- use the existing graph boxes/masks as the crop/region source
- only ask the VLM to verify or explain

---

## 11. Main Cycle Pipeline

```python
from __future__ import annotations

from typing import Dict, List

from .claim_graph import graph_to_claims, build_single_turn_probes
from .captioning import build_caption_prompt, caption_to_claim_votes
from .consistency import aggregate_claim_scores
from .belief_update import revise_graph_from_claims
from .arbitration import build_human_arbitration_queue


def run_cycle_refine(
    *,
    graph: Dict[str, object],
    image_path: str,
    verifier,
    ontology,
    cfg: Dict[str, object],
) -> Dict[str, object]:
    claims = {c.claim_id: c for c in graph_to_claims(graph)}
    votes: List[Dict[str, object]] = []

    if bool((cfg.get("cycle") or {}).get("enable_single_turn_probes", True)):
        for probe in build_single_turn_probes(graph):
            resp = verifier.answer_probe(
                image_path=image_path,
                question=str(probe["question"]),
                regions=[],
            )
            raw = str(resp.get("raw_text", "")).lower()
            if '"answer": "yes"' in raw or raw.startswith("{\"answer\":\"yes"):
                votes.append(
                    {
                        "claim_id": probe["target_claim_id"],
                        "view_type": "single_turn_vqa",
                        "vote": "support",
                        "score": 0.70,
                    }
                )
            elif '"answer": "no"' in raw or raw.startswith("{\"answer\":\"no"):
                votes.append(
                    {
                        "claim_id": probe["target_claim_id"],
                        "view_type": "single_turn_vqa",
                        "vote": "conflict",
                        "score": 0.70,
                    }
                )

    if bool((cfg.get("cycle") or {}).get("enable_caption_probe", True)):
        prompt = build_caption_prompt(graph)
        caption = verifier.generate_caption(
            image_path=image_path,
            prompt=prompt,
            regions=[],
        )
        caption_votes = caption_to_claim_votes(
            str(caption.get("caption", "")),
            graph,
            ontology=ontology,
        )
        votes.extend(caption_votes)

    claims = aggregate_claim_scores(claims, votes)
    revised_graph = revise_graph_from_claims(
        graph,
        claims,
        auto_accept_threshold=float((cfg.get("cycle") or {}).get("auto_accept_threshold", 0.85)),
        auto_reject_threshold=float((cfg.get("cycle") or {}).get("auto_reject_threshold", 0.80)),
    )
    queue = build_human_arbitration_queue(
        claims,
        max_items=int((cfg.get("cycle") or {}).get("max_human_queries_per_frame", 3)),
    )

    return {
        "graph_before": graph,
        "graph_after": revised_graph,
        "claims": {k: vars(v) for k, v in claims.items()},
        "votes": votes,
        "caption": caption if "caption" in locals() else {},
        "human_queue": queue,
    }
```

---

## 12. Correction Memory

### 12.1 Why this matters

Without correction memory, the system will repeatedly make the same errors:

- same wrong label
- same wrong relation
- same missing attribute
- same prompt phrasing weakness

The Action Segmentation workflow already solved this conceptually.

### 12.2 Recommended memory tables

Store these in JSON or pickle first, SQLite later.

```python
{
  "label_confusions": {
    "cup": {"bottle": 7, "glass": 4}
  },
  "relation_confusions": {
    "on": {"inside": 3, "touching": 5}
  },
  "prompt_aliases": {
    "laptop": ["computer", "notebook computer"]
  },
  "verified_locks": {
    "track_0004": {"status": "confirmed", "frame_start": 120, "frame_end": 148}
  }
}
```

### 12.3 Minimal update rule

```python
from __future__ import annotations

from typing import Dict


def update_memory_from_human_decision(
    memory: Dict[str, object],
    *,
    claim_type: str,
    proposed: str,
    corrected: str,
) -> Dict[str, object]:
    out = dict(memory)
    if claim_type == "label" and proposed and corrected and proposed != corrected:
        table = dict(out.get("label_confusions") or {})
        bucket = dict(table.get(corrected) or {})
        bucket[proposed] = int(bucket.get(proposed, 0)) + 1
        table[corrected] = bucket
        out["label_confusions"] = table
    if claim_type == "relation" and proposed and corrected and proposed != corrected:
        table = dict(out.get("relation_confusions") or {})
        bucket = dict(table.get(corrected) or {})
        bucket[proposed] = int(bucket.get(proposed, 0)) + 1
        table[corrected] = bucket
        out["relation_confusions"] = table
    return out
```

### 12.4 Two-timescale learning

Use the same principle as the Action Segmentation document:

- online timescale:
  - one video
  - one frame bundle
  - immediate corrections
- offline timescale:
  - finalized sessions
  - merge correction memory across videos
  - retrain or recalibrate prompt ranking and claim priors

---

## 13. Human Arbitration Design

### 13.1 Never ask the human to review everything

Ask only the smallest question that breaks the ambiguity.

Good examples:

- "Is object `track_0003` a `cup`?"
- "Does `person` hold `cup`?"
- "Should the caption mention a `laptop`?"

Bad examples:

- "Please rewrite this entire scene graph."
- "Please re-caption the whole frame."

### 13.2 Minimal interaction set

Recommended actions:

- accept claim
- reject claim
- relabel entity
- switch relation
- mark unsupported caption mention
- lock verified track for N frames

### 13.3 Region locking

Once a node or relation has strong human confirmation:

- keep it locked for short temporal windows
- allow only downstream evidence to reduce confidence, not overwrite directly

This avoids model-human tug-of-war.

---

## 14. Evaluation Plan

### 14.1 Primary metrics

Keep current graph/VQA metrics, but add these new ones.

#### Graph metrics

- entity label accuracy
- mask IoU accuracy
- relation F1
- attribute F1
- graph edit distance

#### Cross-task consistency metrics

- claim agreement rate
- graph-caption contradiction rate
- graph-VQA contradiction rate
- multi-turn chain consistency

#### Human efficiency metrics

- human queries per frame
- edits per frame
- mean time to verified graph
- automatic resolution rate before human review

#### Caption metrics

Inspired by CompreCap:

- object coverage
- attribute correctness
- relation coverage
- hallucination rate

### 14.2 Required ablations

1. Backbone only
2. Backbone + single-turn verification
3. Backbone + single-turn + caption verification
4. Backbone + full cycle
5. Backbone + full cycle + correction memory
6. Backbone + full cycle + human arbitration

### 14.3 Datasets

Recommended:

- PVSG for mask-grounded video scene graph relevance
- Action Genome / AGQA for compositional QA relevance
- your own tool-collected validation set for human-time metrics

---

## 15. Why This Path Is Better Than Alternatives

### 15.1 Better than "just use a stronger VLM to generate the graph"

Because:

- graph failures become opaque
- debugging is hard
- reproducibility is worse
- human correction is harder to localize

### 15.2 Better than "just polish the current engineering tool"

Because:

- it becomes a method paper, not only a tool paper
- the four tasks become algorithmically connected
- evaluation becomes scientifically meaningful

### 15.3 Better than "train a full video PSG model"

Because:

- data is limited
- training risk is high
- implementation cost is large
- the repository already has a modular interactive platform that benefits more from verification and memory

---

## 16. Concrete Build Order

If you want the fastest path to a strong prototype, implement in this exact order.

### Week 1

- add `cycle_types.py`
- add `claim_graph.py`
- convert graph to atomic claims
- write tests for claim extraction

### Week 2

- add `captioning.py`
- replace heuristic caption with graph-grounded caption prompt
- add constrained caption-to-claim votes

### Week 3

- add `mllm_adapters/base.py`
- add `mllm_adapters/qwen25_vl.py`
- run local single-turn verification on frames

### Week 4

- add `consistency.py`
- add `belief_update.py`
- add `arbitration.py`
- produce first cycle-refined graph

### Week 5

- add `correction_memory.py`
- reuse validation log semantics
- persist label/relation confusion memory

### Week 6

- add `tools/run_cycle_refine.py`
- add new JSON outputs
- add evaluation script extensions

### Week 7+

- optional closed-model escalation
- optional UI panel for conflict review
- optional offline cross-video consolidation

---

## 17. Minimal CLI

```python
import argparse
import json
import os

from core.impact_sg.ontology import load_ontology
from core.impact_sg.pipeline import run_build_scene_graph
from core.impact_sg.cycle_pipeline import run_cycle_refine
from core.impact_sg.mllm_adapters.qwen25_vl import Qwen25VLVerifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--ontology", default="configs/impact_sg_ontology.json")
    parser.add_argument("--pipeline_cfg", default="configs/impact_sg_pipeline.json")
    parser.add_argument("--cycle_cfg", default="configs/impact_cycle.json")
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ontology = load_ontology(args.ontology)
    graph = run_build_scene_graph(
        image_id=args.image_id,
        image_path=args.image_path,
        ontology_path=args.ontology,
        pipeline_cfg_path=args.pipeline_cfg,
        image_size=(args.image_width, args.image_height),
    )
    with open(args.cycle_cfg, "r", encoding="utf-8") as f:
        cycle_cfg = json.load(f)

    verifier = Qwen25VLVerifier(
        model_id=((cycle_cfg.get("local_verifier") or {}).get("model_id") or "Qwen/Qwen2.5-VL-7B-Instruct")
    )
    result = run_cycle_refine(
        graph=graph,
        image_path=os.path.abspath(args.image_path),
        verifier=verifier,
        ontology=ontology,
        cfg=cycle_cfg,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
```

---

## 18. Final Recommendation Summary

### Chosen path

Choose:

`Mask-first graph backbone + local VLM verification + graph/VQA/caption mutual supervision + utility-based lightweight human arbitration + correction memory`

### Default model stack

Default local verifier:

- `Qwen/Qwen2.5-VL-7B-Instruct`

Optional stronger local verifier:

- `InternVL2.5-8B` or `InternVL3`

Optional API escalation:

- any current strong multimodal API, only for top conflicts

### Paper-friendly contribution statement

1. A unified four-task cycle where scene graph, VQA, and caption supervise each other.
2. A conservative claim-level belief update mechanism that turns multi-view agreement into graph refinement.
3. A human-light arbitration policy that requests only high-utility residual verifications.
4. A correction-memory mechanism that transfers interactive learning ideas from temporal annotation to graph-centric video semantics.

### What not to do

- do not make the VLM the only graph generator
- do not rely on free-form caption text to directly overwrite the graph
- do not send every conflict to human review
- do not claim a fully learned end-to-end video PSG model

---

## 19. Source Notes

Primary sources used to determine the recommended path:

- IEEE SMC 2026 event page: human-centric intelligence framing
- IEEE SMC 2025 CFP: human-machine systems, machine vision, image processing, AI
- CVPR 2023 PVSG
- ICCVW 2023 Knowledge Informed Sequential Scene Graph Verification Using VQA
- CVPR 2024 From Pixels to Graphs
- CVPR 2025 CompreCap
- CVPR 2025 MetaVQA
- CVPR 2025 4D PSG generation
- official Qwen2.5-VL model card
- official InternVL repository

These sources indicate that:

- graph verification via VQA is already a credible research direction
- caption evaluation is moving toward graph-centric coverage
- open-vocabulary VLM-based graph reasoning is real but still not the safest end-to-end production strategy
- practical local multimodal verification is now mature enough for a low-risk implementation
