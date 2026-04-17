from __future__ import annotations

from typing import Dict, List, Optional


class AnswerEngine:
    def __init__(self, prefer_deterministic: bool = True, open_ended_hook=None):
        self.prefer_deterministic = bool(prefer_deterministic)
        self.open_ended_hook = open_ended_hook

    def answer(self, question_item: Dict[str, object], graph: Dict[str, object]) -> Dict[str, object]:
        answer_type = str(question_item.get("answer_type", "")).strip().lower()
        if self.prefer_deterministic and answer_type in {"boolean", "count", "attribute", "relation", "reference"}:
            return {
                "answer": str(question_item.get("answer", "")),
                "evidence_node_ids": list(question_item.get("evidence_node_ids") or []),
                "evidence_edge_ids": list(question_item.get("evidence_edge_ids") or []),
                "mode": "deterministic_graph",
            }

        if callable(self.open_ended_hook):
            subgraph = {
                "nodes": [n for n in (graph.get("nodes") or []) if n.get("entity_id") in set(question_item.get("evidence_node_ids") or [])],
                "edges": [e for e in (graph.get("edges") or []) if e.get("edge_id") in set(question_item.get("evidence_edge_ids") or [])],
            }
            response = self.open_ended_hook(question_item=question_item, subgraph=subgraph)
            if isinstance(response, dict):
                return {
                    "answer": str(response.get("answer", "")),
                    "evidence_node_ids": list(question_item.get("evidence_node_ids") or []),
                    "evidence_edge_ids": list(question_item.get("evidence_edge_ids") or []),
                    "mode": "constrained_open_ended",
                }

        return {
            "answer": str(question_item.get("answer", "")),
            "evidence_node_ids": list(question_item.get("evidence_node_ids") or []),
            "evidence_edge_ids": list(question_item.get("evidence_edge_ids") or []),
            "mode": "fallback_cached_answer",
        }
