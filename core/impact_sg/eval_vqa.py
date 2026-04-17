from __future__ import annotations

from typing import Dict, List


def evaluate_vqa(pred_items: List[Dict[str, object]], gt_items: List[Dict[str, object]]) -> Dict[str, float]:
    gt_by_qid = {str(x.get("qid")): x for x in gt_items}
    correct = 0
    total = 0
    grounding_ok = 0
    chain_ok = 0
    rewrite_count = 0
    mismatch = 0

    chain_to_turns = {}

    for item in pred_items:
        qid = str(item.get("qid", ""))
        if not qid:
            continue
        total += 1
        gt = gt_by_qid.get(qid)
        if gt and str(item.get("answer", "")).strip().lower() == str(gt.get("answer", "")).strip().lower():
            correct += 1

        p_nodes = set(str(x) for x in (item.get("evidence_node_ids") or []))
        p_edges = set(str(x) for x in (item.get("evidence_edge_ids") or []))
        g_nodes = set(str(x) for x in ((gt or {}).get("evidence_node_ids") or []))
        g_edges = set(str(x) for x in ((gt or {}).get("evidence_edge_ids") or []))
        if p_nodes == g_nodes and p_edges == g_edges:
            grounding_ok += 1

        if item.get("validator_flags"):
            mismatch += 1

        if bool(item.get("rewritten", False)):
            rewrite_count += 1

        chain = str(item.get("chain_id", "")).strip()
        turn = int(item.get("turn", 0) or 0)
        if chain and turn > 0:
            chain_to_turns.setdefault(chain, []).append(turn)

    for turns in chain_to_turns.values():
        if sorted(turns) == list(range(1, max(turns) + 1)):
            chain_ok += 1

    n = max(1, total)
    c = max(1, len(chain_to_turns))
    return {
        "answer_accuracy": float(correct) / float(n),
        "evidence_grounding_accuracy": float(grounding_ok) / float(n),
        "chain_consistency": float(chain_ok) / float(c),
        "human_rewrite_rate": float(rewrite_count) / float(n),
        "answer_evidence_mismatch_rate": float(mismatch) / float(n),
    }
