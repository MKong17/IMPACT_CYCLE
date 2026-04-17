# Prompt Inventory (Current Effective Prompts)

## 1) Qwen per-frame person attributes prompt
- File: `tools/qwen_batch_summary.py`
- Function: `_person_prompt(graph)`
- Called from: `main()` Step 1, per frame when persons exist.
- Current prompt text (exact):

```
You are a visual attribute extractor for PERSON nodes.
Focus ONLY on the TOP-3 persons by bbox area in this frame.
For each selected person entity_id, output JSON list rows with fixed slots:
state, pose, apparel, action, emotion.
Return ONLY JSON array. If uncertain, use empty string.
Do not create entity_id that is not in the selected list.
Schema:
[{"entity_id":"...","attributes":{"state":"","pose":"","apparel":"","action":"","emotion":""}}]
...
JSON:
```

## 2) Qwen 5-frame global summary prompt
- File: `tools/qwen_batch_summary.py`
- Function: `_make_prompt(chunk_graphs)`
- Called from: `main()` Step 2, every batch (`--batch_size`, default 5).
- Current prompt text (exact):

```
You are a video understanding assistant.
Given 5-frame scene-graph digests, produce ONE concise global semantic summary sentence.
Focus on dominant actors, actions, and spatial context continuity.
Do not output JSON.
...
Summary:
```

## 3) SAM / SceneGraph category prompts source
- Main config file: `configs/impact_sg_pipeline.json`
  - `ontology_prompt_pack_file`: `tmp/sg_prompts_dump.txt`
  - `proposal.person_focus_prompts`: `["person"]`
- Prompt pack file currently used: `tmp/sg_prompts_dump.txt`
  - Contains:
    - `=== CATEGORY PROMPTS ===`
    - `=== SENTENCE PROMPTS ===`
- Loader/caller:
  - File: `core/impact_sg/pipeline.py`
  - `_load_prompt_pack(path)`
  - in `run_build_scene_graph(...)` around prompt-bank build and `backend.discover_entities_by_category(...)`

## 4) Person-priority fallback prompts (hardcoded)
- File: `core/impact_sg/pipeline.py`
- Constant: `HUMAN_PRIORITY_PROMPTS`
- Current values:

```
person
player
football player
goalkeeper
referee
athlete
man
```

- Used by: `_sports_human_prompt_items(...)` and fallback discover pass.

## 5) Person-focus first-pass prompts
- File: `core/impact_sg/pipeline.py`
- Function: `_person_focus_prompt_items(ontology, proposal_cfg)`
- Source: `proposal.person_focus_prompts` in `configs/impact_sg_pipeline.json`
- Default if empty: `["person"]`
- Used before main discover pass.

## 6) Temporary prompts from UI (runtime override)
- File: `ui/video_task_studio.py`
- UI field: `Temp Prompts`
- Parser: `_parse_temporary_prompt_entities(text)`
- Runtime injection: `_scene_graph_runtime_ontology()`
- Format expected per line:
  - `label`
  - or `label: variant1, variant2, ...`

## 7) Optional temporal tracking prompt (currently not main path)
- File: `core/impact_sg/qwen_temporal_tracking.py`
- Constant: `DEFAULT_PROMPT` (Chinese)
- Used by: `run_temporal_tracking(..., system_prompt=DEFAULT_PROMPT)`

## 8) Captioning agent prompt (cycle pipeline side)
- File: `core/impact_sg/captioning.py`
- Function: `build_caption_prompt(...)`
- Structured mode prompt starts with:

```
You are the Captioning agent in a multi-agent scene graph verification loop.
...
Return JSON only with this schema:
```

- Used by: `core/impact_sg/cycle_pipeline.py` in cycle refine flow.

## 9) Where LLM summary worker is launched
- File: `ui/video_task_studio.py`
- Class: `LLMBatchSummaryWorker`
- Entry: `_start_llm_batch_summary_worker(...)`
- Script called: `tools/qwen_batch_summary.py`

