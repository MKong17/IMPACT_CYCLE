# IMPACT-SG Cycle Logic

**Date**: 2026-04-16  
**Scope**: Current implementation of the scene-graph cycle verification loop, including full cycle refine, targeted rerun after human arbitration, and UI integration.

---

## 1. 这套 cycle 在做什么

`cycle` 的职责不是重新生成 scene graph，而是对现有 scene graph 做一轮“结构化复核”。

它把当前图拆成一组可验证的 claim，然后用多种 verifier 视角去交叉检查：

- single-turn VQA：低成本、本地化、二分类或受限纠错问题
- multi-turn VQA：带时序/链式上下文的追问
- caption probe：整体描述视角，对现有图做 holistic consistency check
- HITL queue：对自动系统仍然拿不准的 claim 发起人工仲裁
- geometry review：对空间关系冲突生成 bbox 修复候选

最终输出不是一句话答案，而是一整包结构化结果：

- `graph_after`：根据投票自动修正后的图
- `claims`：每个 claim 的当前支持/冲突状态
- `votes`：所有 probe/caption 产生的投票
- `probe_results`：每个 probe 的问题、响应、schema 状态、原始文本
- `caption`：caption 文本、结构化反馈和 caption-derived votes
- `human_queue`：当前仍需要人工处理的问题
- `summary` / `metrics` / `agents` / `runtime`

核心入口在：

- `core/impact_sg/cycle_pipeline.py::run_cycle_refine`
- `core/impact_sg/cycle_pipeline.py::rerun_cycle_refine_for_claims`

---

## 2. 从 UI 到底层的两条主路径

当前 UI 里有两条和 cycle 相关的主路径。

### 2.1 全量 cycle refine

用户在 Video Scene Graph 页面主动运行 cycle verify 时，UI 会调用：

- `ui/video_task_studio.py::_run_cycle_refine_for_current_graph`

这个函数会：

1. 检查 scene graph job / cycle job 是否忙
2. 检查 verifier 鉴权是否就绪
3. 解析当前 frame image 路径
4. 准备 cycle config
5. 构造 `CycleRefineWorker`
6. 在线程中执行 `run_cycle_refine(...)`
7. 在完成后把结果回写到 UI

相关函数：

- `ui/video_task_studio.py::CycleRefineWorker.run`
- `ui/video_task_studio.py::_on_cycle_worker_done`
- `ui/video_task_studio.py::_apply_cycle_result`

### 2.2 人工仲裁后的 targeted rerun

当用户在 Human Edit Queue / cycle review 面板里确认一个仲裁结果时，UI 会调用：

- `ui/video_task_studio.py::_apply_change_decision`

如果这次仲裁是 `bbox` 修正，当前逻辑不是清空旧结果，而是：

1. 先把修正应用到 `current_graph`
2. 如果是 geometry 变更，则重建 spatial edges
3. 计算图前后差异 scope
4. 把相关 probe 标记为 stale
5. 发起 `rerun_cycle_refine_for_claims(...)`

相关函数：

- `ui/video_task_studio.py::_apply_cycle_arbitration_change_to_graph`
- `ui/video_task_studio.py::_rebuild_scene_graph_after_geometry_change`
- `ui/video_task_studio.py::_graph_delta_target_scope`
- `ui/video_task_studio.py::_rerun_cycle_refine_after_graph_change`

这条 targeted 路径的目标是：

- 不重算整张图里所有 claim
- 但会把与变更节点/边结构上相关的 claim 一起重算
- 并且把旧的相关 votes / probe_results / human_queue 局部替换掉

---

## 3. Verifier 是怎么选的

verifier 创建逻辑在：

- `core/impact_sg/mllm_adapters/factory.py::build_vision_verifier`

当前默认 provider 常量在：

- `core/impact_sg/mllm_adapters/defaults.py`

目前默认值是：

- `DEFAULT_CYCLE_PROVIDER = "gemini_api"`

对应配置文件通常来自：

- `configs/impact_cycle.json`

按当前代码，provider 选择优先级大致是：

1. 如果 `runtime.preferred_provider` 明确指定，就按指定 provider 走
2. 否则如果 `api_verifier.enabled=true` 且 provider 是 Gemini/OpenAI，则优先走 API verifier
3. 再否则尝试本地 `qwen25_vl`
4. 如果允许 fallback，再退回 mock

`configs/impact_cycle.json` 当前默认设置是：

- API verifier 开启
- `api_verifier.provider = "gemini"`
- `runtime.preferred_provider = "gemini_api"`
- `allow_mock_fallback = false`

也就是说，当前正常配置下，cycle 默认是优先走 Gemini API，而不是 Gemini Online 网页模式，也不是本地 Qwen。

---

## 4. full cycle refine 的底层主循环

主流程函数：

- `core/impact_sg/cycle_pipeline.py::run_cycle_refine`

### 4.1 输入标准化

`run_cycle_refine(...)` 先把输入 graph 规整成一个内部工作副本：

- 保留 `image_id`
- 复制 `nodes`
- 复制 `edges`
- 复制 `validator_flags`
- 复制 `metadata`
- 强制写入 `metadata.image_path`

这样做的目的有两个：

- 保证 cycle 不直接修改外部传入对象
- 后续所有 round 都在同一个 `current_graph` 上演化

### 4.2 correction memory

接着它会调用：

- `core/impact_sg/cycle_pipeline.py::_effective_correction_memory`

这里会根据 config 开关决定哪些 memory 生效：

- label confusion memory
- relation confusion memory
- prompt alias memory
- verified locks

这部分 memory 会影响：

- prompt wording
- confusion guidance
- auto accept / auto reject threshold 的微调
- HITL queue 的排序

### 4.3 round 循环

主循环最多跑：

- `cycle.max_revision_rounds`

当前默认配置里是 `2`。

每一轮都会做同样的事情：

1. 构建 focus graph
2. 从 focus graph 生成 claims
3. 生成 single-turn probes
4. 生成 multi-turn probes
5. 生成 caption probe
6. 把所有 vote 做 role policy 加权
7. 聚合 claim score
8. 根据 claim 自动修图
9. 生成 human queue 与 geometry queue
10. 写入 round payload
11. 如果图没有变化，则提前结束

---

## 5. focus graph：先裁剪“需要看的图”

函数：

- `core/impact_sg/claim_graph.py::build_focus_graph`

如果 `enable_person_focus=false`，cycle 直接在整张图上工作。

如果开启 focus，它会：

1. 找到 `focus_subject_label` 对应的节点，默认通常是 `person`
2. 按 score 排序，保留前 `max_subjects`
3. 以这些 subject 为起点，在图里做 `max_hops` 扩散
4. 只保留 focus 区域内的 nodes / edges
5. 在 `metadata.focus_filter` 中记录本轮 focus 的裁剪信息

这一步的作用是：

- 把 cycle 限制在任务更关心的人物或主体附近
- 减少 probe 数量
- 让 caption / VQA 聚焦在当前任务相关区域

需要注意：

- targeted rerun 时，claim scope 是基于 `current_graph` 先算出来的
- 然后实际 probe / claim 生成仍然在 `focused_graph` 上进行
- 所以如果某个目标 claim 被 focus 完全裁掉，targeted rerun 可能退化为全量 rerun，或只能保留当前图而不生成新的相关 probe

---

## 6. claim 是怎么从 graph 里抽出来的

函数：

- `core/impact_sg/claim_graph.py::graph_to_claims`

当前设计不是“每个 node/edge 都无脑变成 claim”，而是有选择地把“不确定、可疑、值得复核”的部分转换成 claim。

### 6.1 node 相关 claim

对 node 来说，只有满足以下任一条件，才会生成 node-level claim：

- label 是 generic label
- prior confidence 低于阈值
- node 上有冲突/低置信 flag

这时会生成：

- `claim_exists_<node_id>`
- `claim_label_<node_id>`

### 6.2 attribute claim

attribute claim 会在以下情况下生成：

- attribute 本身低置信
- 或 node 本身不确定

claim id 形式为：

- `claim_attr_<node_id>_<slot>`

### 6.3 relation claim

relation claim 会在以下情况下生成：

- edge 低置信
- edge 有 relation conflict flag
- 或 relation 是 spatial relation

claim id 形式为：

- `claim_rel_<edge_id>`

这意味着当前 cycle 对 spatial relation 是天然更敏感的，即使 edge 不是低置信，也会更倾向于纳入复核。

---

## 7. single-turn probes：低成本局部验证

函数：

- `core/impact_sg/claim_graph.py::build_single_turn_probes`

single-turn probe 的特点是：

- 一个问题只盯一个 claim
- 问题尽量局部、直接、低歧义
- 支持 yes/no 和 constrained selection 两种形式

### 7.1 node existence / label

常见 probe 包括：

- existence：这个对象是否真的在图像中可见
- label verification：这个对象是否真的是当前 canonical label
- counterfactual label：是否更像另一个标签
- constrained correction：如果现在标签错了，应该从候选 canonical label 中选哪个

### 7.2 attribute

如果 attribute 足够重要且低置信，会生成：

- binary verification
- counterfactual verification

### 7.3 relation

relation 侧也会生成：

- relation 是否成立
- alternative relation 的 counterfactual
- 如果当前 relation 不对，应该选哪个 canonical relation

对于 selection 类 probe，`response_format` 会显式附带：

- `type = "selection"`
- `options`
- `default_selection`

这样 verifier 会被严格约束在候选集合内回答，而不是自由生成。

---

## 8. multi-turn probes：链式和时序复核

函数：

- `core/impact_sg/claim_graph.py::build_multi_turn_probes`

multi-turn 不是简单地把 single-turn 换个名字，而是把“同一个 track / 同一个主体”串成一个 chain。

一个 chain 通常会包含这些类型的问题：

- temporal consistency：这个 track 是否还是同一个对象
- 当前 label 是否仍成立
- counterfactual label
- constrained correction
- relation 连续性或组合验证

每个 chain 会有：

- `chain_id`
- `turn`
- `probe_family`
- 可选的 `temporal_anchor_frame_idx`

如果图的 `metadata.temporal_context` 提供了前后 frame 线索，multi-turn 就会把这些信息带进 prompt。

当前 role policy 里，`temporal_consistency` 的权重高于普通 multi-turn binary verification，说明系统默认更信任时序一致性证据。

---

## 9. prompt 是怎么变成“claim-aware”的

单看 `question` 文本并不足以让 verifier 稳定工作，所以 cycle 还会给 probe 生成一份内部 prompt context。

相关函数：

- `core/impact_sg/cycle_pipeline.py::_probe_prompt_context`
- `core/impact_sg/cycle_pipeline.py::_probe_response_format`
- `core/impact_sg/mllm_adapters/api_verifier.py`

### 9.1 `_probe_prompt_context` 包含什么

它会从 probe 和 graph 中提取上下文：

- `claim_type`
- `probe_family`
- `subject_id` / `subject_label`
- `object_id` / `object_label`
- `current_value`
- `slot`
- `relation`
- `relation_group`
- `is_spatial`
- `temporal_anchor_frame_idx`

### 9.2 verifier 如何使用这些上下文

API verifier 会把这些上下文组织成额外约束，例如：

- 这是 label / attribute / relation / existence 哪一种任务
- 这是 spatial relation，要求模型更多依赖几何关系
- 这是 counterfactual verification，不要按默认直觉回答 yes
- 这是 constrained correction，必须从 canonical options 里选

这些上下文不会直接原样发给 generic transport 层；内部字段会被清洗，避免把 `_prompt_context` 当成外部 API schema 的一部分错误发送。

这一步是当前 cycle 稳定性的关键之一，因为它让“同样是 yes/no 问题”的 relation task 和 label task 拥有不同的行为约束。

---

## 10. probe batch 执行：真正调用 verifier 的地方

函数：

- `core/impact_sg/cycle_pipeline.py::_run_probe_batch`

这个函数对每个 probe 做下面几件事：

1. 根据 probe 生成 `response_format`
2. 根据 probe 类型生成 schema
3. 调用 `verifier.answer_probe(...)`
4. 如果异常，则把结果降级为 `invalid_response`
5. 用 `_probe_to_vote(...)` 把响应转成 vote
6. 把原始响应、schema 状态、provider 等信息存入 `probe_results`

### 10.1 response 到 vote 的映射

函数：

- `core/impact_sg/cycle_pipeline.py::_probe_to_vote`

二值题逻辑：

- 回答与 `expected_answer` 一致 => `support`
- 相反 => `conflict`
- 无法解析或 invalid => `uncertain`

selection 题逻辑：

- 选中当前默认值 => `support`
- 选中别的 canonical option => `conflict`
- 同时把这个 option 记成 `correction_value`

这个 `correction_value` 后续会进入 `correction_candidates`，驱动自动纠错。

### 10.2 invalid response 的处理

如果 verifier 出错、截断、schema 不合法，cycle 不会整轮崩掉，而是把该 probe 记成：

- `answer = uncertain`
- `reason = invalid_response`

这样整轮 cycle 还能继续运行，只是该 probe 对 claim 的贡献变弱。

---

## 11. caption probe：整体视角的结构化复核

相关函数：

- `core/impact_sg/captioning.py::build_caption_prompt`
- `core/impact_sg/captioning.py::caption_to_claim_feedback`

caption probe 的定位不是“生成漂亮描述”，而是：

- 从整体场景角度复核现有 graph claim
- 同时输出 caption 与结构化反馈

### 11.1 prompt 长什么样

`build_caption_prompt(...)` 会把这些信息放进 prompt：

- nodes 列表
- attributes 列表
- relations 列表
- canonical naming hints
- 只能围绕已列出的 graph item 描述，不能凭空造结构

如果 `structured_feedback=true`，它还会明确要求模型返回 JSON：

- `caption`
- `supported_entities`
- `unsupported_entities`
- `supported_attributes`
- `unsupported_attributes`
- `supported_relations`
- `unsupported_relations`
- `hallucinated_mentions`

### 11.2 caption 如何转成 votes

`caption_to_claim_feedback(...)` 有两条路径：

1. 如果能拿到结构化 JSON，就按结构化字段生成 claim vote
2. 如果结构化失败，就退回 caption text fallback，做较弱的文本匹配投票

结构化路径下，caption 会产生：

- entity existence / label support/conflict votes
- attribute support/conflict votes
- relation support/conflict votes

当前 role policy 默认对 caption label 的权重是 `0.0`，除非显式打开 `caption_label_enabled`。这意味着 caption 默认更适合帮助 existence / attribute / relation，而不是直接决定 label。

---

## 12. role policy：不同 agent 的票权不同

函数：

- `core/impact_sg/visual_verifier/policy.py::apply_role_policy`

系统不会把所有 vote 当作等权。

当前默认策略大致是：

- single-turn：大部分 claim type 权重都是 1.0
- multi-turn：
  - `temporal_consistency` = 1.0
  - `binary_verification` = 0.8
  - `constrained_correction` = 0.8
  - `counterfactual_verification` = 0.7
- caption：
  - existence = 0.7
  - attribute = 0.7
  - relation = 0.7
  - label = 0.0

这意味着当前系统的默认偏好是：

- 最信单点局部验证和时序一致性
- caption 作为补充，不直接主导 label 判定

加权之后，每条 vote 会携带：

- `base_score`
- `weight`
- 加权后的 `score`
- `role_policy_applied`

---

## 13. claim score 聚合

函数：

- `core/impact_sg/consistency.py::aggregate_claim_scores`

每个 claim 会累计三类信号：

- `support_score`
- `conflict_score`
- `uncertainty_score`

聚合规则：

- `support` vote 增加 `support_score`
- `conflict` vote 增加 `conflict_score`
- `uncertain` vote 提高 `uncertainty_score`

然后根据支持/冲突比例推断 claim status：

- `supported`
- `conflicted`
- `uncertain`
- `unreviewed`

这一步的输出还只是 claim 层，还没有真正改动 graph。

---

## 14. correction candidates：如何从“冲突”走向“自动纠错”

函数：

- `core/impact_sg/cycle_pipeline.py::_collect_correction_candidates`

只要某个 selection probe 选到了“不是当前值的 canonical option”，系统就会把这个 option 作为一个 correction candidate。

最终每个 claim 会得到一个 bucket：

- `scores`
- `options`
- `ranked`
- `best_value`
- `best_score`

后面的 `revise_graph_from_claims(...)` 会用这个结构来决定是否自动纠错。

---

## 15. revise_graph_from_claims：真正改图的地方

函数：

- `core/impact_sg/belief_update.py::revise_graph_from_claims`

这一步会把 claim 层的结论投射回 graph。

### 15.1 memory-adjusted thresholds

在自动改图之前，每个 claim 还会经过一层 memory 调整：

- confusion penalty
- lock bonus
- prior bonus
- low prior penalty

这些量会共同影响：

- `effective_support`
- `effective_conflict`
- `accept_threshold`
- `reject_threshold`

也就是说，自动接受/自动拒绝不是死阈值，而是会被 correction memory 和 verified lock 微调。

### 15.2 对不同 claim_type 的处理

label：

- 如果 correction candidate 比当前值更强，可能自动改 label
- 否则如果 support 足够高，自动 accept 当前 label
- 如果 conflict 足够高，给 node 打 `cycle_label_conflict`

attribute：

- support 高则自动写回 attribute
- conflict 高则打 `cycle_attribute_conflict`

existence：

- 如果强冲突且 support 很低，可能自动删除 node
- 否则标记 `cycle_existence_conflict`

relation：

- 如果 correction candidate 够强，可能自动改 relation
- support 高则 accept
- conflict 高则打 `cycle_relation_conflict`

### 15.3 cycle_update

每轮修图后都会在 `graph_after.metadata.cycle_update` 中写入：

- `accepted_claim_ids`
- `flagged_claim_ids`
- `memory_adjustments`
- `correction_applied`
- `auto_removed_node_ids`
- `auto_removed_claim_ids`

这是 UI 和 summary 读取 cycle 决策结果的主要入口之一。

---

## 16. Human queue：哪些问题留给人

函数：

- `core/impact_sg/arbitration.py::build_human_arbitration_queue`

对于自动系统仍然没有把握的问题，cycle 会把它放进 `human_queue`。

排序时考虑的因素包括：

- `uncertainty_score`
- `conflict_ratio`
- claim type 的稀缺性 bonus
- confusion frequency 带来的 memory bonus
- 是否存在 correction candidate 及其分数
- 是否存在 verified lock

输出项里会包含：

- `question`
- `subject_id` / `object_id`
- `claim_row`
- `question_options`
- `suggested_value`
- `suggested_score`

也就是说，human queue 不只是一个“有问题请看一下”的列表，而是已经把模型建议和证据一起整理好了。

---

## 17. geometry review：空间关系冲突如何变成 bbox 修复候选

函数：

- `core/impact_sg/geometry_review.py::build_geometry_review_queue`

这是当前 cycle 很关键的一条增强链路。

### 17.1 触发条件

只有满足这些条件才会进入 geometry review：

- claim 是 relation
- relation 在 spatial vocab 里
- `conflict_ratio >= geometry_conflict_threshold`
- subject/object bbox 都有效

### 17.2 它做了什么

geometry review 不直接问“relation 对不对”，而是问：

- 哪个 bbox 改法最可能恢复这个 spatial relation

它会：

1. 根据 anchor label 或 node confidence 决定移动 subject 还是 object
2. 生成一组 bbox candidate
3. 计算每个 candidate 对 relation 的恢复分数
4. 保留比 `keep_current` 明显更优的候选
5. 生成一个 `claim_type = "bbox"` 的 human queue 项

这个 queue 项会带：

- `resolution_options`
- `geometry_candidates`
- `target_node_id`
- `source_relation_claim_id`
- `source_relation_edge_id`
- `suggested_value`
- `suggested_score`

所以 UI 可以直接把它渲染成“选择哪个 box 更合理”的仲裁面板，而不是只显示一句文本问题。

---

## 18. round 停止条件

`run_cycle_refine(...)` 不是固定死跑满所有 round。

当前停止条件是：

- 如果 `revised_graph == current_graph`，说明本轮没有进一步修改图，就提前结束
- 否则继续下一轮，最多到 `max_revision_rounds`

这意味着 cycle 的行为更接近“直到收敛或达到上限”。

---

## 19. full cycle 的输出结构

最终结果由 `run_cycle_refine(...)` 组装，主要字段包括：

- `graph_before`
- `graph_after`
- `rounds`
- `claims`
- `votes`
- `correction_candidates`
- `probe_results`
- `caption`
- `human_queue`
- `policy`
- `memory`
- `focus`
- `agents`
- `summary`
- `metrics`

### 19.1 `agents`

`_build_agent_summary(...)` 会用 agent 视角描述这轮 cycle：

- scene_graph_backbone
- single_turn_vqa
- multi_turn_vqa
- captioning
- hitl

### 19.2 `summary`

`_cycle_summary(...)` 里会汇总：

- rounds_run
- probe_count
- single_turn_count
- multi_turn_count
- caption_vote_count
- queue_count
- geometry_queue_count
- accepted_claim_count
- flagged_claim_count
- memory_adjusted_count
- verifier_provider / verifier_model_id

### 19.3 `metrics`

`core/impact_sg/eval_cycle.py::evaluate_cycle_result` 会计算：

- `claim_agreement_rate`
- `graph_caption_contradiction_rate`
- `graph_vqa_contradiction_rate`
- `human_queries_per_frame`
- `automatic_resolution_rate_before_human_review`
- `uncertain_claim_rate`
- `support_conflict_margin_mean`
- `caption_hallucination_count`
- `caption_structured_feedback_rate`
- `temporal_multi_turn_share`

---

## 20. targeted rerun：为什么不是“全清空重跑”

函数：

- `core/impact_sg/cycle_pipeline.py::rerun_cycle_refine_for_claims`

这是当前为了 bbox / relation / label 人工修正而做的重点增强。

### 20.1 它解决的问题

旧思路是：

- 图改了以后，把旧 cycle 结果清空
- 然后整图再跑

问题是：

- 代价高
- 容易把和本次改动无关的结果也重新扰动
- UI 体验差

现在的 targeted rerun 则是：

- 只重算受影响 scope
- 保留无关 scope 的旧结果
- 把相关 votes / probe_results / queue 局部替换

### 20.2 scope 是怎么定义的

相关函数：

- `core/impact_sg/cycle_pipeline.py::_target_scope_from_claim_ids`
- `core/impact_sg/cycle_pipeline.py::_row_matches_target_scope`
- `core/impact_sg/cycle_pipeline.py::_select_probes_for_target_scope`

scope 不只是“同 claim id”，而是会沿着结构关系扩展：

- claim 对应的 node
- claim 对应的 edge
- row 的 `evidence_node_ids`
- row 的 `evidence_edge_ids`
- `source_relation_claim_id`
- `claim_row.provenance[].relation_claim_id`

这也是为什么现在修一个 bbox 后，与该节点相连的 spatial relation probe 会被一起纳入重算。

### 20.3 targeted rerun 的步骤

`rerun_cycle_refine_for_claims(...)` 大致做下面这些事情：

1. 用当前图和目标 claim 生成 scope
2. 在 `focused_graph` 上重新生成 claims
3. 自动把与 scope 结构相关的 claim 一起纳入 `target_ids`
4. 仅选择落在 scope 里的 single-turn / multi-turn probes
5. caption 也只保留 scope 内的 caption votes / report
6. 对这些 votes 重新做 role policy
7. 只用目标 claims 去修正当前图
8. 重新生成目标 scope 对应的 HITL queue / geometry queue
9. 将新结果与 `base_result` 做局部合并

### 20.4 targeted merge 的重点

为保证 targeted rerun 既不漏更新，也不污染无关结果，代码里做了几类 merge：

- `_filter_rows_outside_scope(...)`
- `_merge_cycle_update(...)`
- `_merge_caption_payload(...)`
- `_merge_policy_report(...)`

具体含义是：

- 旧 votes / probe_results / human_queue 中，落在 scope 里的部分先删掉
- 再拼上新重算出来的部分
- `cycle_update` 里被影响的 claim 状态会被替换
- caption 只替换受影响 claim 对应的 feedback 部分
- policy 的 `vote_count` / `weighted_vote_count` 会按合并后的 votes 重算

### 20.5 空 scope 的 fallback

如果 targeted rerun 拿到的目标 scope 为空，当前实现会退回：

- `run_cycle_refine(...)`

也就是做一次全量 cycle，以避免“什么都没算但 UI 以为算过了”。

---

## 21. UI 里的 geometry 人工修复链

这是当前最值得单独说清楚的一段。

### 21.1 用户看到什么

当 relation claim 被判定为空间冲突时，`geometry_review_queue` 会生成一个 `bbox` 类型的 queue item。

UI 会把它渲染成：

- 当前 box
- 若干候选 box
- `Suggested` 候选
- `Apply Selected Box` / `Keep Current Box`

### 21.2 用户点确认以后发生什么

函数：

- `ui/video_task_studio.py::_apply_cycle_arbitration_change_to_graph`

处理顺序是：

1. 把选中的 bbox 写回 target node
2. 如有必要，清空 mask
3. 移除 `cycle_bbox_conflict` / `human_bbox_rejected` 之类 flag
4. 调用 `_rebuild_scene_graph_after_geometry_change(...)`
5. 用新的 bbox 重新计算 spatial edges
6. 把这次人工操作写进 `human_arbitration_history`

### 21.3 为什么相关 relation 会一起更新

函数：

- `ui/video_task_studio.py::_graph_delta_target_scope`

这里现在不只标记“发生变化的 node”，还会把与这些 node 相连的 edge 一起加入 `changed_edges`。

之后会派生出：

- `claim_exists_<node>`
- `claim_label_<node>`
- `claim_attr_<node>_<slot>`
- `claim_rel_<edge>`

所以 bbox 改动后，相关 spatial relation claim 会一起进入 targeted rerun 的 scope。

这条链是当前“修正一个 spatial 问题后，与其关联的问题也会被更新”的根本原因。

---

## 22. UI 如何回写 cycle 结果

结果返回 UI 后由：

- `ui/video_task_studio.py::_apply_cycle_result`

统一处理。

它会做这些事：

1. 把 `graph_after` 写回 `current_graph`
2. 把 cycle payload 存到 `graph_after.metadata.cycle_verification`
3. 更新 `current_cycle_result`
4. 刷新 probe outputs
5. 刷新 claim tables
6. 刷新 human arbitration view
7. 记录 queue 到 validation changes
8. 刷新 summary / caption / memory 面板
9. 重新渲染 graph
10. 持久化 scene graph bundle

因此，cycle 不是“只在内存里跑完给一段文本”，而是会把结构化结果沉淀回 graph metadata 和 UI 的多个面板。

---

## 23. 当前实现的几个关键保证

### 23.1 不会因为单个 probe 失败而整轮崩掉

probe 失败会变成 `invalid_response`，整轮继续。

### 23.2 selection 类纠错不是自由生成

它必须从 canonical options 中选，这降低了 API 模型自由发挥带来的不稳定性。

### 23.3 caption 默认不主导 label

caption 作为全局证据源，默认对 label 权重为 0。

### 23.4 bbox 修正不会只停留在 UI

确认 geometry 候选后：

- graph 会真的更新
- spatial edges 会重建
- targeted rerun 会重算相关 claims

### 23.5 targeted rerun 不会盲目覆盖整轮结果

当前 merge 是按 scope 局部替换，而不是把无关结果一起洗掉。

---

## 24. 当前实现的几个边界与注意事项

### 24.1 focus 可能影响 targeted rerun 的可见范围

如果 focus 配置过窄，目标 claim 可能在 `focused_graph` 中不可见。

### 24.2 relation 纠错与 geometry 修复是两条并行机制

- relation correction 更偏语义替换
- geometry review 更偏 bbox 几何修复

两者都可能修 relation 问题，但路径不同。

### 24.3 caption 的 fallback 路径较弱

当结构化 caption 失败时，会退回文本匹配投票，这个分支的可靠性明显低于结构化 JSON 分支。

### 24.4 “被影响 scope”是结构相关，而不是语义全传播

当前系统会更新：

- 直接相关的 node
- 直接相关的 edge
- 直接证据行

但不会无限向更远的语义依赖传播。

这意味着它解决的是“局部结构联动更新”，不是“整图知识闭包式联动”。

---

## 25. 一句话总结当前 cycle

当前的 IMPACT-SG cycle 本质上是一个“以 scene graph 为底座、以 claim 为中间层、以多视角 verifier 投票为证据、以自动修图和人工仲裁为出口”的闭环验证系统。

它已经不是简单的 VQA 调 API，而是一套完整的：

- claim selection
- probe generation
- claim-aware prompting
- structured response parsing
- vote weighting
- graph revision
- HITL escalation
- targeted rerun merging

联合工作流。

如果你后续要继续增强它，最值得继续演进的点通常会是：

- 更细粒度的 scope propagation
- 更强的 caption structured parsing 稳定性
- 更明确的 targeted rerun 可解释日志
- relation 语义纠错与 geometry 修复的协同策略

