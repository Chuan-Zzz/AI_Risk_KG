# 小论文补实验方案

> 适用范围：补齐 [小论文.pdf](../paper/小论文.pdf) 中 `6.7 Ablation Study` 与 `7.1 Baseline Comparison` 的占位数据，并形成可直接执行的实验口径。

## 1. 目标与最终产出

本方案只解决两件事：

1. 补齐论文 Table 8（消融实验）
2. 补齐论文 Table 9（基线对比）

最终需要产出以下内容：

- `Table 8: Ablation Study`
- `Table 9: Baseline Comparison`
- 每个设置对应的原始预测输出目录
- 每个设置对应的评测 JSON
- 一份可回溯的运行日志与配置记录

建议统一放在：

```text
output/experiments/
├── full_ontorisk/
├── ablation_no_ontology/
├── ablation_no_event_aggregation/
├── ablation_no_evidence_constraint/
├── ablation_no_controlled_inference/
├── baseline_gpt4o_llm_only/
├── baseline_deepseek_v3_llm_only/
├── baseline_rag_kg/
└── baseline_document_level_kg/

eval/results/
├── table8_ablation.json
├── table9_baseline.json
└── experiment_manifest.json
```

---

## 2. 固定实验设置

### 2.1 评测数据

所有 Table 8 / Table 9 的主结果统一使用：

- 事件集合：[data/gold_standard_100_events.json](/Users/chuan/Projects/AI_Risk_KG/data/gold_standard_100_events.json)
- 原始案例文本：优先 `data/eval_cases.jsonl`
- 若 `eval_cases.jsonl` 不存在，则退化为 `data/translated_docs.json`

不要混用 `500 event` 集合和 `100 gold` 集合。论文中的 Table 8 / Table 9 明确写的是 `100-event gold-standard evaluation set`。

### 2.2 固定评测口径

所有方法必须在**同一批 100 个事件**上评测，且统一输出到同一种中间格式，再进入同一套评测脚本。  
推荐统一中间格式为当前项目已有的 `event_subgraph.json` 风格：

- `nodes`
- `edges`
- `knowledge_statements`
- `support_statistics`

这样可以避免“不同方法用不同评分逻辑”导致表格不可比。

### 2.3 固定模型与随机性控制

推荐口径：

- 正式表格：每个设置完整跑 `1` 次全量 `100 event`
- 稳定性检查：每个设置额外抽 `20 event` 重跑 `2` 次，只做方差检查，不写入主表

原因：

- 你当前是 LLM 驱动系统，完全做 `3 x 100 x 9` 成本较高
- 小论文阶段更需要“主表完整可复现”，而不是大规模显著性检验

如果最终投稿更正式版本，再补：

- 每个设置全量跑 `3` 次
- 主表汇报 `mean ± std`

### 2.4 Full OntoRisk 基准行

Table 8 / Table 9 中 `OntoRisk (Ours)` 或 `Full OntoRisk` 一行，优先沿用项目现有已完成结果：

- `F1 (L1) = 94.5`
- `Acc. (L2) = 95.1`
- `Acc. (L3) = 98.1`
- `ESR = 98.7`

这些数与 [eval/results/eval_results_arbitrated.json](../../eval/results/eval_results_arbitrated.json) 已基本对齐。  
如果你重新全量跑 100 个 gold event，则以新结果覆盖。

---

## 3. 统一评测指标定义

论文里的 Table 8 / Table 9 统一使用下面 7 个指标：

- `F1 (L1)`：显式实体抽取 F1
- `Acc. (L2)`：风险链准确率
- `Acc. (L3)`：推理字段准确率
- `ESR`：Evidence Support Rate
- `CDC`：Cross-Document Consistency
- `RCC`：Risk-Chain Completeness
- `USR`：Unsupported Statement Rate

### 3.1 F1 (L1)

直接沿用现有仲裁后脚本口径：

- 参考脚本：[eval/scripts/compute_metrics_arbitrated.py](../../eval/scripts/compute_metrics_arbitrated.py)
- 以显式实体标注结果为准
- 部分匹配记 `0.5`

### 3.2 Acc. (L2)

直接沿用现有风险链评分口径：

- `Correct = 1.0`
- `Partial = 0.5`
- `Incorrect = 0.0`

最终：

```text
Acc(L2) = (Correct + 0.5 * Partial) / Total
```

### 3.3 Acc. (L3)

直接沿用现有推理字段评分口径：

- `Purpose`
- `AILifecyclePhase`
- `Domain`

最终：

```text
Acc(L3) = (Correct + 0.5 * Partial) / Total
```

### 3.4 ESR

定义为：

```text
ESR = #supported_knowledge_statements / #all_knowledge_statements
```

`supported_knowledge_statement` 判定标准：

- `knowledge_statement.evidence.evidence_sentence` 非空
- `knowledge_statement.evidence.source_doc_id` 非空
- `source_doc_id != "event_level"`

### 3.5 USR

定义为：

```text
USR = #unsupported_knowledge_statements / #all_knowledge_statements
```

按论文写法可直接报告：

```text
USR = 1 - ESR
```

### 3.6 RCC

按当前项目图结构，定义一个事件的风险链为“完整”，需同时满足：

- 存在 `RiskSource`
- 存在 `Risk`
- 存在 `Consequence`
- 存在 `Impact`
- 存在 `AffectedActor`
- 同时存在链式关系：
  - `RiskSource -causes-> Risk`
  - `Risk -leadsTo-> Consequence`
  - `Consequence -impacts-> Impact`
  - `Impact -affects-> AffectedActor`

计算方式：

```text
RCC = #complete_event_chains / #events_with_any_risk_chain_output
```

### 3.7 CDC

这是当前最容易写进论文、但最容易执行时口径飘的指标，所以必须先固定。

推荐操作化定义：

对每个事件的全部 `knowledge_statements`：

1. 按 `(normalized_subject, predicate)` 分组
2. 若同组内出现多个**语义不相同**的 `object`
3. 且这些 object 不是同义词或上下位词映射
4. 则该组记为 `inconsistent`
5. 其余记为 `consistent`

最终：

```text
CDC = #consistent_statement_groups / #all_statement_groups
```

为避免纯人工判定过重，建议第一版采用下面规则：

- 精确归一化相同：视为一致
- 本体同类型、名称仅大小写/符号不同：视为一致
- 同一 `(subject, predicate)` 对应多个不同 object：视为冲突
- `Stakeholder` 子类之间不算 CDC 冲突，因角色多重性是允许的

这一定义虽然是“工程化近似”，但和你论文“跨文档一致性”想表达的东西是一致的，而且能稳定复现。

---

## 4. Table 8：消融实验详细方案

论文里 Table 8 一共需要 5 行：

- Full OntoRisk
- w/o Ontology Constraint
- w/o Event Aggregation
- w/o Evidence Constraint
- w/o Controlled Inference

### 4.1 Full OntoRisk

这是完整系统，不做改动，直接跑当前标准 pipeline。

对应代码主线：

- 入口：[main.py](/Users/chuan/Projects/AI_Risk_KG/main.py:44)
- 图流程：[src/agent/graph.py](/Users/chuan/Projects/AI_Risk_KG/src/agent/graph.py:1)

输出目录建议：

```text
output/experiments/full_ontorisk/<event_id>/event_subgraph.json
```

---

### 4.2 w/o Ontology Constraint

#### 目标

验证“本体约束”对实体类型一致性、关系合法性和推理可靠性的贡献。

#### 核心改动

移除三类本体约束：

1. Stage 2 不再强制限制抽取类型集合
2. Stage 4 L3 不再限制推理候选空间
3. Validation 不再进行 ontology domain/range 检查

#### 最小实现建议

新增一个开关，例如：

```yaml
experiment:
  disable_ontology_constraint: true
```

对应修改：

- [src/agent/nodes/explicit_extract.py](/Users/chuan/Projects/AI_Risk_KG/src/agent/nodes/explicit_extract.py:1)
  - 放宽 `_STAGE2_TYPES`
  - 不因 ontology normalize 失败而直接丢弃
- [src/agent/prompts/stage4_inference.py](/Users/chuan/Projects/AI_Risk_KG/src/agent/prompts/stage4_inference.py:3)
  - 去掉固定候选约束
- [src/agent/nodes/validation.py](/Users/chuan/Projects/AI_Risk_KG/src/agent/nodes/validation.py:1)
  - 跳过 relation domain/range 校验

#### 评测注意

为了能和 gold 标注对齐，最终评测前仍要做一次**最弱归一化映射**：

- 只允许 alias 级别映射
- 不允许额外 LLM 再校正

否则会把“去掉本体约束”的损失偷偷补回来。

#### 预期现象

- `F1 (L1)` 下滑明显
- `CDC` 下降
- `USR` 小幅上升
- `Acc. (L3)` 也会下降，因为自由推理更容易跑偏

---

### 4.3 w/o Event Aggregation

#### 目标

验证事件级聚合是否真的是性能提升的关键来源。

#### 论文口径

论文原文要求的是：

> each document is processed independently and results are merged by simple union without EntityNorm, SupportCount, or ConflictDetect

所以这个消融不能只是“跳过 representative report 选择”，而要更接近“无事件聚合”的真实版本。

#### 推荐实现

对同一事件中的每篇文档分别独立运行：

- explicit extraction
- entity reuse（仅文档内）
- risk chain
- inference
- graph build

然后在事件级做**简单并集**：

- 节点按 `(name_lower, type)` 精确去重
- 边按 `(subject_name, predicate, object_name)` 精确去重
- 不做：
  - EntityNorm
  - SupportCount
  - ConflictDetect
  - representative report selection
  - embedding/BM25/LLM 对齐

#### 为什么不能直接复用当前 aggregation 节点

因为当前 [src/agent/nodes/aggregation.py](/Users/chuan/Projects/AI_Risk_KG/src/agent/nodes/aggregation.py:1) 本身就包含了：

- entity merge
- support statistics
- conflict detection
- representative report selection

这些恰好都是论文里要拿掉的部分。

#### 输出目录

```text
output/experiments/ablation_no_event_aggregation/<event_id>/event_subgraph.json
```

#### 预期现象

- `CDC` 降幅最大
- `RCC` 明显下降
- `F1 (L1)` 中度下降
- `ESR` 可能只小幅下降，因为单文档仍能提供局部证据

---

### 4.4 w/o Evidence Constraint

#### 目标

验证证据约束是否确实降低幻觉、提高可审计性。

#### 核心改动

保留整体 pipeline，但取消“必须给 evidence / reasoning”的硬约束。

#### 最小实现建议

新增开关：

```yaml
experiment:
  disable_evidence_constraint: true
```

对应改动：

- Stage 2 / Stage 4 prompt 中，去掉 “必须提供 evidence/source_doc_id” 的强制措辞
- [src/agent/nodes/validation.py](/Users/chuan/Projects/AI_Risk_KG/src/agent/nodes/validation.py:1)
  - 跳过 evidence 非空校验
  - 跳过 inferred reasoning 非空校验

#### 关键点

这个版本不是“故意删掉 evidence”，而是“模型可以不给 evidence，也照样通过”。  
这样更符合论文里 “evidence spans are not required during extraction” 的表述。

#### 预期现象

- `USR` 大幅上升
- `ESR` 大幅下降
- `Acc. (L3)` 会有下降
- `F1 (L1)` 可能只轻微变化，甚至略升，因为模型少了证据绑定约束

---

### 4.5 w/o Controlled Inference

#### 目标

验证第 4 阶段第 3 层“受控推理”是否真的减少幻觉并提升隐式字段质量。

#### 核心改动

只改 Stage 4 L3：

- 不使用 ontology-defined candidate restriction
- 允许模型自由生成 purpose / lifecycle / domain / governance fields

#### 最小实现建议

新增一个自由推理 prompt，例如：

- 输入：同样的 evidence package
- 输出：同样的 JSON 字段
- 但不再要求从固定候选值中选择

这样保证：

- 输入相同
- 输出格式相同
- 唯一差别是“有无候选约束”

#### 评测注意

这一行主要影响：

- `Acc. (L3)`
- `ESR`
- `CDC`
- `USR`

`F1 (L1)` 与 `Acc. (L2)` 理论上几乎不变。  
因此论文里如果你不想硬写 L1/L2，也可以保持为 `–`，和当前 PDF 口径一致。

---

### 4.6 Table 8 填表顺序

建议按这个顺序跑：

1. Full OntoRisk
2. w/o Event Aggregation
3. w/o Evidence Constraint
4. w/o Controlled Inference
5. w/o Ontology Constraint

原因：

- 前三项最容易直接从当前架构分叉
- `w/o Ontology Constraint` 需要改动最深，最后做更稳

推荐 Table 8 结果文件结构：

```json
{
  "full_ontorisk": {
    "f1_l1": 94.5,
    "acc_l2": 95.1,
    "acc_l3": 98.1,
    "esr": 98.7,
    "cdc": 96.3,
    "rcc": 94.8,
    "usr": 1.3
  },
  "ablation_no_ontology": {},
  "ablation_no_event_aggregation": {},
  "ablation_no_evidence_constraint": {},
  "ablation_no_controlled_inference": {}
}
```

---

## 5. Table 9：基线对比详细方案

论文里 Table 9 一共需要 5 行：

- GPT-4o (LLM-only)
- DeepSeek-V3 (LLM-only)
- RAG-based KG
- Document-level KG
- OntoRisk (Ours)

### 5.1 统一原则

四个 baseline 必须满足两条原则：

1. 输出最终都要转成同一评测中间格式
2. baseline 不能偷偷使用 OntoRisk 的关键能力

具体来说：

- LLM-only baseline 不能用 ontology schema 限制
- Document-level KG 不能用 event aggregation
- RAG baseline 不能用 cross-document fusion / support count / conflict detect

---

### 5.2 GPT-4o (LLM-only)

#### 定义

对每篇文档单独提问，让 GPT-4o 直接输出：

- entities
- relations
- risk chain
- inference fields

不使用：

- ontology constraints
- event aggregation
- evidence grounding hard constraint

#### 推荐 prompt 结构

每篇文档只用一个总 prompt，要求模型返回统一 JSON：

- `entities`
- `relations`
- `risk_chain`
- `inferences`

#### 事件级合并方式

同一事件内多篇文档输出后，只做简单 union：

- 节点：按 `(normalized_name, raw_type)` 精确去重
- 边：按 `(subject, predicate, object)` 精确去重
- statement：每条边附带原 doc_id

不要做 embedding 对齐，不要做 LLM 复核，不要做 support_count 聚合。

#### 结果用途

这是最“强模型、弱结构”的 baseline，用来证明：

- 单纯靠强 LLM 不足以支撑高 ESR / 低 USR / 高 CDC

---

### 5.3 DeepSeek-V3 (LLM-only)

定义与 GPT-4o 相同，只替换模型。

#### 作用

这条不是为了赢，而是为了说明论文结论**不依赖单一模型品牌**。  
如果 GPT-4o 和 DeepSeek-V3 都显著弱于 OntoRisk，就能支持“框架贡献大于 backbone 差异”。

#### 重要提醒

如果你当前并没有真实可用的 GPT-4o / DeepSeek-V3 API：

- 不要先写名字再填模拟数据
- 必须真实跑过再在论文里写模型名

如果暂时只能跑一个外部模型，论文里要如实写成：

- `LLM-only baseline (GPT-4o)` 或
- `LLM-only baseline (available commercial model)`

---

### 5.4 RAG-based KG

#### 定义

将同一事件的全部文档切 chunk，先检索，再让 LLM 基于检索结果抽取结构化知识。

#### 推荐实现

1. 对该事件全部文档按段落切分
2. 每段建立向量索引
3. 针对三类任务分别检索：
   - Query A：显式实体
   - Query B：风险链
   - Query C：推理字段
4. 每类 query 取 top-k 片段
5. 将片段拼接给 LLM，输出结构化 JSON

#### 推荐参数

- `chunk_size = 300~500 tokens`
- `overlap = 50 tokens`
- `top_k = 5`
- 每事件最多使用 `15` 个片段上下文

#### 与 OntoRisk 的边界

RAG baseline 可以看到多文档内容，但它不是事件聚合：

- 没有显式 EntityNorm
- 没有 SupportCount
- 没有 ConflictDetect
- 没有 event-centric merge object

#### 预期表现

- 比纯 LLM-only 强
- 但显著弱于 OntoRisk
- 尤其在 `CDC`、`RCC` 上会明显落后

---

### 5.5 Document-level KG

#### 定义

这是最重要的 baseline。  
它和 OntoRisk 共用相同 ontology schema、相似的抽取结构，但**不做跨文档事件聚合**。

#### 推荐实现

对每篇文档单独运行一个“文档级简化版 pipeline”：

- Stage 2 explicit extraction
- Stage 4 risk chain
- Stage 4 inference
- Stage 5 graph build

然后在事件级只做简单 union：

- 节点按 `(name_lower, type)` 精确去重
- 边按 `(subject_name, predicate, object_name)` 精确去重

不允许使用：

- aggregation node
- representative report selection
- support statistics merge
- conflict detection
- semantic aligner / llm verifier 跨文档对齐

#### 它和 w/o Event Aggregation 的区别

两者很接近，但论文里用途不同：

- `w/o Event Aggregation` 属于消融，强调“去掉一个模块”
- `Document-level KG` 属于 baseline，强调“现有 pipeline 范式”

实际执行时，为了节省成本，这两行可以共用同一套预测输出；论文写作时分别放在 Table 8 和 Table 9。

---

### 5.6 Table 9 填表顺序

推荐顺序：

1. Document-level KG
2. GPT-4o (LLM-only)
3. DeepSeek-V3 (LLM-only)
4. RAG-based KG
5. Ours

原因：

- Document-level KG 最接近现有项目，最好先跑出来作为“结构化弱化 baseline”
- 两个 LLM-only baseline 最花 API 成本，但逻辑简单
- RAG baseline 需要单独做 chunk + retrieval 适配，最后做更省返工

推荐结果文件：

```json
{
  "gpt4o_llm_only": {},
  "deepseek_v3_llm_only": {},
  "rag_based_kg": {},
  "document_level_kg": {},
  "ontorisk_ours": {
    "f1_l1": 94.5,
    "acc_l2": 95.1,
    "acc_l3": 98.1,
    "esr": 98.7,
    "cdc": 96.3,
    "rcc": 94.8,
    "usr": 1.3
  }
}
```

---

## 6. 统一执行流程

### 6.1 每个设置都走同一流水

对任一 ablation 或 baseline，都按以下顺序执行：

1. 读取 `gold_standard_100_events.json`
2. 生成该设置的预测结果
3. 输出到独立目录
4. 转换成统一 `event_subgraph.json` 格式
5. 跑统一评测脚本
6. 写入 `table8_ablation.json` 或 `table9_baseline.json`

### 6.2 推荐新增脚本

建议新增以下脚本，而不是把逻辑塞进现有入口：

```text
scripts/
├── run_ablation_gold.py
├── run_baseline_gold.py
├── convert_baseline_to_subgraph.py
└── collect_table_metrics.py
```

功能建议：

- `run_ablation_gold.py --variant full|no_ontology|no_agg|no_evidence|no_inference`
- `run_baseline_gold.py --baseline gpt4o|deepseek|rag|doclevel`
- `convert_baseline_to_subgraph.py`
  - 把 baseline 原始输出适配成统一 JSON
- `collect_table_metrics.py`
  - 汇总 F1/Acc/ESR/CDC/RCC/USR

### 6.3 manifest 记录

每次正式跑表，必须保存：

- 模型名
- base_url
- 代码 commit id
- 配置文件快照
- 运行日期
- 事件列表
- 失败事件列表

建议写入：

```json
{
  "experiment_name": "table8_ablation_no_event_aggregation",
  "date": "2026-07-04",
  "model": "glm-5.1",
  "dataset": "gold_standard_100_events",
  "code_commit": "xxxx",
  "config_snapshot": {},
  "success_events": 97,
  "failed_events": 3
}
```

---

## 7. 时间与工作量预估

按“先可发论文、后补最优实现”的思路，推荐分三轮做。

### 第一轮：先补表格主干

目标：

- 跑出 `Full OntoRisk`
- 跑出 `Document-level KG`
- 跑出 `w/o Evidence Constraint`

意义：

- 这三行最容易先形成论文核心对比
- 能最快填掉大部分占位符

### 第二轮：补关键对照

目标：

- 跑出 `w/o Event Aggregation`
- 跑出 `w/o Controlled Inference`
- 跑出 `RAG-based KG`

意义：

- 补齐论文“结构贡献”和“RAG 对照”

### 第三轮：补完整性

目标：

- 跑出 `w/o Ontology Constraint`
- 跑出 GPT-4o / DeepSeek-V3 两条 LLM-only baseline

意义：

- 把论文从“能交稿”提升到“结构完整”

---

## 8. 论文写作注意事项

### 8.1 不要把“同一输出”写成两个不同实验

如果 `Document-level KG` 与 `w/o Event Aggregation` 最终共用了同一套预测结果：

- 可以
- 但论文中要分别解释其角色

一个是：

- `Table 8`: component ablation

一个是：

- `Table 9`: external baseline paradigm

### 8.2 不要先写模型名后补跑

特别是：

- GPT-4o
- DeepSeek-V3

如果没有真实 API 跑过，不要在正文写死。

### 8.3 占位数据优先补这三列

如果时间有限，优先先补：

- `CDC`
- `RCC`
- `USR`

因为你现有项目里：

- `F1 (L1)`
- `Acc. (L2)`
- `Acc. (L3)`
- `ESR`

已经有较明确基础结果了。  
真正让 Table 8 / Table 9 更像论文“方法贡献证明”的，是后面这三列。

---

## 9. 建议的最终表格模板

### Table 8

```text
Table 8: Ablation study results on the 100-event gold-standard evaluation set.

| Setting                     | F1 (L1) | Acc. (L2) | Acc. (L3) | ESR  | CDC  | RCC  | USR |
|----------------------------|---------|-----------|-----------|------|------|------|-----|
| Full OntoRisk              | 94.5    | 95.1      | 98.1      | 98.7 | xx.x | xx.x | x.x |
| w/o Ontology Constraint    | xx.x    | xx.x      | xx.x      | xx.x | xx.x | xx.x | x.x |
| w/o Event Aggregation      | xx.x    | xx.x      | xx.x      | xx.x | xx.x | xx.x | x.x |
| w/o Evidence Constraint    | xx.x    | xx.x      | xx.x      | xx.x | xx.x | xx.x | x.x |
| w/o Controlled Inference   | -       | -         | xx.x      | xx.x | xx.x | -    | x.x |
```

### Table 9

```text
Table 9: Baseline comparison on the 100-event gold-standard evaluation set.

| Method                   | F1 (L1) | Acc. (L2) | Acc. (L3) | ESR  | CDC  | RCC  | USR |
|-------------------------|---------|-----------|-----------|------|------|------|-----|
| GPT-4o (LLM-only)       | xx.x    | xx.x      | xx.x      | xx.x | xx.x | xx.x | xx.x |
| DeepSeek-V3 (LLM-only)  | xx.x    | xx.x      | xx.x      | xx.x | xx.x | xx.x | xx.x |
| RAG-based KG            | xx.x    | xx.x      | xx.x      | xx.x | xx.x | xx.x | xx.x |
| Document-level KG       | xx.x    | xx.x      | xx.x      | xx.x | xx.x | xx.x | xx.x |
| OntoRisk (Ours)         | 94.5    | 95.1      | 98.1      | 98.7 | xx.x | xx.x | x.x |
```

---

## 10. 最推荐的执行顺序

如果你现在就开始补实验，我建议直接按下面顺序做：

1. 把 `Full OntoRisk` 的 `CDC/RCC/USR` 先补出来
2. 跑 `Document-level KG`
3. 跑 `w/o Evidence Constraint`
4. 跑 `w/o Event Aggregation`
5. 跑 `w/o Controlled Inference`
6. 跑 `RAG-based KG`
7. 最后补 `w/o Ontology Constraint` 与两个 LLM-only baseline

这样做的好处是：

- 先把最容易出结果的行补齐
- 先让论文表格从“占位稿”变成“有实数的实验稿”
- 再逐步补难度更高、改动更大的设置
