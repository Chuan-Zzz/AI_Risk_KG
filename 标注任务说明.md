# AI 风险知识图谱标注任务说明

## 项目背景

本项目旨在从新闻报道中自动构建 AI 风险事件知识图谱（Knowledge Graph）。知识图谱基于 **AIRO Extended Ontology**（人工智能风险本体）进行建模，包含以下核心元素：

### 核心实体类型（Entity Types）

| 实体类型 | 说明 | 示例 |
|---------|------|------|
| **AISystem** | AI 系统/产品 | YouTube Kids app, ChatGPT, Autopilot |
| **AIModel** | AI 模型 | GPT-4, Stable Diffusion, Claude |
| **AITechnique** | AI 技术/方法 | 深度学习, 强化学习, 人脸识别 |
| **AICapability** | AI 能力 | 内容推荐, 自动驾驶, 语音识别 |
| **Stakeholder** | 利益相关者（组织/个人） | Google, OpenAI, 监管机构 |
| **AIDeveloper** | AI 开发者 | 谷歌, 微软 |
| **AIProvider** | AI 提供商 | OpenAI, Anthropic |
| **AIDeployer** | AI 部署方 | 特斯拉, 苹果 |
| **Regulator** | 监管机构 | FTC, 欧盟委员会 |
| **AffectedActor** | 受影响方 | 儿童, 用户, 企业 |
| **RiskSource** | 风险来源 | 不当内容审核, 算法偏见 |
| **Risk** | 风险 | 有害内容传播, 隐私泄露 |
| **Consequence** | 后果 | 心理伤害, 财产损失 |
| **Impact** | 影响 | 社会影响, 经济损失 |
| **RiskControl** | 风险控制措施 | 内容审核机制, 隐私保护政策 |
| **Regulation** | 法规 | GDPR, AI Act |
| **Standard** | 标准 | ISO/IEC 42001 |
| **NewsReport** | 新闻报道 | 本任务待标注的文档 |
| **AIRiskIncident** | AI 风险事件 | 事件节点 |

### 核心关系类型（Relation Types）

| 关系类型 | 说明 | 示例 |
|---------|------|------|
| **involvesAISystem** | 事件涉及 AI 系统 | 事件 → ChatGPT |
| **involvesStakeholder** | 事件涉及利益相关者 | 事件 → Google |
| **hasRisk** | 存在风险 | 事件 → 隐私泄露风险 |
| **causes** | 导致风险 | 风险来源 → 风险 |
| **leadsTo** | 导致后果 | 风险 → 心理伤害 |
| **impacts** | 影响 | 后果 → 经济损失 |
| **affects** | 影响（作用于受影响方） | 影响 → 受影响群体 |
| **governs** | 治理/管辖 | 法规 → AI系统 |
| **usesTechnique** | 使用技术 | AI系统 → 深度学习 |
| **roleHeldBy** | 角色持有者 | 角色 → 组织 |

---

## 评测任务

本项目包含 **3 个评测任务**，请在 Label Studio 中完成标注。

---

## 任务 1：实体抽取评测（Entity Extraction）

### 任务描述

给定一篇新闻报道文本，请从中抽取所有与 AI 风险相关的实体，并为每个实体标注：
1. **实体名称**（原文中的表述）
2. **实体类型**（从上面的实体类型列表中选择）
3. **实体的证据句**（标注实体在文本中出现的那句话）

### 标注指南

#### 应该抽取的实体

1. **AI 系统/产品**：具体命名的 AI 应用或产品
   - ✅ "YouTube Kids app" → AISystem
   - ✅ "ChatGPT" → AISystem
   - ✅ "Autopilot" → AISystem

2. **AI 模型**：具体的 AI 模型名称
   - ✅ "GPT-4" → AIModel
   - ✅ "DALL-E" → AIModel

3. **AI 技术**：AI 相关技术方法
   - ✅ "深度学习" → AITechnique
   - ✅ "人脸识别技术" → AITechnique

4. **利益相关者**：公司、组织、机构、个人
   - ✅ "Google" → Stakeholder
   - ✅ "FTC" → Regulator
   - ✅ "OpenAI" → AIDeveloper

5. **风险相关实体**：风险、后果、影响等
   - ✅ "有害内容" → Risk
   - ✅ "心理伤害" → Consequence
   - ✅ "儿童用户" → AffectedActor

6. **法规/标准**
   - ✅ "GDPR" → Regulation
   - ✅ "AI Act" → Regulation

#### 不应该抽取的实体

- ❌ 泛泛而谈的通用词汇（如 "AI"、"人工智能" 而非具体系统）
- ❌ 纯描述性短语而非具体实体名称
- ❌ 与 AI 风险无关的实体

#### 标注示例

**原文**：
> "Apple's self-driving car project was involved in a minor accident in California. The California DMV reported that the Apple vehicle rear-ended another car during testing. Apple declined to comment."

**正确标注**：
| 实体名称 | 实体类型 | 证据句 |
|---------|---------|--------|
| Apple | AIDeveloper | "Apple's self-driving car project was involved..." |
| Apple vehicle | AISystem | "the Apple vehicle rear-ended another car..." |
| California DMV | Regulator | "The California DMV reported that..." |
| self-driving car | AISystem | "Apple's self-driving car project..." |

---

## 任务 2：关系抽取评测（Relation Extraction）

### 任务描述

给定一篇新闻报道及其已标注的实体，请标注实体之间的关系。

### 标注指南

#### 应该标注的关系

1. **事件核心关系**：
   - 事件 → 涉及 → AI系统（involvesAISystem）
   - 事件 → 涉及 → 利益相关者（involvesStakeholder）
   - 事件 → 存在 → 风险（hasRisk）

2. **风险传播链**：
   - 风险来源 → 导致 → 风险（causes）
   - 风险 → 导致 → 后果（leadsTo）
   - 后果 → 影响 → 影响（impacts）
   - 影响 → 作用于 → 受影响方（affects）

3. **治理关系**：
   - 法规 → 管辖 → AI系统/组织（governs）
   - 监管机构 → 调查 → 事件（investigates）

4. **技术关系**：
   - AI系统 → 使用技术 → AI技术（usesTechnique）
   - AI系统 → 拥有能力 → AI能力（hasCapability）

#### 关系完整性要求

风险传播链应尽量完整，示例：
```
风险来源 (causes) → 风险 (leadsTo) → 后果 (impacts) → 影响 (affects) → 受影响方
```

#### 标注示例

**实体**：
- 事件：AIRiskIncident
- AI系统：YouTube Kids app
- 组织：Google
- 风险：有害内容
- 后果：心理伤害
- 受影响方：儿童

**关系**：
| 主语 | 关系 | 宾语 | 证据句 |
|-----|------|-----|--------|
| 事件 | involvesAISystem | YouTube Kids app | "Google's new YouTube Kids app..." |
| 事件 | involvesStakeholder | Google | "Google launched the app..." |
| 事件 | hasRisk | 有害内容 | "app contains inappropriate content..." |
| 有害内容 | leadsTo | 心理伤害 | "explicit sexual language...harm to children" |
| 心理伤害 | affects | 儿童 | "harm to children" |

---

## 任务 3：事件知识图谱评测（Event KG Quality）

### 任务描述

评估模型构建的完整事件知识图谱的质量，包括：
1. 实体抽取的完整性和准确性
2. 关系抽取的完整性和准确性
3. 知识图谱整体结构的合理性

### 评估维度

| 维度 | 说明 | 评分标准 |
|-----|------|---------|
| **实体完整性** | 关键实体是否都被抽取 | 1-5 分 |
| **实体准确性** | 实体类型标注是否正确 | 1-5 分 |
| **关系完整性** | 关键关系是否都被抽取 | 1-5 分 |
| **关系准确性** | 关系类型和方向是否正确 | 1-5 分 |
| **风险链完整性** | 风险传播链是否完整 | 1-5 分 |
| **整体质量** | 综合评价 | 1-5 分 |

### 标注指南

#### 评分标准

| 分数 | 含义 |
|-----|------|
| 5 | 优秀 - 完全符合预期，几乎无需修改 |
| 4 | 良好 - 基本符合，有小瑕疵 |
| 3 | 中等 - 基本正确，但有明显缺失 |
| 2 | 较差 - 缺失或错误较多 |
| 1 | 很差 - 几乎全部错误 |

#### 需要注意的问题

1. **实体问题**：
   - 实体名称是否准确对应原文
   - 实体类型选择是否正确
   - 是否遗漏重要实体

2. **关系问题**：
   - 关系类型是否正确
   - 关系方向是否正确（如 "A 导致 B" 而非 "B 导致 A"）
   - 是否遗漏重要关系

3. **风险链问题**：
   - 风险传播链是否完整（应有因果关系）
   - 因果关系是否合理

#### 标注示例

**场景**：YouTube Kids app 被投诉含有不当内容

| 维度 | 评分 | 理由 |
|-----|------|------|
| 实体完整性 | 4 | 主要实体已抽取，但缺少"消费者倡导团体" |
| 实体准确性 | 5 | 实体类型标注准确 |
| 关系完整性 | 4 | 主要关系已抽取 |
| 关系准确性 | 4 | 关系类型正确，但缺少"监管机构调查"关系 |
| 风险链完整性 | 5 | 风险链完整：不当内容→心理伤害→儿童 |
| 整体质量 | 4 | 整体良好，有小改进空间 |

---

## Label Studio 操作指南

### 1. 登录与项目选择

1. 访问 Label Studio 服务器
2. 使用分配的账号登录
3. 在项目列表中选择对应的评测任务

### 2. 标注界面说明

- **左侧**：文档原文显示区
- **右侧**：标注区域
- **顶部**：工具栏（保存、跳转、快捷键说明）

### 3. 标注操作

#### 实体标注（任务 1）
1. 选中原文中的实体文本
2. 在弹出菜单中选择实体类型
3. 系统自动记录证据句

#### 关系标注（任务 2）
1. 点击主语实体
2. 拖拽到宾语实体
3. 在弹出菜单中选择关系类型

#### 质量评分（任务 3）
1. 查看模型生成的知识图谱
2. 对照原文进行评估
3. 在评分区域给出各维度评分
4. 在备注区说明主要问题

### 4. 快捷键

| 快捷键 | 功能 |
|-------|------|
| `Ctrl + S` | 保存标注 |
| `Ctrl + Enter` | 提交并跳转到下一条 |
| `Esc` | 取消当前标注操作 |

### 5. 注意事项

- 🔴 **重要**：请确保标注与原文保持一致，不要添加原文未提及的信息
- 🔴 **重要**：关系方向要正确（谁作用于谁）
- 🔴 **重要**：风险链要有因果逻辑
- 🟡 **建议**：遇到不确定的情况，参考原文上下文判断
- 🟡 **建议**：可以在备注区说明特殊情况

---

## 常见问题解答

### Q1: 一个实体有多个类型应该怎么标注？
**A**: 选择最具体、最准确的那个类型。例如 "Google" 可以是 Stakeholder，但如果上下文明确是"开发者"，可以选择 AIDeveloper。

### Q2: 如何判断关系是否存在？
**A**: 必须有明确的证据支持。关系应该是原文明确表达的，而非推测得出。

### Q3: 风险传播链不完整怎么办？
**A**: 只标注有明确证据的部分，不要推测。如果整个链条都有证据，应该完整标注。

### Q4: 实体名称太长怎么办？
**A**: 截取最能代表该实体的名称即可，但需确保不引起歧义。

### Q5: 遇到标注冲突怎么办？
**A**: 如果发现模型标注有明显错误，请按正确理解标注，并在备注区说明。

---

## 联系方式

如有疑问，请联系项目负责人。

---

**标注质量非常重要，请认真对待每一份标注！**
