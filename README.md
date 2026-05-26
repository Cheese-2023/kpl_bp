# KPL BP 推荐 Agent

这是一个基于 KPL 职业比赛数据的 BP 推荐 Agent。当前版本的目标不是直接预测整局比赛胜负，而是在给定当前 BP 局面的情况下，推荐下一手应该 ban 或 pick 的英雄，并返回候选英雄的综合排序。

**本次更新核心能力：Agentic Workflow —— 大模型重排与辅助分析**

在本地 BP Agent（策略模型 + 阵容价值 + 浅层搜索）给出 Top-K 候选后，可接入 OpenAI 兼容云端大模型，形成两层工作流：

1. **队伍画像**：自定义蓝/红方选手姓名、分路、打法风格、状态、擅长/回避英雄，以及队伍整体风格与胜利条件。画像会注入云端 AI 与重排序 prompt，影响最终建议。
2. **AI 重排序**（`POST /api/bp/llm-rerank`）：对本地模型 top-10 候选做 LLM 重排，返回逐英雄理由、Top3 综合分析、下一手核心判断。
3. **云端 AI 复盘**（`POST /api/cloud-ai`）：结合 BP 状态、本地推荐、标签分析、队伍画像和 `docs/BP_EXPERIENCE.md` 做结构化 BP 教练式分析。

简化流程：

```text
BP 局面 -> 本地 Agent Top-10 ->（可选）队伍画像 -> LLM 重排序 / 云端复盘 -> 前端展示
```

项目目前使用三类信息：

- 历史 BP 顺序：学习职业队在不同 BP 阶段的真实选择。
- 单局选手与胜负数据：推断双方队伍、选手、阵容和最终胜负。
- 英雄赛季统计：提供英雄 pick 率、ban 率、胜率、KDA 等先验。

当前实现更适合做“当前版本内的 BP 辅助推荐”。由于版本变化会明显影响英雄强度，跨版本训练时需要谨慎。

## 项目目标

Agent 输入一个 BP 状态：

- 当前 BP 顺序 `order`
- 当前阵营 `camp`
- 当前动作类型 `ban` 或 `pick`
- 已经 ban 的英雄
- 蓝红双方已经 pick 的英雄
- 可选的双方队伍名称

Agent 输出 TopK 推荐：

- `hero`：推荐英雄
- `policy_score`：历史职业 BP 策略分
- `value_score`：阵容价值评估分
- `score`：综合分

## 项目结构

```text
.
├── KPL_hero_2023_2026.csv
├── kpl_bp.csv
├── kpl_players.csv
├── hero_meta.csv
├── docs/
│   └── BP_EXPERIENCE.md
├── kplbp/
│   ├── agent.py
│   ├── analysis.py
│   ├── api.py
│   ├── data.py
│   ├── models.py
│   ├── recommend.py
│   ├── schema.py
│   └── train.py
├── web/
│   └── index.html
├── models/
│   └── kpl_bp_agent.json
├── reports/
│   ├── audit.json
│   └── metrics.json
└── README.md
```

核心代码说明：

- `kplbp/schema.py`：定义 BP 步骤、选手数据、英雄统计、BP 状态、训练样本，以及 `PlayerProfile` / `TeamProfile` 队伍画像结构。
- `kplbp/data.py`：读取 CSV、审计数据、展开 BP 样本、构建阵容胜率样本。
- `kplbp/analysis.py`：基于英雄元数据做阵容结构、缺陷、强点和游戏思路分析。
- `kplbp/api.py`：提供前端页面、本地推荐 API、云端 AI 与 LLM 重排序 API。
- `kplbp/models.py`：实现策略模型 `PolicyModel` 和阵容价值模型 `ValueModel`。
- `kplbp/agent.py`：组合策略模型、价值模型和浅层搜索，输出最终推荐。
- `kplbp/train.py`：训练入口。
- `kplbp/recommend.py`：命令行推理入口。

## 数据文件

### `kpl_bp.csv`

逐步 BP 数据，一行代表一次 ban 或 pick。

字段：

- `match_id`：比赛 ID
- `battle_id`：单局 ID
- `order`：BP 顺序
- `camp`：阵营，`1` 和 `2`
- `type`：动作类型，`ban` 或 `pick`
- `hero`：英雄名称

当前数据中大多数完整局是 20 步 BP。训练默认只使用完整 20 步 BP 局，避免短局影响完整 BP 流程学习。

### `kpl_players.csv`

单局选手、英雄、表现和胜负数据。

字段：

- `match_id`：比赛 ID
- `battle_id`：单局 ID
- `team`：战队
- `player`：选手
- `hero`：使用英雄
- `kill`、`death`、`assist`、`kda`、`gold`：局内表现
- `win`：是否获胜，`1` 表示胜利，`0` 表示失败

这个表非常关键。模型会通过 BP 表中的 pick 英雄匹配 `players` 表，从而反推出每一局 `camp=1` 和 `camp=2` 分别是哪支队伍、哪些选手。

### `KPL_hero_2023_2026.csv`

英雄赛季级统计数据。

字段：

- `league_id`：赛事或版本标识
- `hero_name`：英雄名称
- `battle_count`：出场局数
- `win_rate`：胜率
- `avg_kda`：平均 KDA
- `ban_rate`：ban 率
- `pick_rate`：pick 率

这些数据会作为先验，帮助模型在样本量较少时避免过拟合。

### `hero_meta.csv`

英雄元数据表，用于前端查询、阵容结构分析和问答 API。

字段：

- `hero`：英雄名称，需与 BP 和选手表中的英雄名称一致。
- `lane`：主推荐分路，例如对抗路、打野、中路、发育路、游走。
- `role`：英雄职业或功能定位，例如射手、法师、战士、坦克、辅助、刺客、法刺。
- `damage_type`：主要伤害类型，例如物理、法术、真实伤害、混伤。
- `tags`：阵容标签，用英文分号 `;` 分隔，例如开团、保护、消耗、后期、单带、切后排。
- `alt_lanes`：可选副分路，用英文分号 `;` 分隔，例如 `打野;游走`。若没有可留空。

这张表会被 `/api/heroes`、`/api/analyze` 和 `/api/ask` 使用。后续如果想提升阵容分析质量，优先维护这张表。

英雄位置不是固定的。前端会按照 `lane + alt_lanes` 将英雄展示到多个分路中，阵容分析也会基于多分路判断阵容是否完整。

## 当前建模思路

当前 Agent 由三个部分组成：

```text
BPState -> PolicyModel -> 候选英雄
       -> ValueModel  -> 阵容价值评估
       -> BPAgent     -> 搜索和重排
```

### 1. 策略模型 `PolicyModel`

`PolicyModel` 的目标是模仿职业比赛中的 BP 决策。它不是直接判断哪个英雄最强，而是学习“在类似 BP 局面下，职业队通常会怎么选”。

当前使用的特征包括：

- 当前 BP 顺序
- 当前动作类型：ban 或 pick
- 当前阵营
- 己方已经 pick 的英雄
- 敌方已经 pick 的英雄
- 已经 ban 的英雄
- 己方队伍历史 pick 偏好
- 己方队伍历史 ban 偏好
- 针对敌方队伍时常 ban 的英雄
- 英雄赛季 pick 率、ban 率、胜率先验

这样做的好处是：在数据量有限的情况下，频率模型比复杂神经网络更稳，不容易因为几百局数据就严重过拟合。

### 2. 阵容价值模型 `ValueModel`

`ValueModel` 的目标是评估一个阵容本身是否更有价值。这里面有一个关键问题：比赛胜负不只由阵容决定，还会受到队伍实力、选手状态、临场发挥的影响。

例如：一支强队即使拿了理论上不占优的阵容，也可能靠实力赢下比赛。如果直接用胜负训练英雄价值，模型会误以为这套阵容很强。

为减少这个问题，当前版本做了队伍强度校正：

1. 先从 `kpl_players.csv` 估计队伍和选手的基础强度。
2. 对每局比赛计算一个“仅由队伍/选手实力带来的预期胜率”。
3. 用真实胜负减去这个预期值，得到残差信号。
4. 用残差信号学习英雄和阵容的价值。

因此，Agent 在 BP 搜索时默认使用“不包含队伍强度”的阵容价值分，而不是直接预测哪支队伍更可能赢。

### 3. 搜索与重排 `BPAgent`

`BPAgent` 会先让 `PolicyModel` 给出一批候选英雄，然后用 `ValueModel` 和浅层 minimax 搜索进行重排。

简化理解：

- 如果轮到我方行动，搜索会倾向选择让我方阵容价值更高的英雄。
- 如果轮到敌方行动，搜索会假设敌方会选择对我方更不利的英雄。
- 最终综合职业 BP 习惯和阵容价值，得到 TopK 推荐。

## 为什么暂时不用复杂强化学习

KPL 当前版本有效数据量有限，只有几百局完整比赛。直接上深度强化学习或复杂神经网络会遇到几个问题：

- 数据量不足，模型容易记住历史比赛而不是学到通用规律。
- BP 行为强烈依赖版本，跨版本数据会引入噪声。
- 阵容强弱和队伍强弱混在一起，直接训练容易学偏。
- 强化学习需要可靠的环境评估器，而当前胜率模型还不够强。

所以当前版本采用“监督策略模型 + 阵容价值模型 + 浅层搜索”的路线。这个路线更适合小数据场景，也更容易解释和调试。

## 训练

在项目根目录执行：

```bash
python3 -m kplbp.train
```

训练会生成：

- `models/kpl_bp_agent.json`：训练后的 Agent，包括策略模型、价值模型、英雄池和 BP 顺序表。
- `reports/audit.json`：数据审计结果。
- `reports/metrics.json`：训练样本数、测试指标和特征数量。

默认只使用完整 20 步 BP 局。如果希望把 7/8 步短局也作为早期局面样本，可以执行：

```bash
python3 -m kplbp.train --include-incomplete
```

## 推理

推荐下一手 pick：

```bash
python3 -m kplbp.recommend \
  --order 4 \
  --camp 1 \
  --action-type pick \
  --banned 狄仁杰,鲁班大师,狂铁,盾山 \
  --camp1-team 北京JDG \
  --camp2-team 武汉eStarPro \
  --top-k 5
```

推荐下一手 ban：

```bash
python3 -m kplbp.recommend \
  --order 10 \
  --camp 2 \
  --action-type ban \
  --banned 狄仁杰,鲁班大师,狂铁,盾山 \
  --camp1-picks 空空儿,戈娅,小乔 \
  --camp2-picks 张飞,公孙离,沈梦溪 \
  --camp1-team 北京JDG \
  --camp2-team 武汉eStarPro \
  --top-k 5
```

也可以直接传 JSON：

```bash
python3 -m kplbp.recommend --state-json '{"order":4,"camp":1,"action_type":"pick","banned":["狄仁杰","鲁班大师","狂铁","盾山"],"camp1_picks":[],"camp2_picks":[],"camp1_team":"北京JDG","camp2_team":"武汉eStarPro"}'
```

## 前端页面和 API

当前项目已经可以作为一个完整的轻量前后端项目运行：

- 后端：`python3 -m kplbp.api`
- 前端：`web/index.html`，由后端静态托管
- 本地模型：`models/kpl_bp_agent.json`
- 英雄元数据：`hero_meta.csv`
- 云端 AI：OpenAI 兼容接口，可选启用；复盘默认 `DeepSeek-R1`，重排序默认 `Qwen3.6-Max`

启动本地服务：

```bash
python3 -m kplbp.api
```

默认访问：

```text
http://127.0.0.1:8000
```

可以指定端口：

```bash
python3 -m kplbp.api --host 0.0.0.0 --port 8080
```

### 云端 AI 配置

不要把 API Key 写进代码、README 或提交到仓库。启动服务前在终端设置环境变量：

```bash
export MOARK_API_KEY="你的云端 API Key"
# 或
export CLOUD_AI_API_KEY="你的云端 API Key"
python3 -m kplbp.api
```

后端会读取 `CLOUD_AI_API_KEY` / `MOARK_API_KEY` 等环境变量，通过 `/api/cloud-ai` 和 `/api/bp/llm-rerank` 调用云端大模型。若没有设置环境变量，接口会返回本地分析、完整 `prompt`、`messages` 和 `request_body`，方便调试。

也可复制 `cloud_api_config.example.json` 为 `cloud_api_config.json`（不要提交真实 key）：

```json
{
  "base_url": "https://api.moark.com/v1",
  "model": "DeepSeek-R1",
  "rerank_model": "Qwen3.6-Max",
  "api_key": "建议使用环境变量"
}
```

默认配置等价于 OpenAI 兼容调用：

```bash
python3 -m kplbp.api \
  --cloud-ai-base-url https://api.moark.com/v1 \
  --cloud-ai-model DeepSeek-R1 \
  --cloud-ai-rerank-model Qwen3.6-Max
```

请求会自动携带 `X-Failover-Enabled: true` header（Moark 故障转移）。

### BP 模拟器

访问 `http://127.0.0.1:8000` 后，可以在页面完成以下操作：

1. 输入蓝方和红方队伍名称。
2. 在「队伍画像」面板填写选手风格、段位、擅长英雄等（会随 AI 请求一起发送）。
3. 选择赛制：单局、BO3 全局 BP、BO5 全局 BP、BO7 全局 BP、BO5/BO7 带最后一局巅峰对决。
4. 选择当前第几局，并在页面中点击选择双方全局已用英雄池。
5. **中间栏**按分路浏览并点击英雄完成 ban/pick；**左栏**为模拟设置，**右栏**为推荐与 AI 分析。
6. 每一步后端会调用本地 BP Agent 返回 top-10 推荐候选。
7. 点击「✨ AI 重排序」：调用大模型对 top-10 重排并给出逐英雄分析与核心判断（通常需等待 30–90 秒，页面会显示等待状态）。
8. 可手动调用「本地标签分析」「云端 AI 提问」或「一键实时分析当前 BP」。
9. 可以撤销一步或重置 BP。

页面采用白色主色 + 深蓝/金色辅色的轻量 UI；所有大模型调用在等待期间会显示 loading 提示。

### 赛制规则说明

当前规则层实现了几种常用模拟方式：

- `single`：单局普通 BP，使用模型从历史数据学习到的 20 步 BP 顺序。
- `bo3_global`：BO3 全局 BP，后续局本方不能重复 pick 自己前面局已经使用过的英雄。
- `bo5_global`：BO5 全局 BP。
- `bo7_global`：BO7 全局 BP。
- `bo5_peak`：BO5 全局 BP + 最后一局巅峰对决。
- `bo7_peak`：BO7 全局 BP + 第七局巅峰对决。

全局 BP 的过滤规则：

- pick 时过滤本方在前面局已经使用过的英雄。
- ban 推荐会优先过滤敌方已经无法再 pick 的英雄，避免推荐低价值 ban。
- 当前局已经 ban/pick 的英雄不会再次出现。
- 页面中“全局已用英雄”支持可视化多选，可直接点击英雄加入/移除，不需要手打。

巅峰对决当前实现为无 ban 的 10 手 pick 顺序，用于模拟最后一局快速阵容选择。由于历史训练数据主要来自普通 BP，巅峰对决推荐仍然复用本地策略模型和阵容价值模型，后续如果有专门的巅峰对决数据，可以单独训练该模式。

### API 列表

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

查询英雄元数据：

```bash
curl http://127.0.0.1:8000/api/heroes
```

获取 BP 模拟器配置，包括英雄池和 BP 顺序：

```bash
curl http://127.0.0.1:8000/api/bp/config
```

获取当前 BP 状态下的本地模型推荐：

```bash
curl -X POST http://127.0.0.1:8000/api/bp/recommend \
  -H 'Content-Type: application/json' \
  -d '{"mode":"bo7_peak","game_index":2,"actions":[{"hero":"狄仁杰"},{"hero":"鲁班大师"},{"hero":"狂铁"},{"hero":"盾山"}],"camp1_global_used":"公孙离,沈梦溪","camp2_global_used":"戈娅,小乔","camp1_team":"北京JDG","camp2_team":"武汉eStarPro","top_k":10}'
```

LLM 重排序（对本地 top-10 候选做 AI 重排与分析）：

```bash
curl -X POST http://127.0.0.1:8000/api/bp/llm-rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "mode":"single",
    "actions":[],
    "top_k":10,
    "recommendations":[
      {"hero":"公孙离","policy_score":0.08,"value_score":0.51,"score":0.19},
      {"hero":"后羿","policy_score":0.07,"value_score":0.50,"score":0.18}
    ],
    "camp1_profile":{
      "camp":1,"team_name":"北京JDG",
      "players":[{"name":"选手A","lane":"发育路","style":"激进开团","tier":"巅峰","preferred_heroes":["公孙离"]}],
      "overall_style":"早期压制"
    },
    "question":"请结合队伍画像重排序并分析"
  }'
```

返回字段：

- `reranked`：重排后的英雄列表（含 `rank`、`hero`、`reason`、`style_bonus`、`original_rank`）
- `top3_analysis`：前三名综合分析
- `key_decision`：下一手核心判断
- `original_recommendations`：本地模型原始推荐
- `error` / `prompt` / `raw_answer`：错误或调试信息

队伍画像 payload 字段（`camp1_profile` / `camp2_profile`）：

- `team_name`：队伍名称
- `players[]`：`name`、`lane`、`style`、`tier`、`preferred_heroes`、`avoid_heroes`
- `overall_style`：整体风格
- `win_condition`：胜利条件偏好

分析阵容：

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"heroes":["公孙离","沈梦溪","张飞"],"enemy_heroes":["戈娅","小乔","苏烈"]}'
```

基于英雄标签回答问题：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"heroes":["公孙离","沈梦溪","张飞"],"enemy_heroes":["戈娅","小乔","苏烈"],"question":"这个阵容有什么缺陷？游戏思路怎么打？"}'
```

当前问答 API 是规则型分析器，不依赖外部大模型。它会查询 `hero_meta.csv`，根据英雄分路、职业、伤害类型和标签回答阵容缺陷、强点、游戏思路和对敌方阵容的应对。

调用云端 AI 分析当前 BP：

```bash
curl -X POST http://127.0.0.1:8000/api/cloud-ai \
  -H 'Content-Type: application/json' \
  -d '{"mode":"bo7_peak","game_index":2,"actions":[{"hero":"狄仁杰"},{"hero":"鲁班大师"},{"hero":"狂铁"},{"hero":"盾山"}],"heroes":["公孙离","沈梦溪","张飞"],"enemy_heroes":["戈娅","小乔","苏烈"],"question":"请分析当前 BP 下一手怎么选，以及这套阵容的游戏思路。"}'
```

云端 AI 的提示词会包含：

- 用户问题
- 当前 BP 状态
- 本地 BP Agent 推荐
- 蓝/红方队伍画像（选手风格、擅长英雄等）
- 基于 `hero_meta.csv` 的阵容标签分析
- `docs/BP_EXPERIENCE.md` 中的 BP 经验知识库
- 要求模型分别输出蓝方优势/短板、红方优势/短板、下一手建议、全局 BP 资源判断、双方前中后期思路和风险点

接口返回中会包含完整上传内容：

- `prompt`：最终用户提示词
- `messages`：OpenAI 兼容 messages
- `request_body`：最终发送给云端模型的请求体

### BP 经验知识库

`docs/BP_EXPERIENCE.md` 用来给本地检索和云端 AI 分析提供背景知识，包含：

- BP 基本原则
- 阵容结构检查
- 常见体系，例如大乔体系、太乙保护体系、盾山反消耗体系
- 常见克制关系，例如强开克制消耗、保护克制突进、真实伤害克制厚前排
- 全局 BP 资源管理经验
- 巅峰对决经验

后续如果你想让云端回答更贴近你的理解，可以优先维护这份文档。

### 全局 BP 推荐策略

在 BO3/BO5/BO7 全局 BP 中，本地推荐会额外做“资源保留”重排：

- 根据英雄 pick 率、ban 率、胜率估计英雄资源优先级。
- 早期局如果还有较多后续局，会轻微降低高优先级英雄的推荐分。
- 目的不是禁止选择强势英雄，而是提醒模型：当多个候选接近时，不要过早把所有核心资源消耗完。
- 推荐结果中可能出现 `reserve_penalty` 和 `strategy_note`，表示该英雄受到了全局 BP 资源保留策略影响。

## 指标解释

训练完成后查看：

```bash
reports/metrics.json
```

主要字段：

- `policy.top1`：真实职业选择是否排在第 1。
- `policy.top3`：真实职业选择是否出现在前 3。
- `policy.top5`：真实职业选择是否出现在前 5。
- `policy.legal_rate`：推荐是否始终合法，即不会推荐已 ban 或已 pick 的英雄。
- `value`：只看阵容质量的胜率评估，适合 BP 搜索使用。
- `value_with_team_strength`：加入队伍强度后的真实赛果预测，适合赛果预测参考。
- `feature_counts`：当前训练出的队伍、选手、英雄组合、英雄对抗等特征数量。

当前版本中，`ValueModel` 会学习英雄组合和英雄对抗残差，并导出到模型文件中。但由于当前数据量较小，这类两两特征方差较大，直接加入阵容打分会让验证集下降，所以默认不参与最终打分。它们目前更适合用于分析，而不是作为主决策信号。

## 当前版本效果

当前训练结果大致为：

- `Top1`：约 17.5%
- `Top3`：约 34.8%
- `Top5`：约 44.2%
- 合法动作率：100%
- 只看阵容质量的胜率模型：约 47.0%
- 加入队伍强度后的赛果预测：约 59.0%

需要注意：BP 推荐的 TopK 命中率不是唯一目标。真实职业 BP 受到训练赛信息、队伍英雄池、对手准备、版本理解和临场策略影响，历史真实选择不一定是唯一正确选择。

## 当前局限

### 1. 数据量有限

当前完整 BP 局数量有限。对单英雄频率、队伍偏好这类特征还可以支撑，但对英雄组合、英雄克制、阵容体系这类高维特征仍然偏少。

### 2. 缺少英雄位置和职业标签

目前模型不知道英雄是打野、中路、发育路、对抗路还是辅助，也不知道阵容是否缺前排、缺控制、缺开团、缺后期输出。这会限制阵容价值模型的判断。

### 3. 版本因素仍然较粗

英雄强度强依赖版本。当前英雄表有 `league_id`，但还没有更细粒度的补丁版本、比赛日期、版本可用英雄池和版本强势体系信息。

### 4. 队伍英雄池只有弱建模

当前队伍偏好主要来自历史 pick/ban 频率。更理想的方式是建模“某队某选手是否会某英雄”、“某英雄是否适合该队体系”。

### 5. 胜率模型还比较弱

当前阵容胜率模型是轻量经验模型，适合做辅助重排，但还不足以作为完全可靠的阵容强度评估器。

## 后续优化思路

### 1. 补充英雄基础特征

建议新增一张英雄元数据表，例如 `hero_meta.csv`：

```csv
hero,role,lane,damage_type,tags
公孙离,射手,发育路,物理,位移;后期;持续输出
张飞,辅助,游走,物理,保护;开团;坦克
沈梦溪,法师,中路,法术,消耗;支援;清线
```

可用字段包括：

- 分路：对抗路、打野、中路、发育路、游走
- 职业：战士、刺客、法师、射手、辅助、坦克
- 伤害类型：物理、法术、真实伤害、混伤
- 阵容标签：开团、反开、保护、消耗、强开、带线、前期、后期、控制、坦度

补充后可以让模型判断阵容结构，例如：

- 是否五个位置完整
- 是否缺少前排
- 是否缺少法术伤害
- 是否控制不足
- 是否前期太弱
- 是否阵容过脆

### 2. 建模选手英雄池

基于 `kpl_players.csv` 可以统计：

- 选手使用某英雄次数
- 选手使用某英雄胜率
- 选手近期常用英雄
- 某队某位置常用英雄

这能解决一个很实际的问题：某英雄理论上很强，但某个队伍或某个选手未必会选。

### 3. 加入时间衰减

越新的比赛越接近当前版本理解，权重应更高。可以给样本加时间衰减：

```text
weight = exp(-days_from_latest / decay_days)
```

如果没有日期，可以先用 `match_id` 或 `league_id` 的顺序近似。

### 4. 升级策略模型

当前 `PolicyModel` 是频率和平滑模型。下一步可以尝试：

- LightGBM / XGBoost 多分类
- 逻辑回归 + 稀疏特征
- 小型 MLP
- Transformer / Set Transformer

在当前数据量下，推荐优先尝试 LightGBM 或逻辑回归，因为它们对小数据更稳，也更容易解释。

### 5. 升级胜率模型

当英雄元数据和选手英雄池补齐后，可以把 `ValueModel` 升级为：

- 逻辑回归：可解释、抗过拟合
- LightGBM：适合表格特征
- 双塔模型：分别编码蓝方阵容和红方阵容
- Bradley-Terry 风格模型：拆分队伍强度、选手强度、阵容强度

重点仍然是把“队伍强度”和“阵容强度”分开。

### 6. 更强的搜索

当前搜索是浅层 minimax。后续可以优化为：

- Beam Search：每一步保留多个高分局面
- MCTS：用策略模型扩展候选，用价值模型评估叶子节点
- 约束搜索：强制阵容位置完整、伤害结构合理

搜索质量取决于胜率模型质量，所以建议先增强阵容评估，再增强搜索。

### 7. 做人工复盘评估

除了看 TopK 命中率，还应该做人工复盘：

- 选择几场经典 BP
- 在关键节点让 Agent 给推荐
- 对比解说、赛后复盘和实际阵容
- 标记推荐是否合理，而不只看是否命中真实选择

这对 BP Agent 很重要，因为真实选择不一定是唯一正确答案。

## 推荐的下一步

最优先建议做三件事：

1. 新增 `hero_meta.csv`，补齐英雄分路、职业、伤害类型和阵容标签。
2. 从 `kpl_players.csv` 统计选手英雄池，把“会不会这个英雄”加入推荐。
3. 用 LightGBM 或逻辑回归替换当前频率策略模型，保留当前模型作为 baseline。

在这三步完成前，不建议直接做复杂强化学习。当前数据规模下，先把特征和评估做好，收益会更稳定。
