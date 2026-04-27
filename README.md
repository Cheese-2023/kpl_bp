# KPL BP 推荐 Agent

这个项目把三张 KPL 数据表训练成一个轻量 BP 推荐 Agent。第一版目标是：给定当前 BP 局面，推荐下一手 ban/pick 英雄，并返回候选英雄的策略分、胜率评估分和综合分。

## 数据文件

- `kpl_bp.csv`：逐步 BP 数据，字段为 `match_id,battle_id,order,camp,type,hero`。
- `kpl_players.csv`：单局选手、英雄与胜负数据，字段为 `match_id,battle_id,team,player,hero,kill,death,assist,kda,gold,win`。
- `KPL_hero_2023_2026.csv`：赛季级英雄统计，字段为 `league_id,hero_name,battle_count,win_rate,avg_kda,ban_rate,pick_rate`。

## 训练

```bash
python3 -m kplbp.train
```

训练会生成：

- `models/kpl_bp_agent.json`：策略模型、胜率模型、BP 顺序表和英雄池。
- `reports/audit.json`：数据审计结果。
- `reports/metrics.json`：训练样本数和测试指标。

默认只使用完整 20 步 BP 局训练。如果希望把 7/8 步短局也作为早期局面样本，可以加：

```bash
python3 -m kplbp.train --include-incomplete
```

## 推理示例

```bash
python3 -m kplbp.recommend \
  --order 4 \
  --camp 1 \
  --action-type pick \
  --banned 狄仁杰,鲁班大师,狂铁,盾山 \
  --top-k 5
```

也可以直接传 JSON：

```bash
python3 -m kplbp.recommend --state-json '{"order":4,"camp":1,"action_type":"pick","banned":["狄仁杰","鲁班大师","狂铁","盾山"],"camp1_picks":[],"camp2_picks":[]}'
```

## 当前模型设计

`PolicyModel` 是监督学习基线：它从历史 BP 样本中学习当前轮次、阵营、动作类型、己方已选、敌方已选对应的英雄选择分布，并用英雄赛季 pick/ban/win 率做平滑。

`ValueModel` 是阵容胜率评估器：它从完整对局中学习英雄在己方或敌方阵容中出现时的经验胜率，并结合英雄表中的赛季胜率作为先验。

`BPAgent` 组合两者：先由策略模型给出候选，再用胜率模型和浅层 minimax 搜索重排，输出 TopK 推荐。
