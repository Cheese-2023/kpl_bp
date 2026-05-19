from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .schema import HeroMeta


LANES = ("对抗路", "打野", "中路", "发育路", "游走")


@dataclass(frozen=True)
class LineupReport:
    heroes: list[str]
    missing_lanes: list[str]
    lane_counts: dict[str, int]
    role_counts: dict[str, int]
    damage_counts: dict[str, int]
    tag_counts: dict[str, int]
    strengths: list[str]
    weaknesses: list[str]
    game_plan: list[str]
    unknown_heroes: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "heroes": self.heroes,
            "missing_lanes": self.missing_lanes,
            "lane_counts": self.lane_counts,
            "role_counts": self.role_counts,
            "damage_counts": self.damage_counts,
            "tag_counts": self.tag_counts,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "game_plan": self.game_plan,
            "unknown_heroes": self.unknown_heroes,
        }


def split_hero_names(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace("，", ",").split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def analyze_lineup(heroes: Iterable[str], hero_meta: dict[str, HeroMeta]) -> LineupReport:
    hero_list = split_hero_names(heroes)
    known = [hero_meta[hero] for hero in hero_list if hero in hero_meta]
    unknown = [hero for hero in hero_list if hero not in hero_meta]

    lane_counts = Counter(lane for meta in known for lane in meta.lanes)
    role_counts = Counter(meta.role for meta in known)
    damage_counts = Counter(meta.damage_type for meta in known)
    tag_counts = Counter(tag for meta in known for tag in meta.tags)
    missing_lanes = [lane for lane in LANES if lane_counts[lane] == 0]
    strengths: list[str] = []
    weaknesses: list[str] = []
    game_plan: list[str] = []

    if tag_counts["开团"] + tag_counts["强开"] >= 2:
        strengths.append("具备较强先手能力，可以主动逼团或围绕资源点开战。")
    if tag_counts["保护"] + tag_counts["反打"] >= 2:
        strengths.append("保护和反打能力较好，适合围绕核心输出打阵地战。")
    if tag_counts["消耗"] + tag_counts["poke"] + tag_counts["远程"] >= 2:
        strengths.append("远程消耗能力不错，可以先压血线再接资源团。")
    if tag_counts["单带"] + tag_counts["带线"] >= 1:
        strengths.append("有边路单带点，可以通过兵线牵制创造视野和资源空间。")
    if damage_counts["物理"] >= 3 and damage_counts["法术"] >= 1:
        strengths.append("物理输出充足，并具备一定法术补伤。")
    if damage_counts["法术"] >= 2:
        strengths.append("法术压力较高，能迫使对手提前补魔抗或分散防装。")

    if missing_lanes:
        weaknesses.append(f"阵容分路不完整，缺少：{'、'.join(missing_lanes)}。")
    if damage_counts["法术"] == 0:
        weaknesses.append("缺少法术伤害，容易被对手堆物抗针对。")
    if damage_counts["物理"] == 0:
        weaknesses.append("缺少物理伤害，推塔和持续输出压力可能不足。")
    if tag_counts["开团"] + tag_counts["强开"] == 0:
        weaknesses.append("缺少稳定开团点，逆风时较难主动找机会。")
    if tag_counts["坦度"] + tag_counts["前排"] + tag_counts["承伤"] == 0:
        weaknesses.append("前排和承伤不足，正面接团容错较低。")
    if tag_counts["后期"] >= 2 and tag_counts["前期"] == 0:
        weaknesses.append("阵容偏发育，前期如果被入侵或丢资源会比较被动。")
    if tag_counts["怕切"] >= 1 and tag_counts["保护"] == 0:
        weaknesses.append("核心输出存在被切风险，但保护标签不足。")

    if tag_counts["前期"] + tag_counts["节奏"] + tag_counts["入侵"] >= 2:
        game_plan.append("前期应主动抢线、控野区和争夺中立资源，把节奏转化为塔和经济差。")
    if tag_counts["后期"] >= 2:
        game_plan.append("前中期以稳线和换资源为主，优先保证核心输出发育到关键装备。")
    if tag_counts["消耗"] + tag_counts["poke"] + tag_counts["远程"] >= 2:
        game_plan.append("中期围绕龙坑和防御塔提前站位，用远程技能压低血线后再开团。")
    if tag_counts["单带"] + tag_counts["带线"] >= 1:
        game_plan.append("通过边路线牵制迫使对手分人处理，再利用人数差控资源或开团。")
    if tag_counts["保护"] + tag_counts["反打"] >= 2:
        game_plan.append("团战不要急于先手，优先保护核心输出，等对手进场后反打。")
    if not game_plan:
        game_plan.append("以分路完整和资源团为核心，先保证视野与兵线，再根据敌方失误开团。")

    if not strengths:
        strengths.append("阵容强点不够集中，需要结合选手英雄池和对手阵容进一步判断。")
    if not weaknesses:
        weaknesses.append("没有明显结构性短板，但仍需关注对手克制关系和版本强势英雄。")

    return LineupReport(
        heroes=hero_list,
        missing_lanes=missing_lanes,
        lane_counts=dict(lane_counts),
        role_counts=dict(role_counts),
        damage_counts=dict(damage_counts),
        tag_counts=dict(tag_counts),
        strengths=strengths,
        weaknesses=weaknesses,
        game_plan=game_plan,
        unknown_heroes=unknown,
    )


def compare_lineups(
    own_heroes: Iterable[str],
    enemy_heroes: Iterable[str],
    hero_meta: dict[str, HeroMeta],
) -> dict[str, object]:
    own = analyze_lineup(own_heroes, hero_meta)
    enemy = analyze_lineup(enemy_heroes, hero_meta)
    suggestions: list[str] = []
    own_tags = Counter(own.tag_counts)
    enemy_tags = Counter(enemy.tag_counts)

    if enemy_tags["突进"] + enemy_tags["切后排"] >= 2 and own_tags["保护"] == 0:
        suggestions.append("敌方切后排能力较强，建议补保护、反打或强控制英雄。")
    if enemy_tags["消耗"] + enemy_tags["poke"] >= 2 and own_tags["开团"] + own_tags["强开"] == 0:
        suggestions.append("敌方消耗能力强，己方需要补强开或加速开团手段，避免被持续压血。")
    if enemy_tags["坦度"] + enemy_tags["前排"] >= 2 and own.damage_counts.get("真实伤害", 0) == 0:
        suggestions.append("敌方前排较厚，可以考虑补真实伤害、持续输出或百分比打坦英雄。")
    if own_tags["后期"] >= 2 and enemy_tags["前期"] + enemy_tags["节奏"] >= 2:
        suggestions.append("己方偏后期而敌方前期节奏强，前期需要收缩防守并避免野区硬碰。")
    if not suggestions:
        suggestions.append("双方结构没有明显单点克制，建议结合当前 BP 轮次优先补齐分路和开团/保护短板。")

    return {
        "own": own.to_dict(),
        "enemy": enemy.to_dict(),
        "suggestions": suggestions,
    }


def answer_question(
    question: str,
    own_heroes: Iterable[str],
    enemy_heroes: Iterable[str],
    hero_meta: dict[str, HeroMeta],
) -> dict[str, object]:
    question = question.strip() or "请分析阵容缺陷和游戏思路"
    comparison = compare_lineups(own_heroes, enemy_heroes, hero_meta)
    own = comparison["own"]
    enemy = comparison["enemy"]

    sections: list[str] = []
    if any(keyword in question for keyword in ("缺陷", "短板", "问题", "弱点")):
        sections.append("阵容缺陷：\n" + "\n".join(f"- {item}" for item in own["weaknesses"]))
    if any(keyword in question for keyword in ("思路", "打法", "运营", "怎么玩", "游戏")):
        sections.append("游戏思路：\n" + "\n".join(f"- {item}" for item in own["game_plan"]))
    if any(keyword in question for keyword in ("克制", "对面", "敌方", "应对")):
        sections.append("对敌方阵容的应对：\n" + "\n".join(f"- {item}" for item in comparison["suggestions"]))
    if any(keyword in question for keyword in ("优势", "强点")):
        sections.append("阵容强点：\n" + "\n".join(f"- {item}" for item in own["strengths"]))

    if not sections:
        sections = [
            "阵容强点：\n" + "\n".join(f"- {item}" for item in own["strengths"]),
            "阵容缺陷：\n" + "\n".join(f"- {item}" for item in own["weaknesses"]),
            "游戏思路：\n" + "\n".join(f"- {item}" for item in own["game_plan"]),
            "对敌方阵容的应对：\n" + "\n".join(f"- {item}" for item in comparison["suggestions"]),
        ]

    answer = "\n\n".join(sections)
    return {
        "question": question,
        "answer": answer,
        "analysis": comparison,
        "enemy_summary": enemy,
    }
