from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PICK_ORDER_FROM_KPL = (1, 2, 2, 1, 1, 2, 2, 1, 1, 2)


@dataclass(frozen=True)
class MatchMode:
    mode: str
    label: str
    max_games: int
    global_bp: bool
    peak_duel_game: int | None = None


MATCH_MODES: dict[str, MatchMode] = {
    "single": MatchMode("single", "单局普通 BP", 1, False),
    "bo3_global": MatchMode("bo3_global", "BO3 全局 BP", 3, True),
    "bo5_global": MatchMode("bo5_global", "BO5 全局 BP", 5, True),
    "bo7_global": MatchMode("bo7_global", "BO7 全局 BP", 7, True),
    "bo5_peak": MatchMode("bo5_peak", "BO5 全局 BP + 最后一局巅峰对决", 5, True, 5),
    "bo7_peak": MatchMode("bo7_peak", "BO7 全局 BP + 第七局巅峰对决", 7, True, 7),
}


def mode_from_name(mode: str | None) -> MatchMode:
    return MATCH_MODES.get(mode or "single", MATCH_MODES["single"])


def peak_duel_schedule() -> list[dict[str, int | str]]:
    return [
        {"order": index, "camp": camp, "action_type": "pick"}
        for index, camp in enumerate(PICK_ORDER_FROM_KPL)
    ]


def schedule_for_mode(
    base_schedule: list[dict[str, int | str]],
    mode: str | None,
    game_index: int,
) -> list[dict[str, int | str]]:
    match_mode = mode_from_name(mode)
    if match_mode.peak_duel_game is not None and game_index == match_mode.peak_duel_game:
        return peak_duel_schedule()
    return base_schedule


def legal_heroes_for_state(
    all_heroes: Iterable[str],
    state_order: int,
    state_camp: int,
    action_type: str,
    mode: str | None,
    game_index: int,
    current_banned: Iterable[str],
    current_camp1_picks: Iterable[str],
    current_camp2_picks: Iterable[str],
    camp1_global_used: Iterable[str],
    camp2_global_used: Iterable[str],
) -> list[str]:
    match_mode = mode_from_name(mode)
    legal = set(all_heroes)
    banned = set(current_banned)
    camp1_picks = set(current_camp1_picks)
    camp2_picks = set(current_camp2_picks)
    used_current = banned | camp1_picks | camp2_picks

    legal -= used_current
    if match_mode.global_bp and game_index > 1:
        if action_type == "pick":
            legal -= set(camp1_global_used if state_camp == 1 else camp2_global_used)
        elif action_type == "ban":
            # Ban recommendations should focus on heroes the opponent can still pick.
            opponent_used = set(camp2_global_used if state_camp == 1 else camp1_global_used)
            legal -= opponent_used

    return sorted(legal)


def modes_payload() -> list[dict[str, object]]:
    return [
        {
            "mode": item.mode,
            "label": item.label,
            "max_games": item.max_games,
            "global_bp": item.global_bp,
            "peak_duel_game": item.peak_duel_game,
        }
        for item in MATCH_MODES.values()
    ]


def rerank_for_global_bp(
    recommendations: list[dict[str, object]],
    mode: str | None,
    game_index: int,
    action_type: str,
    hero_priorities: dict[str, float],
) -> list[dict[str, object]]:
    match_mode = mode_from_name(mode)
    if not match_mode.global_bp or action_type != "pick" or match_mode.max_games <= 1:
        return recommendations
    games_left_after_current = max(0, match_mode.max_games - game_index)
    if games_left_after_current <= 0:
        return recommendations

    # In global BP, very high priority heroes are a limited series resource.
    # Early games should still allow taking them, but not blindly exhaust them
    # when several lower-cost alternatives are close in model score.
    if mode == "bo3_global":
        reserve_factor = min(0.10, 0.045 * games_left_after_current)
    elif mode in {"bo5_global", "bo5_peak"}:
        reserve_factor = min(0.24, 0.065 * games_left_after_current)
    else:
        reserve_factor = min(0.34, 0.075 * games_left_after_current)
    adjusted: list[dict[str, object]] = []
    for item in recommendations:
        hero = str(item["hero"])
        priority = hero_priorities.get(hero, 0.0)
        penalty = reserve_factor * priority
        new_item = dict(item)
        new_item["raw_score"] = item.get("score", 0.0)
        new_item["reserve_penalty"] = round(penalty, 6)
        new_item["score"] = round(float(item.get("score", 0.0)) - penalty, 6)
        if penalty > 0.025:
            new_item["strategy_note"] = "全局 BP 早期局高优先级英雄，除非阵容刚需，否则可考虑保留。"
        new_item["mode_policy"] = mode or "single"
        adjusted.append(new_item)
    return sorted(adjusted, key=lambda value: float(value["score"]), reverse=True)


def strategy_profile_for_mode(mode: str | None, game_index: int) -> dict[str, object]:
    match_mode = mode_from_name(mode)
    if not match_mode.global_bp:
        return {
            "name": "single_game_best",
            "description": "单局最优策略，优先当前局面最强推荐。",
            "reserve_strength": 0.0,
        }
    games_left = max(0, match_mode.max_games - game_index)
    if match_mode.peak_duel_game is not None and game_index == match_mode.peak_duel_game:
        return {
            "name": "peak_duel_stability",
            "description": "巅峰对决策略，优先阵容完整、容错和熟练英雄。",
            "reserve_strength": 0.0,
        }
    return {
        "name": "global_bp_resource_management",
        "description": "全局 BP 策略，会在早期局保留部分高优先级英雄资源。",
        "reserve_strength": round(_reserve_strength(match_mode.mode, games_left), 4),
        "games_left_after_current": games_left,
    }


def _reserve_strength(mode: str, games_left_after_current: int) -> float:
    if mode == "bo3_global":
        return min(0.10, 0.045 * games_left_after_current)
    if mode in {"bo5_global", "bo5_peak"}:
        return min(0.24, 0.065 * games_left_after_current)
    return min(0.34, 0.075 * games_left_after_current)
