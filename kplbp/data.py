from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .schema import BPSample, BPState, BPStep, HeroStats, LineupSample, PlayerRow


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_bp_steps(path: str | Path) -> list[BPStep]:
    rows = read_csv_dicts(path)
    return [
        BPStep(
            match_id=row["match_id"],
            battle_id=row["battle_id"],
            order=int(row["order"]),
            camp=int(row["camp"]),
            action_type=row["type"].strip().lower(),
            hero=row["hero"].strip(),
        )
        for row in rows
    ]


def load_player_rows(path: str | Path) -> list[PlayerRow]:
    rows = read_csv_dicts(path)
    return [
        PlayerRow(
            match_id=row["match_id"],
            battle_id=row["battle_id"],
            team=row["team"].strip(),
            player=row["player"].strip(),
            hero=row["hero"].strip(),
            kill=int(row["kill"]),
            death=int(row["death"]),
            assist=int(row["assist"]),
            kda=float(row["kda"]),
            gold=int(row["gold"]),
            win=int(row["win"]),
        )
        for row in rows
    ]


def load_hero_stats(path: str | Path) -> list[HeroStats]:
    rows = read_csv_dicts(path)
    return [
        HeroStats(
            league_id=row["league_id"],
            hero_name=row["hero_name"].strip(),
            battle_count=int(row["battle_count"]),
            win_rate=float(row["win_rate"]),
            avg_kda=float(row["avg_kda"]),
            ban_rate=float(row["ban_rate"]),
            pick_rate=float(row["pick_rate"]),
        )
        for row in rows
    ]


def group_bp_by_battle(steps: Iterable[BPStep]) -> dict[str, list[BPStep]]:
    battles: dict[str, list[BPStep]] = defaultdict(list)
    for step in steps:
        battles[step.battle_id].append(step)
    return {battle_id: sorted(items, key=lambda item: item.order) for battle_id, items in battles.items()}


def group_players_by_battle(rows: Iterable[PlayerRow]) -> dict[str, list[PlayerRow]]:
    battles: dict[str, list[PlayerRow]] = defaultdict(list)
    for row in rows:
        battles[row.battle_id].append(row)
    return dict(battles)


def infer_camp_metadata(
    bp_steps: list[BPStep],
    player_rows: list[PlayerRow],
) -> dict[tuple[str, int], dict[str, object]]:
    bp_by_battle = group_bp_by_battle(bp_steps)
    players_by_battle = group_players_by_battle(player_rows)
    metadata: dict[tuple[str, int], dict[str, object]] = {}
    for battle_id, steps in bp_by_battle.items():
        player_rows_for_battle = players_by_battle.get(battle_id, [])
        hero_to_player = {row.hero: row for row in player_rows_for_battle}
        for camp in (1, 2):
            pick_steps = [
                step for step in steps if step.action_type == "pick" and step.camp == camp
            ]
            rows = [
                hero_to_player[step.hero]
                for step in pick_steps
                if step.hero in hero_to_player
            ]
            teams = Counter(row.team for row in rows)
            team = teams.most_common(1)[0][0] if teams else None
            metadata[(battle_id, camp)] = {
                "team": team,
                "players": tuple(row.player for row in rows),
                "wins": tuple(row.win for row in rows),
            }
    return metadata


def all_heroes(
    steps: Iterable[BPStep],
    player_rows: Iterable[PlayerRow],
    hero_stats: Iterable[HeroStats],
) -> list[str]:
    heroes = {step.hero for step in steps}
    heroes.update(row.hero for row in player_rows)
    heroes.update(stat.hero_name for stat in hero_stats)
    return sorted(hero for hero in heroes if hero)


def latest_hero_stats_by_name(stats: Iterable[HeroStats]) -> dict[str, HeroStats]:
    latest: dict[str, HeroStats] = {}
    for stat in stats:
        old = latest.get(stat.hero_name)
        if old is None or stat.league_id > old.league_id:
            latest[stat.hero_name] = stat
    return latest


def audit_tables(
    bp_steps: list[BPStep],
    player_rows: list[PlayerRow],
    hero_stats: list[HeroStats],
) -> dict[str, object]:
    bp_by_battle = group_bp_by_battle(bp_steps)
    players_by_battle = group_players_by_battle(player_rows)
    camp_metadata = infer_camp_metadata(bp_steps, player_rows)
    bp_lengths = Counter(len(rows) for rows in bp_by_battle.values())
    player_lengths = Counter(len(rows) for rows in players_by_battle.values())
    action_types = Counter(step.action_type for step in bp_steps)
    camps = Counter(step.camp for step in bp_steps)
    duplicate_step_battles = 0
    non_monotonic_battles = 0
    for rows in bp_by_battle.values():
        orders = [row.order for row in rows]
        if len(set(orders)) != len(orders):
            duplicate_step_battles += 1
        if orders != sorted(orders):
            non_monotonic_battles += 1

    return {
        "rows": {
            "bp": len(bp_steps),
            "players": len(player_rows),
            "hero_stats": len(hero_stats),
        },
        "battles": {
            "bp": len(bp_by_battle),
            "players": len(players_by_battle),
            "bp_not_in_players": len(set(bp_by_battle) - set(players_by_battle)),
            "players_not_in_bp": len(set(players_by_battle) - set(bp_by_battle)),
        },
        "bp_lengths": dict(sorted(bp_lengths.items())),
        "player_rows_per_battle": dict(sorted(player_lengths.items())),
        "action_types": dict(action_types),
        "camps": dict(camps),
        "duplicate_order_battles": duplicate_step_battles,
        "non_monotonic_battles": non_monotonic_battles,
        "inferred_camp_teams": sum(1 for meta in camp_metadata.values() if meta["team"]),
        "heroes": len(all_heroes(bp_steps, player_rows, hero_stats)),
    }


def build_bp_samples(
    bp_steps: list[BPStep],
    player_rows: list[PlayerRow] | None = None,
    full_battles_only: bool = True,
) -> list[BPSample]:
    samples: list[BPSample] = []
    camp_metadata = infer_camp_metadata(bp_steps, player_rows or [])
    for battle_id, steps in group_bp_by_battle(bp_steps).items():
        if full_battles_only and len(steps) != 20:
            continue
        camp1_team = camp_metadata.get((battle_id, 1), {}).get("team")
        camp2_team = camp_metadata.get((battle_id, 2), {}).get("team")
        banned: list[str] = []
        camp1_picks: list[str] = []
        camp2_picks: list[str] = []
        for step in steps:
            state = BPState(
                match_id=step.match_id,
                battle_id=battle_id,
                order=step.order,
                camp=step.camp,
                action_type=step.action_type,
                banned=list(banned),
                camp1_picks=list(camp1_picks),
                camp2_picks=list(camp2_picks),
                camp1_team=str(camp1_team) if camp1_team else None,
                camp2_team=str(camp2_team) if camp2_team else None,
            )
            samples.append(BPSample(state=state, label=step.hero))
            if step.action_type == "ban":
                banned.append(step.hero)
            elif step.camp == 1:
                camp1_picks.append(step.hero)
            else:
                camp2_picks.append(step.hero)
    return samples


def infer_camp_wins(
    bp_steps: list[BPStep],
    player_rows: list[PlayerRow],
) -> dict[tuple[str, int], int]:
    bp_by_battle = group_bp_by_battle(bp_steps)
    camp_metadata = infer_camp_metadata(bp_steps, player_rows)
    result: dict[tuple[str, int], int] = {}
    for battle_id in bp_by_battle:
        for camp in (1, 2):
            wins = list(camp_metadata.get((battle_id, camp), {}).get("wins", ()))
            if wins:
                result[(battle_id, camp)] = 1 if sum(wins) >= len(wins) / 2 else 0
    return result


def build_lineup_samples(
    bp_steps: list[BPStep],
    player_rows: list[PlayerRow],
    full_battles_only: bool = True,
) -> list[LineupSample]:
    camp_wins = infer_camp_wins(bp_steps, player_rows)
    camp_metadata = infer_camp_metadata(bp_steps, player_rows)
    samples: list[LineupSample] = []
    for battle_id, steps in group_bp_by_battle(bp_steps).items():
        if full_battles_only and len(steps) != 20:
            continue
        picks_by_camp: dict[int, list[str]] = {1: [], 2: []}
        for step in steps:
            if step.action_type == "pick":
                picks_by_camp[step.camp].append(step.hero)
        if len(picks_by_camp[1]) != 5 or len(picks_by_camp[2]) != 5:
            continue
        for camp in (1, 2):
            win = camp_wins.get((battle_id, camp))
            if win is None:
                continue
            enemy_camp = 2 if camp == 1 else 1
            own_meta = camp_metadata.get((battle_id, camp), {})
            enemy_meta = camp_metadata.get((battle_id, enemy_camp), {})
            samples.append(
                LineupSample(
                    battle_id=battle_id,
                    camp=camp,
                    own_picks=tuple(picks_by_camp[camp]),
                    enemy_picks=tuple(picks_by_camp[enemy_camp]),
                    win=win,
                    team=own_meta.get("team"),
                    enemy_team=enemy_meta.get("team"),
                    players=tuple(own_meta.get("players", ())),
                    enemy_players=tuple(enemy_meta.get("players", ())),
                )
            )
    return samples


def split_battles_by_time(
    battle_ids: Iterable[str],
    train_ratio: float = 0.8,
) -> tuple[set[str], set[str]]:
    ordered = sorted(set(battle_ids))
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * train_ratio)))
    return set(ordered[:cut]), set(ordered[cut:])


def dump_json(data: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: str | Path) -> object:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
