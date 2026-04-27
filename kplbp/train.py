from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .agent import BPAgent
from .data import (
    all_heroes,
    audit_tables,
    build_bp_samples,
    build_lineup_samples,
    dump_json,
    group_bp_by_battle,
    latest_hero_stats_by_name,
    load_bp_steps,
    load_hero_stats,
    load_player_rows,
    split_battles_by_time,
)
from .models import PolicyModel, ValueModel, value_accuracy


def infer_schedule(bp_steps) -> list[dict[str, int | str]]:
    slots: dict[int, Counter[tuple[int, str]]] = defaultdict(Counter)
    for steps in group_bp_by_battle(bp_steps).values():
        if len(steps) != 20:
            continue
        for step in steps:
            slots[step.order][(step.camp, step.action_type)] += 1
    schedule = []
    for order in sorted(slots):
        (camp, action_type), _ = slots[order].most_common(1)[0]
        schedule.append({"order": order, "camp": camp, "action_type": action_type})
    return schedule


def train_agent(args: argparse.Namespace) -> dict[str, object]:
    bp_steps = load_bp_steps(args.bp)
    player_rows = load_player_rows(args.players)
    hero_stats = load_hero_stats(args.heroes)
    hero_stats_by_name = latest_hero_stats_by_name(hero_stats)
    hero_pool = all_heroes(bp_steps, player_rows, hero_stats)

    audit = audit_tables(bp_steps, player_rows, hero_stats)
    bp_samples = build_bp_samples(bp_steps, full_battles_only=not args.include_incomplete)
    lineup_samples = build_lineup_samples(bp_steps, player_rows, full_battles_only=not args.include_incomplete)
    train_battles, test_battles = split_battles_by_time(sample.state.battle_id for sample in bp_samples)

    train_bp_samples = [sample for sample in bp_samples if sample.state.battle_id in train_battles]
    test_bp_samples = [sample for sample in bp_samples if sample.state.battle_id in test_battles]
    train_lineup_samples = [sample for sample in lineup_samples if sample.battle_id in train_battles]
    test_lineup_samples = [sample for sample in lineup_samples if sample.battle_id in test_battles]

    policy_model = PolicyModel(hero_pool, hero_stats_by_name).fit(train_bp_samples)
    value_model = ValueModel(hero_pool, hero_stats_by_name).fit(train_lineup_samples)
    schedule = infer_schedule(bp_steps)
    agent = BPAgent(policy_model, value_model, schedule, hero_pool)

    policy_eval = policy_model.evaluate(test_bp_samples)
    value_eval = value_accuracy(value_model, test_lineup_samples)
    metrics = {
        "audit": audit,
        "samples": {
            "bp_total": len(bp_samples),
            "bp_train": len(train_bp_samples),
            "bp_test": len(test_bp_samples),
            "lineup_total": len(lineup_samples),
            "lineup_train": len(train_lineup_samples),
            "lineup_test": len(test_lineup_samples),
        },
        "policy": {
            "test_total": policy_eval.total,
            "top1": policy_eval.top1,
            "top3": policy_eval.top3,
            "top5": policy_eval.top5,
            "legal_rate": policy_eval.legal_rate,
        },
        "value": value_eval,
        "schedule_steps": len(schedule),
    }

    dump_json(agent.to_dict(), args.model_out)
    dump_json(metrics, args.metrics_out)
    dump_json(audit, args.audit_out)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a KPL BP recommendation agent.")
    parser.add_argument("--bp", default="kpl_bp.csv", help="BP step CSV path.")
    parser.add_argument("--players", default="kpl_players.csv", help="Player/game result CSV path.")
    parser.add_argument("--heroes", default="KPL_hero_2023_2026.csv", help="Hero statistics CSV path.")
    parser.add_argument("--model-out", default="models/kpl_bp_agent.json", help="Output model JSON.")
    parser.add_argument("--metrics-out", default="reports/metrics.json", help="Output metrics JSON.")
    parser.add_argument("--audit-out", default="reports/audit.json", help="Output data audit JSON.")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Use battles with fewer than 20 BP steps as early-state samples.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
    metrics = train_agent(args)
    print("Training complete")
    print(f"BP samples: {metrics['samples']['bp_total']}")
    print(f"Policy Top1/Top3/Top5: {metrics['policy']['top1']:.3f} / {metrics['policy']['top3']:.3f} / {metrics['policy']['top5']:.3f}")
    print(f"Value accuracy: {metrics['value']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
