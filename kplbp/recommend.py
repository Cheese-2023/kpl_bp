from __future__ import annotations

import argparse
import json

from .agent import BPAgent
from .data import load_json
from .schema import BPState


def parse_hero_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_agent(path: str) -> BPAgent:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Model file must contain a JSON object.")
    return BPAgent.from_dict(data)


def build_state(args: argparse.Namespace) -> BPState:
    if args.state_json:
        raw = json.loads(args.state_json)
        return BPState(
            match_id=str(raw.get("match_id", "manual")),
            battle_id=str(raw.get("battle_id", "manual")),
            order=int(raw["order"]),
            camp=int(raw["camp"]),
            action_type=str(raw["action_type"]),
            banned=list(raw.get("banned", [])),
            camp1_picks=list(raw.get("camp1_picks", [])),
            camp2_picks=list(raw.get("camp2_picks", [])),
            camp1_team=raw.get("camp1_team"),
            camp2_team=raw.get("camp2_team"),
        )
    return BPState(
        match_id="manual",
        battle_id="manual",
        order=args.order,
        camp=args.camp,
        action_type=args.action_type,
        banned=parse_hero_list(args.banned),
        camp1_picks=parse_hero_list(args.camp1_picks),
        camp2_picks=parse_hero_list(args.camp2_picks),
        camp1_team=args.camp1_team,
        camp2_team=args.camp2_team,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recommend the next KPL BP action.")
    parser.add_argument("--model", default="models/kpl_bp_agent.json", help="Trained model JSON path.")
    parser.add_argument("--state-json", help="State JSON string. Overrides individual state args.")
    parser.add_argument("--order", type=int, default=0, help="Current BP order.")
    parser.add_argument("--camp", type=int, choices=[1, 2], default=1, help="Current camp.")
    parser.add_argument("--action-type", choices=["ban", "pick"], default="ban", help="Current action.")
    parser.add_argument("--banned", default="", help="Comma-separated banned heroes.")
    parser.add_argument("--camp1-picks", default="", help="Comma-separated camp 1 picked heroes.")
    parser.add_argument("--camp2-picks", default="", help="Comma-separated camp 2 picked heroes.")
    parser.add_argument("--camp1-team", help="Camp 1 team name, used for team preference features.")
    parser.add_argument("--camp2-team", help="Camp 2 team name, used for team preference features.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations.")
    parser.add_argument("--search-depth", type=int, default=2, help="Lookahead depth.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    agent = load_agent(args.model)
    state = build_state(args)
    recommendations = agent.recommend(state, top_k=args.top_k, search_depth=args.search_depth)
    print(json.dumps(recommendations, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
