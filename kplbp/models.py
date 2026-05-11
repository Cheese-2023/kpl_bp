from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .schema import BPSample, BPState, HeroStats, LineupSample


ContextKey = tuple[str, ...]


def _logit(value: float) -> float:
    clipped = max(0.01, min(0.99, value))
    return math.log(clipped / (1 - clipped))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            pairs.append(tuple(sorted((first, second))))
    return pairs


def _add_counter_bonus(
    scores: dict[str, float],
    legal: set[str],
    counter: Counter[str] | None,
    weight: float,
    heroes: list[str],
) -> None:
    if not counter:
        return
    total = sum(counter.values())
    denominator = total + len(heroes)
    for hero in legal:
        scores[hero] += weight * ((counter[hero] + 1.0) / denominator)


def _context_to_json_key(key: ContextKey) -> str:
    return "\t".join(key)


def _json_key_to_context(key: str) -> ContextKey:
    return tuple(key.split("\t"))


@dataclass
class Evaluation:
    total: int
    top1: float
    top3: float
    top5: float
    legal_rate: float


class PolicyModel:
    """A supervised, context-aware frequency baseline for BP actions."""

    def __init__(
        self,
        heroes: Iterable[str],
        hero_stats: dict[str, HeroStats] | None = None,
    ) -> None:
        self.heroes = sorted(set(heroes))
        self.hero_stats = hero_stats or {}
        self.counts: dict[ContextKey, Counter[str]] = defaultdict(Counter)
        self.hero_priors: Counter[str] = Counter()
        self.team_pick_priors: dict[str, Counter[str]] = defaultdict(Counter)
        self.team_ban_priors: dict[str, Counter[str]] = defaultdict(Counter)
        self.target_ban_priors: dict[str, Counter[str]] = defaultdict(Counter)

    def _contexts(self, state: BPState) -> list[tuple[float, ContextKey]]:
        own_picks = state.own_picks()
        enemy_picks = state.enemy_picks()
        own_team = state.own_team()
        enemy_team = state.enemy_team()
        contexts: list[tuple[float, ContextKey]] = [
            (4.0, ("exact", state.action_type, str(state.order), str(state.camp))),
            (2.0, ("order", state.action_type, str(state.order))),
            (1.0, ("action", state.action_type)),
        ]
        if own_team:
            contexts.extend(
                [
                    (1.2, ("team", state.action_type, own_team)),
                    (0.9, ("team_order", state.action_type, own_team, str(state.order))),
                ]
            )
        if enemy_team and state.action_type == "ban":
            contexts.append((1.0, ("ban_vs_team", enemy_team, str(state.order))))
        for hero in own_picks:
            contexts.extend(
                [
                    (0.75, ("own", state.action_type, hero)),
                    (0.85, ("synergy", state.action_type, str(state.order), hero)),
                ]
            )
        for hero in enemy_picks:
            contexts.extend(
                [
                    (0.75, ("enemy", state.action_type, hero)),
                    (0.85, ("counter", state.action_type, str(state.order), hero)),
                ]
            )
        for hero in state.banned:
            contexts.append((0.25, ("after_ban", state.action_type, hero)))
        for first, second in _pairs(own_picks):
            contexts.append((0.45, ("own_pair", state.action_type, first, second)))
        for first, second in _pairs(enemy_picks):
            contexts.append((0.45, ("enemy_pair", state.action_type, first, second)))
        contexts.append(
            (
                0.8,
                (
                    "shape",
                    state.action_type,
                    str(len(own_picks)),
                    str(len(enemy_picks)),
                    str(len(state.banned)),
                ),
            )
        )
        return contexts

    def fit(self, samples: Iterable[BPSample]) -> "PolicyModel":
        for sample in samples:
            self.hero_priors[sample.label] += 1
            own_team = sample.state.own_team()
            enemy_team = sample.state.enemy_team()
            if own_team and sample.state.action_type == "pick":
                self.team_pick_priors[own_team][sample.label] += 1
            if own_team and sample.state.action_type == "ban":
                self.team_ban_priors[own_team][sample.label] += 1
            if enemy_team and sample.state.action_type == "ban":
                self.target_ban_priors[enemy_team][sample.label] += 1
            for _, context in self._contexts(sample.state):
                self.counts[context][sample.label] += 1
        return self

    def predict(
        self,
        state: BPState,
        top_k: int = 10,
        legal_heroes: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        legal = set(legal_heroes if legal_heroes is not None else self.heroes)
        legal -= state.used_heroes()
        if not legal:
            return []

        scores = {hero: 1e-6 for hero in legal}
        for hero in legal:
            prior_count = self.hero_priors[hero]
            stat = self.hero_stats.get(hero)
            if stat is not None:
                if state.action_type == "ban":
                    scores[hero] += 0.25 * stat.ban_rate + 0.05 * stat.pick_rate
                else:
                    scores[hero] += 0.25 * stat.pick_rate + 0.10 * stat.win_rate
            scores[hero] += math.log1p(prior_count) * 0.02

        own_team = state.own_team()
        enemy_team = state.enemy_team()
        if state.action_type == "pick" and own_team:
            _add_counter_bonus(scores, legal, self.team_pick_priors.get(own_team), 0.45, self.heroes)
        if state.action_type == "ban":
            if own_team:
                _add_counter_bonus(scores, legal, self.team_ban_priors.get(own_team), 0.25, self.heroes)
            if enemy_team:
                _add_counter_bonus(scores, legal, self.target_ban_priors.get(enemy_team), 0.55, self.heroes)

        for weight, context in self._contexts(state):
            counter = self.counts.get(context)
            if not counter:
                continue
            total = sum(counter.values())
            denominator = total + len(self.heroes)
            for hero in legal:
                scores[hero] += weight * ((counter[hero] + 1.0) / denominator)

        total_score = sum(scores.values())
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(hero, score / total_score) for hero, score in ranked[:top_k]]

    def evaluate(self, samples: Iterable[BPSample], top_k: int = 5) -> Evaluation:
        total = 0
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        legal_hits = 0
        for sample in samples:
            total += 1
            ranked = self.predict(sample.state, top_k=max(top_k, 5))
            heroes = [hero for hero, _ in ranked]
            if sample.label not in sample.state.used_heroes():
                legal_hits += 1
            if heroes[:1] and sample.label == heroes[0]:
                top1_hits += 1
            if sample.label in heroes[:3]:
                top3_hits += 1
            if sample.label in heroes[:5]:
                top5_hits += 1
        if total == 0:
            return Evaluation(total=0, top1=0.0, top3=0.0, top5=0.0, legal_rate=0.0)
        return Evaluation(
            total=total,
            top1=top1_hits / total,
            top3=top3_hits / total,
            top5=top5_hits / total,
            legal_rate=legal_hits / total,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "heroes": self.heroes,
            "counts": {
                _context_to_json_key(key): dict(counter)
                for key, counter in self.counts.items()
            },
            "hero_priors": dict(self.hero_priors),
            "team_pick_priors": {
                team: dict(counter) for team, counter in self.team_pick_priors.items()
            },
            "team_ban_priors": {
                team: dict(counter) for team, counter in self.team_ban_priors.items()
            },
            "target_ban_priors": {
                team: dict(counter) for team, counter in self.target_ban_priors.items()
            },
            "hero_stats": {
                hero: {
                    "league_id": stat.league_id,
                    "hero_name": stat.hero_name,
                    "battle_count": stat.battle_count,
                    "win_rate": stat.win_rate,
                    "avg_kda": stat.avg_kda,
                    "ban_rate": stat.ban_rate,
                    "pick_rate": stat.pick_rate,
                }
                for hero, stat in self.hero_stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PolicyModel":
        raw_stats = data.get("hero_stats", {})
        hero_stats = {
            hero: HeroStats(
                league_id=str(value["league_id"]),
                hero_name=str(value["hero_name"]),
                battle_count=int(value["battle_count"]),
                win_rate=float(value["win_rate"]),
                avg_kda=float(value["avg_kda"]),
                ban_rate=float(value["ban_rate"]),
                pick_rate=float(value["pick_rate"]),
            )
            for hero, value in raw_stats.items()
        }
        model = cls(heroes=data["heroes"], hero_stats=hero_stats)
        for key, counter in data.get("counts", {}).items():
            model.counts[_json_key_to_context(key)] = Counter(counter)
        model.hero_priors = Counter(data.get("hero_priors", {}))
        model.team_pick_priors = defaultdict(
            Counter,
            {team: Counter(counter) for team, counter in data.get("team_pick_priors", {}).items()},
        )
        model.team_ban_priors = defaultdict(
            Counter,
            {team: Counter(counter) for team, counter in data.get("team_ban_priors", {}).items()},
        )
        model.target_ban_priors = defaultdict(
            Counter,
            {team: Counter(counter) for team, counter in data.get("target_ban_priors", {}).items()},
        )
        return model


class ValueModel:
    """A small empirical win-rate model for evaluating partial or full lineups."""

    def __init__(self, heroes: Iterable[str], hero_stats: dict[str, HeroStats] | None = None) -> None:
        self.heroes = sorted(set(heroes))
        self.hero_stats = hero_stats or {}
        self.base_rate = 0.5
        self.own_rates: dict[str, float] = {}
        self.enemy_rates: dict[str, float] = {}
        self.hero_effects: dict[str, float] = {}
        self.team_strengths: dict[str, float] = {}
        self.player_strengths: dict[str, float] = {}
        self.residual_hero_effects: dict[str, float] = {}
        self.pair_effects: dict[str, float] = {}
        self.matchup_effects: dict[str, float] = {}

    def fit(self, samples: Iterable[LineupSample]) -> "ValueModel":
        sample_list = list(samples)
        if not sample_list:
            return self
        self.base_rate = sum(sample.win for sample in sample_list) / len(sample_list)
        self._fit_team_and_player_strengths(sample_list)
        own_counts: dict[str, list[int]] = defaultdict(list)
        enemy_counts: dict[str, list[int]] = defaultdict(list)
        residuals_by_hero: dict[str, list[float]] = defaultdict(list)
        residuals_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
        residuals_by_matchup: dict[tuple[str, str], list[float]] = defaultdict(list)
        for sample in sample_list:
            expected_by_strength = self._strength_only_prediction(sample.team, sample.enemy_team)
            residual = sample.win - expected_by_strength
            own_picks = list(sample.own_picks)
            enemy_picks = list(sample.enemy_picks)
            for hero in sample.own_picks:
                own_counts[hero].append(sample.win)
                residuals_by_hero[hero].append(residual)
            for hero in sample.enemy_picks:
                enemy_counts[hero].append(sample.win)
                residuals_by_hero[hero].append(-residual)
            for pair in _pairs(own_picks):
                residuals_by_pair[pair].append(residual)
            for pair in _pairs(enemy_picks):
                residuals_by_pair[pair].append(-residual)
            for own_hero in own_picks:
                for enemy_hero in enemy_picks:
                    residuals_by_matchup[(own_hero, enemy_hero)].append(residual)
                    residuals_by_matchup[(enemy_hero, own_hero)].append(-residual)
        for hero in self.heroes:
            stat = self.hero_stats.get(hero)
            stat_prior = stat.win_rate if stat is not None else self.base_rate
            own = own_counts.get(hero, [])
            enemy = enemy_counts.get(hero, [])
            self.own_rates[hero] = (sum(own) + 4 * stat_prior) / (len(own) + 4)
            self.enemy_rates[hero] = (sum(enemy) + 4 * (1 - stat_prior)) / (len(enemy) + 4)
            empirical_effect = _logit(self.own_rates[hero]) - _logit(self.base_rate)
            residuals = residuals_by_hero.get(hero, [])
            residual_effect = sum(residuals) / (len(residuals) + 8) if residuals else 0.0
            stat_effect = _logit(stat_prior) - _logit(0.5) if stat is not None else 0.0
            self.residual_hero_effects[hero] = residual_effect
            self.hero_effects[hero] = 0.45 * empirical_effect + 0.40 * residual_effect + 0.15 * stat_effect
        self.pair_effects = {
            _context_to_json_key(pair): sum(values) / (len(values) + 10)
            for pair, values in residuals_by_pair.items()
            if len(values) >= 6
        }
        self.matchup_effects = {
            _context_to_json_key(matchup): sum(values) / (len(values) + 12)
            for matchup, values in residuals_by_matchup.items()
            if len(values) >= 6
        }
        return self

    def _fit_team_and_player_strengths(self, samples: list[LineupSample]) -> None:
        team_results: dict[str, list[int]] = defaultdict(list)
        player_results: dict[str, list[int]] = defaultdict(list)
        for sample in samples:
            if sample.team:
                team_results[sample.team].append(sample.win)
            if sample.enemy_team:
                team_results[sample.enemy_team].append(1 - sample.win)
            for player in sample.players:
                player_results[player].append(sample.win)
            for player in sample.enemy_players:
                player_results[player].append(1 - sample.win)
        self.team_strengths = {
            team: _logit((sum(results) + 3 * self.base_rate) / (len(results) + 3))
            - _logit(self.base_rate)
            for team, results in team_results.items()
        }
        self.player_strengths = {
            player: _logit((sum(results) + 2 * self.base_rate) / (len(results) + 2))
            - _logit(self.base_rate)
            for player, results in player_results.items()
        }

    def _strength_only_prediction(
        self,
        team: str | None,
        enemy_team: str | None,
        players: Iterable[str] = (),
        enemy_players: Iterable[str] = (),
    ) -> float:
        score = _logit(self.base_rate)
        if team:
            score += 0.65 * self.team_strengths.get(team, 0.0)
        if enemy_team:
            score -= 0.65 * self.team_strengths.get(enemy_team, 0.0)
        own_player_strength = sum(self.player_strengths.get(player, 0.0) for player in players)
        enemy_player_strength = sum(
            self.player_strengths.get(player, 0.0) for player in enemy_players
        )
        score += 0.08 * (own_player_strength - enemy_player_strength)
        return _sigmoid(score)

    def predict(
        self,
        own_picks: Iterable[str],
        enemy_picks: Iterable[str],
        team: str | None = None,
        enemy_team: str | None = None,
        players: Iterable[str] = (),
        enemy_players: Iterable[str] = (),
        include_strength: bool = False,
    ) -> float:
        own = list(own_picks)
        enemy = list(enemy_picks)
        scale = max(1, len(own) + len(enemy))
        if include_strength:
            score = _logit(
                self._strength_only_prediction(team, enemy_team, players, enemy_players)
            )
        else:
            score = _logit(0.5)
        score += sum(self.hero_effects.get(hero, 0.0) for hero in own) / scale
        score -= sum(self.hero_effects.get(hero, 0.0) for hero in enemy) / scale
        # Pair and matchup effects are learned and exported for analysis, but not
        # used by default because the current dataset is too small for them to
        # generalize reliably.
        return _sigmoid(score)

    def _pair_score(self, heroes: list[str]) -> float:
        return sum(
            self.pair_effects.get(_context_to_json_key(pair), 0.0)
            for pair in _pairs(heroes)
        )

    def _matchup_score(self, own: list[str], enemy: list[str]) -> float:
        score = 0.0
        for own_hero in own:
            for enemy_hero in enemy:
                score += self.matchup_effects.get(
                    _context_to_json_key((own_hero, enemy_hero)),
                    0.0,
                )
        return score

    def to_dict(self) -> dict[str, object]:
        return {
            "heroes": self.heroes,
            "base_rate": self.base_rate,
            "own_rates": self.own_rates,
            "enemy_rates": self.enemy_rates,
            "hero_effects": self.hero_effects,
            "team_strengths": self.team_strengths,
            "player_strengths": self.player_strengths,
            "residual_hero_effects": self.residual_hero_effects,
            "pair_effects": self.pair_effects,
            "matchup_effects": self.matchup_effects,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ValueModel":
        model = cls(heroes=data["heroes"])
        model.base_rate = float(data.get("base_rate", 0.5))
        model.own_rates = {hero: float(value) for hero, value in data.get("own_rates", {}).items()}
        model.enemy_rates = {
            hero: float(value) for hero, value in data.get("enemy_rates", {}).items()
        }
        model.hero_effects = {
            hero: float(value) for hero, value in data.get("hero_effects", {}).items()
        }
        model.team_strengths = {
            team: float(value) for team, value in data.get("team_strengths", {}).items()
        }
        model.player_strengths = {
            player: float(value) for player, value in data.get("player_strengths", {}).items()
        }
        model.residual_hero_effects = {
            hero: float(value) for hero, value in data.get("residual_hero_effects", {}).items()
        }
        model.pair_effects = {
            key: float(value) for key, value in data.get("pair_effects", {}).items()
        }
        model.matchup_effects = {
            key: float(value) for key, value in data.get("matchup_effects", {}).items()
        }
        return model


def value_accuracy(
    model: ValueModel,
    samples: Iterable[LineupSample],
    include_strength: bool = False,
) -> dict[str, float]:
    total = 0
    correct = 0
    brier = 0.0
    for sample in samples:
        total += 1
        probability = model.predict(
            sample.own_picks,
            sample.enemy_picks,
            sample.team,
            sample.enemy_team,
            sample.players,
            sample.enemy_players,
            include_strength=include_strength,
        )
        correct += int((probability >= 0.5) == bool(sample.win))
        brier += (probability - sample.win) ** 2
    if total == 0:
        return {"total": 0, "accuracy": 0.0, "brier": 0.0}
    return {"total": total, "accuracy": correct / total, "brier": brier / total}
